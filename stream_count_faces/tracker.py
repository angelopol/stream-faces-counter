"""
tracker.py - Rastreo de rostros para deduplicación de pasajeros usando collage

Proporciona la clase FaceTracker para evitar contar la misma persona
múltiples veces dentro de un período de tiempo configurable.

Usa un COLLAGE de rostros para hacer una única llamada a compare_faces
en lugar de comparar contra cada rostro individualmente.
"""

import os
import cv2
import uuid
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """
    Representa un rostro rastreado.
    """
    face_id: str
    first_seen: datetime
    last_seen: datetime
    position_in_collage: int  # Posición del rostro en el collage
    match_count: int = 1
    
    def is_expired(self, ttl_minutes: int) -> bool:
        """Verifica si el registro ha expirado."""
        expiry_time = self.first_seen + timedelta(minutes=ttl_minutes)
        return datetime.now() > expiry_time


class FaceTracker:
    """
    Rastrea rostros usando un COLLAGE para evitar conteos duplicados.
    
    En lugar de comparar contra cada rostro individualmente, mantiene
    un collage (imagen horizontal) con todos los rostros detectados.
    Una única llamada a compare_faces compara el nuevo rostro contra
    todos los rostros del collage simultáneamente.
    
    Ventajas del enfoque de collage:
    - Una sola llamada API por rostro nuevo (vs N llamadas)
    - Más eficiente en costos de AWS
    - Mismo resultado de deduplicación
    
    Atributos:
        ttl_minutes: Tiempo en minutos antes de expirar un registro (default: 180 = 3 horas)
        similarity_threshold: Umbral de similitud para considerar coincidencia (default: 80%)
        max_faces: Número máximo de rostros en collage
        
    Ejemplo:
        >>> tracker = FaceTracker(ttl_minutes=180)
        >>> is_new, face_id = tracker.is_new_passenger(face_image_bytes)
        >>> if is_new:
        ...     print("Nuevo pasajero detectado")
    """
    
    # Tamaño estándar para cada rostro en el collage
    FACE_SIZE = (100, 100)  # width, height
    
    # Número máximo de rostros por collage antes de crear uno nuevo
    MAX_FACES_PER_COLLAGE = 100
    
    def __init__(
        self,
        ttl_minutes: int = 180,  # 3 horas por defecto
        similarity_threshold: float = 80.0,
        max_faces: int = 500,
        dry_run: bool = False,
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Inicializa el rastreador de rostros.
        
        Args:
            ttl_minutes: Minutos antes de expirar un registro (default: 180 = 3 horas)
            similarity_threshold: Umbral de similitud % (default: 80.0)
            max_faces: Máximo de rostros en cache (default: 500)
            dry_run: Si True, no hace llamadas a AWS
            region: Región AWS (default: env o us-east-1)
            aws_access_key_id: AWS Access Key ID (opcional)
            aws_secret_access_key: AWS Secret Access Key (opcional)
            aws_session_token: AWS Session Token (opcional)
        """
        self.ttl_minutes = ttl_minutes
        self.similarity_threshold = similarity_threshold
        self.max_faces = max_faces
        self.dry_run = dry_run
        
        # AWS Configuration
        self.region = region or os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self._aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self._aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self._aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        
        # Collages de rostros (lista de imágenes numpy)
        self._collages: List[np.ndarray] = []
        self._current_collage_faces = 0  # Rostros en el collage actual
        
        # Metadatos de rostros rastreados
        self._tracked_faces: Dict[str, TrackedFace] = {}
        self._lock = threading.Lock()
        
        # Estadísticas
        self._total_comparisons = 0
        self._total_api_calls = 0
        self._total_new_passengers = 0
        self._total_duplicates = 0
        self._last_cleanup = datetime.now()
        
        # Cliente AWS
        self._client = None
        if not dry_run:
            self._init_client()
        
        logger.info(
            f"FaceTracker (collage mode) inicializado: ttl={ttl_minutes}min, "
            f"similarity={similarity_threshold}%, max_faces={max_faces}, "
            f"dry_run={dry_run}"
        )
    
    def _init_client(self) -> bool:
        """Inicializa el cliente de AWS Rekognition."""
        try:
            client_kwargs = {"region_name": self.region}
            
            if self._aws_access_key_id and self._aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = self._aws_access_key_id
                client_kwargs["aws_secret_access_key"] = self._aws_secret_access_key
                if self._aws_session_token:
                    client_kwargs["aws_session_token"] = self._aws_session_token
            
            self._client = boto3.client("rekognition", **client_kwargs)
            logger.debug("Cliente Rekognition para tracking inicializado")
            return True
        except NoCredentialsError:
            logger.error("No se encontraron credenciales AWS para tracking")
            return False
        except Exception as e:
            logger.error(f"Error inicializando cliente tracking: {e}")
            return False
    
    def _resize_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Redimensiona un rostro al tamaño estándar del collage.
        
        Args:
            face_img: Imagen del rostro (BGR)
            
        Returns:
            Imagen redimensionada
        """
        return cv2.resize(face_img, self.FACE_SIZE, interpolation=cv2.INTER_AREA)
    
    def _add_face_to_collage(self, face_img: np.ndarray) -> int:
        """
        Agrega un rostro al collage actual.
        
        Args:
            face_img: Imagen del rostro (BGR)
            
        Returns:
            Índice del collage donde se agregó
        """
        resized = self._resize_face(face_img)
        
        # Crear nuevo collage si es necesario
        if not self._collages or self._current_collage_faces >= self.MAX_FACES_PER_COLLAGE:
            self._collages.append(resized)
            self._current_collage_faces = 1
            return len(self._collages) - 1
        
        # Agregar al collage actual (horizontalmente)
        current_idx = len(self._collages) - 1
        current_collage = self._collages[current_idx]
        
        # Asegurar misma altura
        if resized.shape[0] != current_collage.shape[0]:
            resized = cv2.resize(resized, (self.FACE_SIZE[0], current_collage.shape[0]))
        
        self._collages[current_idx] = np.hstack([current_collage, resized])
        self._current_collage_faces += 1
        
        return current_idx
    
    def _get_collage_bytes(self, collage_idx: int) -> bytes:
        """
        Obtiene el collage como bytes JPEG.
        
        Args:
            collage_idx: Índice del collage
            
        Returns:
            Collage codificado como JPEG bytes
        """
        if collage_idx >= len(self._collages):
            return b""
        
        _, buffer = cv2.imencode('.jpg', self._collages[collage_idx], [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def is_new_passenger(
        self, 
        face_image: bytes,
        frame: Optional[np.ndarray] = None,
        bounding_box: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Determina si un rostro corresponde a un nuevo pasajero.
        
        Compara el rostro contra TODOS los collages existentes con
        una única llamada compare_faces por collage. Si encuentra
        coincidencia, retorna False. Si no hay coincidencia en
        ningún collage, agrega el rostro y retorna True.
        
        Args:
            face_image: Imagen del rostro en bytes (JPEG)
            frame: Frame original (opcional)
            bounding_box: Bounding box del rostro (opcional)
            
        Returns:
            Tupla (is_new, face_id):
                - is_new: True si es un nuevo pasajero
                - face_id: ID del rostro (nuevo o existente)
        """
        self._total_comparisons += 1
        
        # Limpiar cache periódicamente
        self._maybe_cleanup()
        
        if self.dry_run:
            return self._simulate_comparison(face_image)
        
        if self._client is None:
            if not self._init_client():
                return self._add_new_face(face_image)
        
        with self._lock:
            # Si no hay collages, es definitivamente nuevo
            if not self._collages:
                return self._add_new_face(face_image)
            
            # Comparar contra cada collage (una llamada API por collage)
            for collage_idx, collage in enumerate(self._collages):
                try:
                    collage_bytes = self._get_collage_bytes(collage_idx)
                    if not collage_bytes:
                        continue
                    
                    self._total_api_calls += 1
                    
                    response = self._client.compare_faces(
                        SourceImage={"Bytes": face_image},
                        TargetImage={"Bytes": collage_bytes},
                        SimilarityThreshold=self.similarity_threshold
                    )
                    
                    matches = response.get("FaceMatches", [])
                    if matches:
                        # Encontramos coincidencia
                        similarity = matches[0].get("Similarity", 0)
                        self._total_duplicates += 1
                        
                        # Buscar el face_id correspondiente (aproximado por posición)
                        matched_face_id = self._find_face_id_by_collage(collage_idx)
                        
                        logger.debug(
                            f"Pasajero duplicado: collage={collage_idx}, "
                            f"similarity={similarity:.1f}%, face_id={matched_face_id}"
                        )
                        return False, matched_face_id
                        
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'InvalidParameterException':
                        # No se detectó rostro, continuar con siguiente collage
                        continue
                    logger.warning(f"Error comparando con collage {collage_idx}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error en compare_faces: {e}")
                    continue
            
            # No se encontró en ningún collage, es nuevo
            return self._add_new_face(face_image)
    
    def _find_face_id_by_collage(self, collage_idx: int) -> Optional[str]:
        """
        Encuentra un face_id asociado a un collage.
        
        Args:
            collage_idx: Índice del collage
            
        Returns:
            face_id si se encuentra, None si no
        """
        for face_id, tracked in self._tracked_faces.items():
            if tracked.position_in_collage == collage_idx:
                return face_id
        return None
    
    def _add_new_face(self, face_image: bytes) -> Tuple[bool, str]:
        """
        Agrega un nuevo rostro al sistema.
        
        Args:
            face_image: Imagen del rostro en bytes
            
        Returns:
            Tupla (True, face_id) indicando nuevo pasajero
        """
        # Verificar límite de cache
        if len(self._tracked_faces) >= self.max_faces:
            self._evict_oldest()
        
        face_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        collage_idx = -1  # Por defecto, sin collage
        
        # Decodificar imagen (solo si no estamos en dry_run o si es imagen válida)
        if not self.dry_run:
            nparr = np.frombuffer(face_image, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if face_img is not None:
                # Agregar al collage
                collage_idx = self._add_face_to_collage(face_img)
            else:
                logger.warning("No se pudo decodificar imagen del rostro")
        
        tracked_face = TrackedFace(
            face_id=face_id,
            first_seen=now,
            last_seen=now,
            position_in_collage=collage_idx
        )
        
        self._tracked_faces[face_id] = tracked_face
        self._total_new_passengers += 1
        
        logger.debug(f"Nuevo pasajero: {face_id}, collage={collage_idx}")
        return True, face_id
    
    def _simulate_comparison(self, face_image: bytes) -> Tuple[bool, Optional[str]]:
        """
        Simula comparación para modo dry_run.
        
        En dry_run, siempre considera los rostros como nuevos
        pero los rastrea correctamente.
        """
        return self._add_new_face(face_image)
    
    def _maybe_cleanup(self) -> None:
        """Limpia cache si ha pasado suficiente tiempo."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() > 300:  # 5 minutos
            self.cleanup_expired()
            self._last_cleanup = now
    
    def cleanup_expired(self) -> int:
        """
        Elimina rostros expirados.
        
        Nota: Los rostros en el collage permanecen pero sus metadatos
        se eliminan. El collage se reconstruye periódicamente.
        
        Returns:
            Número de rostros eliminados
        """
        with self._lock:
            expired_ids = [
                face_id for face_id, face in self._tracked_faces.items()
                if face.is_expired(self.ttl_minutes)
            ]
            
            for face_id in expired_ids:
                del self._tracked_faces[face_id]
            
            # Si todos los rostros de un collage expiraron, reconstruir
            if expired_ids and len(self._tracked_faces) == 0:
                self._collages.clear()
                self._current_collage_faces = 0
                logger.info("Todos los rostros expiraron, collages limpiados")
            
            if expired_ids:
                logger.info(f"Limpieza: {len(expired_ids)} rostros expirados")
            
            return len(expired_ids)
    
    def _evict_oldest(self) -> None:
        """Elimina el rostro más antiguo."""
        if not self._tracked_faces:
            return
        
        oldest_id = min(
            self._tracked_faces.keys(),
            key=lambda k: self._tracked_faces[k].first_seen
        )
        
        del self._tracked_faces[oldest_id]
        logger.debug(f"Rostro más antiguo eliminado: {oldest_id}")
    
    def clear(self) -> int:
        """
        Limpia todo el cache.
        
        Returns:
            Número de rostros eliminados
        """
        with self._lock:
            count = len(self._tracked_faces)
            self._tracked_faces.clear()
            self._collages.clear()
            self._current_collage_faces = 0
            logger.info(f"Cache limpiado: {count} rostros")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del rastreador.
        
        Returns:
            Diccionario con estadísticas
        """
        with self._lock:
            active_faces = sum(
                1 for f in self._tracked_faces.values()
                if not f.is_expired(self.ttl_minutes)
            )
            
            return {
                "tracked_faces": len(self._tracked_faces),
                "active_faces": active_faces,
                "collages": len(self._collages),
                "faces_in_current_collage": self._current_collage_faces,
                "total_comparisons": self._total_comparisons,
                "total_api_calls": self._total_api_calls,
                "new_passengers": self._total_new_passengers,
                "duplicates": self._total_duplicates,
                "duplicate_rate": (
                    self._total_duplicates / self._total_comparisons * 100
                    if self._total_comparisons > 0 else 0
                ),
                "api_efficiency": (
                    f"{self._total_api_calls} calls for {self._total_comparisons} comparisons"
                ),
                "ttl_minutes": self.ttl_minutes,
                "similarity_threshold": self.similarity_threshold,
                "max_faces": self.max_faces,
                "dry_run": self.dry_run
            }


def extract_face_image(
    frame: np.ndarray,
    bounding_box: Dict[str, float],
    padding: float = 0.1
) -> bytes:
    """
    Extrae una imagen de rostro de un frame dado el bounding box.
    
    Args:
        frame: Frame BGR de OpenCV
        bounding_box: Dict con Left, Top, Width, Height (normalizados 0-1)
        padding: Padding adicional alrededor del rostro (default: 10%)
        
    Returns:
        Imagen del rostro codificada como JPEG bytes
    """
    height, width = frame.shape[:2]
    
    # Convertir coordenadas normalizadas a píxeles
    left = int(bounding_box['Left'] * width)
    top = int(bounding_box['Top'] * height)
    box_width = int(bounding_box['Width'] * width)
    box_height = int(bounding_box['Height'] * height)
    
    # Agregar padding
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    
    # Calcular coordenadas con padding (clipping a límites)
    x1 = max(0, left - pad_x)
    y1 = max(0, top - pad_y)
    x2 = min(width, left + box_width + pad_x)
    y2 = min(height, top + box_height + pad_y)
    
    # Recortar rostro
    face_img = frame[y1:y2, x1:x2]
    
    # Codificar como JPEG
    _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Probando FaceTracker (collage mode) en dry_run...")
    
    tracker = FaceTracker(dry_run=True, ttl_minutes=5)
    
    # Simular detección
    fake_face_1 = b"fake_face_image_1"
    fake_face_2 = b"fake_face_image_2"
    
    is_new, face_id = tracker.is_new_passenger(fake_face_1)
    print(f"Rostro 1: is_new={is_new}, face_id={face_id}")
    
    is_new, face_id = tracker.is_new_passenger(fake_face_2)
    print(f"Rostro 2: is_new={is_new}, face_id={face_id}")
    
    stats = tracker.get_stats()
    print(f"Stats: {stats}")
    
    print("\nTest completado")
