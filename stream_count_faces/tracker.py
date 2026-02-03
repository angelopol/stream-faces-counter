"""
tracker.py - Rastreo de rostros para deduplicación de pasajeros usando collage

Proporciona la clase FaceTracker para evitar contar la misma persona
múltiples veces dentro de un período de tiempo configurable.

Características:
- Usa COLLAGE de rostros para hacer una única llamada a compare_faces
- Soporta EXCLUSIÓN de personal autorizado (no cuentan como pasajeros)
- TTL configurable para "olvidar" pasajeros después de cierto tiempo
"""

import os
import cv2
import uuid
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from pathlib import Path

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
    Rastrea rostros usando COLLAGES para evitar conteos duplicados.
    
    Características:
    - Mantiene collages de pasajeros detectados (con TTL)
    - Soporta EXCLUSIÓN de personal autorizado (sin TTL)
    - Una única llamada API compara contra AMBOS collages
    
    Ejemplo:
        >>> # Fotos del personal (operador, conductor)
        >>> staff_photos = ["conductor.jpg", "operador.jpg"]
        >>> 
        >>> tracker = FaceTracker(
        ...     ttl_minutes=180,
        ...     excluded_faces=staff_photos
        ... )
        >>> 
        >>> is_new, face_id, is_excluded = tracker.is_new_passenger(face_bytes)
        >>> if is_excluded:
        ...     print("Personal autorizado detectado")
        >>> elif is_new:
        ...     print("Nuevo pasajero")
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
        excluded_faces: Optional[List[Union[str, bytes, np.ndarray]]] = None,
        offline_cache_path: Optional[str] = None,
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
            excluded_faces: Lista de rostros a excluir (personal autorizado).
                           Puede ser: rutas a imágenes, bytes, o arrays numpy.
                           Estos rostros NO cuentan como pasajeros.
            offline_cache_path: Ruta para cache offline de rostros (SQLite).
                               Si se provee, los rostros se almacenan cuando
                               AWS no está disponible para procesamiento posterior.
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
        
        # Collages de PASAJEROS (con TTL)
        self._collages: List[np.ndarray] = []
        self._current_collage_faces = 0
        
        # Collage de PERSONAL EXCLUIDO (sin TTL, permanente)
        self._excluded_collage: Optional[np.ndarray] = None
        self._excluded_faces_count = 0
        
        # Metadatos de rostros rastreados
        self._tracked_faces: Dict[str, TrackedFace] = {}
        self._lock = threading.Lock()
        
        # Estadísticas
        self._total_comparisons = 0
        self._total_api_calls = 0
        self._total_new_passengers = 0
        self._total_duplicates = 0
        self._total_excluded_detected = 0
        self._total_offline_cached = 0
        self._last_cleanup = datetime.now()
        
        # Cliente AWS
        self._client = None
        self._is_offline = False
        if not dry_run:
            self._init_client()
        
        # Cache offline (cuando AWS no está disponible)
        self._offline_cache = None
        if offline_cache_path:
            from .storage import FaceCache
            self._offline_cache = FaceCache(offline_cache_path)
            logger.info(f"Offline cache habilitado: {offline_cache_path}")
        
        # Cargar rostros excluidos
        if excluded_faces:
            self._load_excluded_faces(excluded_faces)
        
        logger.info(
            f"FaceTracker inicializado: ttl={ttl_minutes}min, "
            f"similarity={similarity_threshold}%, max_faces={max_faces}, "
            f"excluded_faces={self._excluded_faces_count}, dry_run={dry_run}"
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
    
    def _load_excluded_faces(self, faces: List[Union[str, bytes, np.ndarray]]) -> int:
        """
        Carga rostros excluidos (personal autorizado).
        
        Args:
            faces: Lista de rostros (rutas, bytes o arrays numpy)
            
        Returns:
            Número de rostros cargados exitosamente
        """
        loaded = 0
        for face in faces:
            try:
                face_img = self._load_face_image(face)
                if face_img is not None:
                    self._add_to_excluded_collage(face_img)
                    loaded += 1
                    logger.debug(f"Rostro excluido cargado: {face if isinstance(face, str) else 'bytes/array'}")
            except Exception as e:
                logger.warning(f"Error cargando rostro excluido: {e}")
        
        self._excluded_faces_count = loaded
        logger.info(f"Rostros excluidos cargados: {loaded}")
        return loaded
    
    def _load_face_image(self, source: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """
        Carga una imagen de rostro desde diferentes fuentes.
        
        Args:
            source: Ruta a imagen, bytes, o array numpy
            
        Returns:
            Imagen BGR numpy array, o None si falla
        """
        if isinstance(source, np.ndarray):
            return source
        
        if isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if isinstance(source, str):
            path = Path(source)
            if path.exists():
                return cv2.imread(str(path))
            else:
                logger.warning(f"Archivo no encontrado: {source}")
                return None
        
        return None
    
    def _add_to_excluded_collage(self, face_img: np.ndarray) -> None:
        """
        Agrega un rostro al collage de excluidos.
        
        Args:
            face_img: Imagen del rostro (BGR)
        """
        resized = self._resize_face(face_img)
        
        if self._excluded_collage is None:
            self._excluded_collage = resized
        else:
            # Asegurar misma altura
            if resized.shape[0] != self._excluded_collage.shape[0]:
                resized = cv2.resize(resized, (self.FACE_SIZE[0], self._excluded_collage.shape[0]))
            self._excluded_collage = np.hstack([self._excluded_collage, resized])
    
    def add_excluded_face(self, face: Union[str, bytes, np.ndarray]) -> bool:
        """
        Agrega un nuevo rostro a la lista de excluidos en tiempo de ejecución.
        
        Args:
            face: Ruta a imagen, bytes, o array numpy
            
        Returns:
            True si se agregó exitosamente
        """
        with self._lock:
            face_img = self._load_face_image(face)
            if face_img is not None:
                self._add_to_excluded_collage(face_img)
                self._excluded_faces_count += 1
                logger.info(f"Rostro excluido agregado. Total: {self._excluded_faces_count}")
                return True
            return False
    
    def _resize_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Redimensiona un rostro al tamaño estándar del collage.
        """
        return cv2.resize(face_img, self.FACE_SIZE, interpolation=cv2.INTER_AREA)
    
    def _add_face_to_collage(self, face_img: np.ndarray) -> int:
        """
        Agrega un rostro al collage de pasajeros.
        
        Returns:
            Índice del collage donde se agregó
        """
        resized = self._resize_face(face_img)
        
        if not self._collages or self._current_collage_faces >= self.MAX_FACES_PER_COLLAGE:
            self._collages.append(resized)
            self._current_collage_faces = 1
            return len(self._collages) - 1
        
        current_idx = len(self._collages) - 1
        current_collage = self._collages[current_idx]
        
        if resized.shape[0] != current_collage.shape[0]:
            resized = cv2.resize(resized, (self.FACE_SIZE[0], current_collage.shape[0]))
        
        self._collages[current_idx] = np.hstack([current_collage, resized])
        self._current_collage_faces += 1
        
        return current_idx
    
    def _build_combined_collage(self, passenger_collage_idx: int) -> Tuple[bytes, int]:
        """
        Construye un collage combinado: excluidos + pasajeros.
        
        El orden es importante: primero excluidos, luego pasajeros.
        Esto permite determinar si la coincidencia es con personal excluido
        basándose en la posición del rostro coincidente.
        
        Args:
            passenger_collage_idx: Índice del collage de pasajeros
            
        Returns:
            Tupla (collage_bytes, excluded_width):
                - collage_bytes: Collage combinado en JPEG bytes
                - excluded_width: Ancho del collage de excluidos (para determinar coincidencias)
        """
        excluded_width = 0
        
        passenger_collage = self._collages[passenger_collage_idx]
        
        if self._excluded_collage is not None:
            excluded_width = self._excluded_collage.shape[1]
            
            # Asegurar misma altura
            excluded = self._excluded_collage
            if excluded.shape[0] != passenger_collage.shape[0]:
                scale = passenger_collage.shape[0] / excluded.shape[0]
                new_width = int(excluded.shape[1] * scale)
                excluded = cv2.resize(excluded, (new_width, passenger_collage.shape[0]))
                excluded_width = new_width
            
            # Combinar: excluidos a la izquierda, pasajeros a la derecha
            combined = np.hstack([excluded, passenger_collage])
        else:
            combined = passenger_collage
        
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes(), excluded_width
    
    def _get_excluded_only_bytes(self) -> Optional[bytes]:
        """
        Obtiene el collage de excluidos como bytes JPEG.
        
        Returns:
            Bytes JPEG del collage de excluidos, o None si no hay
        """
        if self._excluded_collage is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self._excluded_collage, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def is_new_passenger(
        self, 
        face_image: bytes,
        frame: Optional[np.ndarray] = None,
        bounding_box: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Optional[str], bool]:
        """
        Determina si un rostro corresponde a un nuevo pasajero.
        
        Compara el rostro contra:
        1. Collage de excluidos (personal autorizado)
        2. Collages de pasajeros anteriores
        
        Todo en UNA SOLA llamada API (collage combinado).
        
        Args:
            face_image: Imagen del rostro en bytes (JPEG)
            frame: Frame original (opcional)
            bounding_box: Bounding box del rostro (opcional)
            
        Returns:
            Tupla (is_new, face_id, is_excluded):
                - is_new: True si es un nuevo pasajero
                - face_id: ID del rostro (o None si es excluido)
                - is_excluded: True si es personal autorizado (no cuenta)
        """
        self._total_comparisons += 1
        
        self._maybe_cleanup()
        
        if self.dry_run:
            return self._simulate_comparison(face_image)
        
        if self._client is None:
            if not self._init_client():
                is_new, face_id = self._add_new_face(face_image)
                return is_new, face_id, False
        
        with self._lock:
            # Caso 1: Solo hay excluidos, no hay pasajeros previos
            if not self._collages and self._excluded_collage is not None:
                excluded_bytes = self._get_excluded_only_bytes()
                if excluded_bytes:
                    is_excluded = self._check_against_excluded(face_image, excluded_bytes)
                    if is_excluded:
                        self._total_excluded_detected += 1
                        logger.debug("Personal autorizado detectado (solo excluidos)")
                        return False, None, True
                
                # No es excluido, es nuevo pasajero
                is_new, face_id = self._add_new_face(face_image)
                return is_new, face_id, False
            
            # Caso 2: No hay collages de pasajeros ni excluidos
            if not self._collages:
                is_new, face_id = self._add_new_face(face_image)
                return is_new, face_id, False
            
            # Caso 3: Hay collages de pasajeros (y posiblemente excluidos)
            for collage_idx in range(len(self._collages)):
                try:
                    # Construir collage combinado
                    combined_bytes, excluded_width = self._build_combined_collage(collage_idx)
                    
                    self._total_api_calls += 1
                    
                    response = self._client.compare_faces(
                        SourceImage={"Bytes": face_image},
                        TargetImage={"Bytes": combined_bytes},
                        SimilarityThreshold=self.similarity_threshold
                    )
                    
                    matches = response.get("FaceMatches", [])
                    if matches:
                        # Determinar si la coincidencia es con excluido o pasajero
                        match = matches[0]
                        similarity = match.get("Similarity", 0)
                        face_bbox = match.get("Face", {}).get("BoundingBox", {})
                        
                        # La posición X del rostro coincidente determina si es excluido
                        match_left = face_bbox.get("Left", 0)
                        match_x_pixels = int(match_left * (excluded_width + self._collages[collage_idx].shape[1]))
                        
                        if excluded_width > 0 and match_x_pixels < excluded_width:
                            # Coincidencia con personal excluido
                            self._total_excluded_detected += 1
                            logger.debug(f"Personal autorizado detectado: similarity={similarity:.1f}%")
                            return False, None, True
                        else:
                            # Coincidencia con pasajero anterior
                            self._total_duplicates += 1
                            matched_face_id = self._find_face_id_by_collage(collage_idx)
                            logger.debug(f"Pasajero duplicado: similarity={similarity:.1f}%")
                            return False, matched_face_id, False
                        
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'InvalidParameterException':
                        continue
                    logger.warning(f"Error comparando con collage {collage_idx}: {e}")
                    # Store face offline for later processing
                    return self._handle_offline_storage(face_image, str(e))
                except Exception as e:
                    error_str = str(e)
                    # Detect connection errors (no internet)
                    if 'EndpointConnectionError' in error_str or 'ConnectionError' in error_str or 'Timeout' in error_str:
                        logger.warning(f"Sin conexión a AWS, almacenando rostro offline: {e}")
                        self._is_offline = True
                        return self._handle_offline_storage(face_image, error_str)
                    logger.warning(f"Error en compare_faces: {e}")
                    continue
            
            # No se encontró en ningún collage
            is_new, face_id = self._add_new_face(face_image)
            return is_new, face_id, False
    
    def _handle_offline_storage(
        self, 
        face_image: bytes, 
        error_message: str
    ) -> Tuple[bool, Optional[str], bool]:
        """
        Almacena un rostro en cache offline cuando AWS no está disponible.
        
        Args:
            face_image: Imagen del rostro
            error_message: Mensaje de error de la conexión
            
        Returns:
            Tupla indicando rostro pendiente (True, None, False) si se almacenó
        """
        if self._offline_cache:
            cache_id = self._offline_cache.store_pending(face_image)
            if cache_id:
                self._total_offline_cached += 1
                logger.debug(f"Rostro almacenado en cache offline: id={cache_id}")
            return True, None, False  # Considerar como nuevo por ahora
        else:
            # Sin cache offline, considerar como nuevo (comportamiento anterior)
            is_new, face_id = self._add_new_face(face_image)
            return is_new, face_id, False
    
    def _check_against_excluded(self, face_image: bytes, excluded_bytes: bytes) -> bool:
        """
        Verifica si un rostro coincide con algún excluido.
        
        Args:
            face_image: Rostro a verificar
            excluded_bytes: Collage de excluidos
            
        Returns:
            True si coincide con algún excluido
        """
        try:
            self._total_api_calls += 1
            response = self._client.compare_faces(
                SourceImage={"Bytes": face_image},
                TargetImage={"Bytes": excluded_bytes},
                SimilarityThreshold=self.similarity_threshold
            )
            return len(response.get("FaceMatches", [])) > 0
        except Exception as e:
            logger.warning(f"Error verificando excluidos: {e}")
            return False
    
    def _find_face_id_by_collage(self, collage_idx: int) -> Optional[str]:
        """Encuentra un face_id asociado a un collage."""
        for face_id, tracked in self._tracked_faces.items():
            if tracked.position_in_collage == collage_idx:
                return face_id
        return None
    
    def _add_new_face(self, face_image: bytes) -> Tuple[bool, str]:
        """Agrega un nuevo rostro al sistema."""
        if len(self._tracked_faces) >= self.max_faces:
            self._evict_oldest()
        
        face_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        collage_idx = -1
        
        if not self.dry_run:
            nparr = np.frombuffer(face_image, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if face_img is not None:
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
    
    def _simulate_comparison(self, face_image: bytes) -> Tuple[bool, Optional[str], bool]:
        """Simula comparación para modo dry_run."""
        is_new, face_id = self._add_new_face(face_image)
        return is_new, face_id, False
    
    def _maybe_cleanup(self) -> None:
        """Limpia cache si ha pasado suficiente tiempo."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() > 300:
            self.cleanup_expired()
            self._last_cleanup = now
    
    def cleanup_expired(self) -> int:
        """Elimina rostros expirados (solo pasajeros, no excluidos)."""
        with self._lock:
            expired_ids = [
                face_id for face_id, face in self._tracked_faces.items()
                if face.is_expired(self.ttl_minutes)
            ]
            
            for face_id in expired_ids:
                del self._tracked_faces[face_id]
            
            if expired_ids and len(self._tracked_faces) == 0:
                self._collages.clear()
                self._current_collage_faces = 0
                logger.info("Todos los pasajeros expiraron, collages limpiados")
            
            if expired_ids:
                logger.info(f"Limpieza: {len(expired_ids)} pasajeros expirados")
            
            return len(expired_ids)
    
    def _evict_oldest(self) -> None:
        """Elimina el pasajero más antiguo."""
        if not self._tracked_faces:
            return
        
        oldest_id = min(
            self._tracked_faces.keys(),
            key=lambda k: self._tracked_faces[k].first_seen
        )
        
        del self._tracked_faces[oldest_id]
        logger.debug(f"Pasajero más antiguo eliminado: {oldest_id}")
    
    def clear(self) -> int:
        """Limpia cache de pasajeros (mantiene excluidos)."""
        with self._lock:
            count = len(self._tracked_faces)
            self._tracked_faces.clear()
            self._collages.clear()
            self._current_collage_faces = 0
            logger.info(f"Cache de pasajeros limpiado: {count}")
            return count
    
    def clear_excluded(self) -> int:
        """Limpia el collage de excluidos."""
        with self._lock:
            count = self._excluded_faces_count
            self._excluded_collage = None
            self._excluded_faces_count = 0
            logger.info(f"Excluidos limpiados: {count}")
            return count
    
    def process_offline_queue(self, limit: int = 50) -> Dict[str, int]:
        """
        Procesa rostros almacenados en cache offline.
        
        Debe llamarse periódicamente cuando la conexión se restablezca.
        Procesa rostros pendientes y los clasifica como nuevos, duplicados
        o excluidos.
        
        Args:
            limit: Número máximo de rostros a procesar por llamada
            
        Returns:
            Diccionario con estadísticas del procesamiento:
            - processed: Total procesados
            - new_passengers: Nuevos pasajeros encontrados
            - duplicates: Duplicados detectados
            - excluded: Personal autorizado detectado
            - failed: Fallaron al procesar
        """
        if not self._offline_cache:
            return {"processed": 0, "new_passengers": 0, "duplicates": 0, "excluded": 0, "failed": 0}
        
        pending = self._offline_cache.get_pending_faces(limit=limit)
        if not pending:
            return {"processed": 0, "new_passengers": 0, "duplicates": 0, "excluded": 0, "failed": 0}
        
        logger.info(f"Procesando {len(pending)} rostros del cache offline...")
        
        stats = {"processed": 0, "new_passengers": 0, "duplicates": 0, "excluded": 0, "failed": 0}
        
        for face_record in pending:
            try:
                face_image = face_record["image_data"]
                cache_id = face_record["id"]
                
                # Intentar procesar con AWS
                # Nota: usamos lógica similar a is_new_passenger pero sin recursión
                is_new, face_id, is_excluded = self._process_cached_face(face_image)
                
                # Marcar como procesado
                self._offline_cache.mark_processed(
                    face_id=cache_id,
                    is_new=is_new,
                    tracked_face_id=face_id,
                    is_excluded=is_excluded
                )
                
                stats["processed"] += 1
                if is_excluded:
                    stats["excluded"] += 1
                elif is_new:
                    stats["new_passengers"] += 1
                else:
                    stats["duplicates"] += 1
                    
            except Exception as e:
                logger.warning(f"Error procesando rostro offline {face_record['id']}: {e}")
                self._offline_cache.mark_failed(face_record["id"], str(e))
                stats["failed"] += 1
        
        logger.info(
            f"Cache offline procesado: {stats['processed']} rostros, "
            f"nuevos={stats['new_passengers']}, dup={stats['duplicates']}, "
            f"excluidos={stats['excluded']}, fallidos={stats['failed']}"
        )
        
        # Reset offline flag if successful processing
        if stats["processed"] > 0 and stats["failed"] == 0:
            self._is_offline = False
        
        return stats
    
    def _process_cached_face(self, face_image: bytes) -> Tuple[bool, Optional[str], bool]:
        """
        Procesa un rostro del cache (sin almacenar en cache si falla).
        
        Similar a is_new_passenger pero sin el fallback a offline storage.
        """
        # Mismo flujo que is_new_passenger pero más simple
        if self._client is None and not self._init_client():
            raise Exception("No se pudo inicializar cliente AWS")
        
        # Si no hay collages ni excluidos, es nuevo
        if not self._collages and self._excluded_collage is None:
            return self._add_new_face(face_image)[0], None, False
        
        # Verificar solo contra excluidos si no hay pasajeros
        if not self._collages and self._excluded_collage is not None:
            excluded_bytes = self._get_excluded_only_bytes()
            if excluded_bytes and self._check_against_excluded(face_image, excluded_bytes):
                return False, None, True
            is_new, face_id = self._add_new_face(face_image)
            return is_new, face_id, False
        
        # Verificar contra collages combinados
        for collage_idx in range(len(self._collages)):
            combined_bytes, excluded_width = self._build_combined_collage(collage_idx)
            
            self._total_api_calls += 1
            response = self._client.compare_faces(
                SourceImage={"Bytes": face_image},
                TargetImage={"Bytes": combined_bytes},
                SimilarityThreshold=self.similarity_threshold
            )
            
            matches = response.get("FaceMatches", [])
            if matches:
                match = matches[0]
                face_bbox = match.get("Face", {}).get("BoundingBox", {})
                match_left = face_bbox.get("Left", 0)
                match_x_pixels = int(match_left * (excluded_width + self._collages[collage_idx].shape[1]))
                
                if excluded_width > 0 and match_x_pixels < excluded_width:
                    self._total_excluded_detected += 1
                    return False, None, True
                else:
                    self._total_duplicates += 1
                    return False, self._find_face_id_by_collage(collage_idx), False
        
        # No coincide con nadie, es nuevo
        is_new, face_id = self._add_new_face(face_image)
        return is_new, face_id, False
    
    def get_offline_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas del cache offline si está habilitado."""
        if self._offline_cache:
            return self._offline_cache.get_stats()
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del rastreador."""
        with self._lock:
            active_faces = sum(
                1 for f in self._tracked_faces.values()
                if not f.is_expired(self.ttl_minutes)
            )
            
            stats = {
                "tracked_faces": len(self._tracked_faces),
                "active_faces": active_faces,
                "collages": len(self._collages),
                "faces_in_current_collage": self._current_collage_faces,
                "excluded_faces": self._excluded_faces_count,
                "total_comparisons": self._total_comparisons,
                "total_api_calls": self._total_api_calls,
                "new_passengers": self._total_new_passengers,
                "duplicates": self._total_duplicates,
                "excluded_detected": self._total_excluded_detected,
                "offline_cached": self._total_offline_cached,
                "is_offline": self._is_offline,
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
            
            # Agregar stats del cache offline si existe
            if self._offline_cache:
                stats["offline_cache"] = self._offline_cache.get_stats()
            
            return stats


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
    
    left = int(bounding_box['Left'] * width)
    top = int(bounding_box['Top'] * height)
    box_width = int(bounding_box['Width'] * width)
    box_height = int(bounding_box['Height'] * height)
    
    pad_x = int(box_width * padding)
    pad_y = int(box_height * padding)
    
    x1 = max(0, left - pad_x)
    y1 = max(0, top - pad_y)
    x2 = min(width, left + box_width + pad_x)
    y2 = min(height, top + box_height + pad_y)
    
    face_img = frame[y1:y2, x1:x2]
    
    _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Probando FaceTracker con excluded_faces...")
    
    # Test básico en dry_run
    tracker = FaceTracker(dry_run=True, ttl_minutes=5)
    
    fake_face_1 = b"fake_face_image_1"
    is_new, face_id, is_excluded = tracker.is_new_passenger(fake_face_1)
    print(f"Rostro 1: is_new={is_new}, face_id={face_id}, is_excluded={is_excluded}")
    
    stats = tracker.get_stats()
    print(f"Stats: tracked={stats['tracked_faces']}, excluded={stats['excluded_faces']}")
    
    print("\nTest completado")
