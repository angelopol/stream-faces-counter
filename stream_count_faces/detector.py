"""
detector.py - Detección y conteo de rostros con AWS Rekognition

Proporciona la clase FaceCounter como wrapper para la lógica de detección
de rostros, integrando AWS Rekognition con filtrado de calidad.
"""

import os
import cv2
import boto3
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from botocore.exceptions import ClientError, NoCredentialsError

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables directly

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """
    Representa un rostro detectado con sus atributos.
    """
    confidence: float
    bounding_box: Dict[str, float]
    pose: Dict[str, float]
    is_frontal: bool
    is_occluded: bool
    quality: Dict[str, float]
    
    def get_cv2_rect(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Convierte el bounding box a coordenadas de OpenCV.
        
        Args:
            width: Ancho del frame
            height: Alto del frame
            
        Returns:
            Tupla (left, top, right, bottom) en píxeles
        """
        box = self.bounding_box
        left = int(box['Left'] * width)
        top = int(box['Top'] * height)
        right = int((box['Left'] + box['Width']) * width)
        bottom = int((box['Top'] + box['Height']) * height)
        return left, top, right, bottom


class FaceCounter:
    """
    Contador de rostros usando AWS Rekognition.
    
    Esta clase actúa como wrapper para la detección de rostros,
    aplicando filtros de calidad para asegurar conteos precisos:
    - Confianza mínima configurable
    - Filtrado de rostros no frontales
    - Detección de oclusión facial
    
    Soporta modo dry_run para desarrollo sin conexión a AWS.
    
    Las credenciales AWS pueden configurarse de tres formas (en orden de prioridad):
    1. Parámetros directos al constructor
    2. Variables de entorno (.env o sistema)
    3. Archivo ~/.aws/credentials
    
    Atributos:
        face_confidence_threshold: Confianza mínima para considerar un rostro
        face_occluded_threshold: Umbral de confianza para oclusión
        frontal_threshold: Ángulo máximo para considerar rostro frontal
        dry_run: Si True, simula detección sin llamar a AWS
        
    Ejemplo:
        >>> counter = FaceCounter(dry_run=False)
        >>> faces = counter.count_faces(frame)
        >>> print(f"Rostros detectados: {len(faces)}")
        
        # Con credenciales explícitas
        >>> counter = FaceCounter(
        ...     aws_access_key_id="AKIA...",
        ...     aws_secret_access_key="...",
        ...     region="us-east-1"
        ... )
    """
    
    def __init__(
        self,
        face_confidence_threshold: float = 90.0,
        face_occluded_threshold: float = 80.0,
        frontal_threshold: float = 35.0,
        dry_run: bool = False,
        engine: str = "aws",
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Inicializa el contador de rostros.
        
        Args:
            face_confidence_threshold: Confianza mínima (0-100)
            face_occluded_threshold: Umbral de oclusión (0-100)
            frontal_threshold: Ángulo máximo para frontalidad
            dry_run: Si True, no hace llamadas reales a AWS
            engine: Motor de detección ("aws" o "local")
            region: Región de AWS para Rekognition (default: env o us-east-1)
            aws_access_key_id: AWS Access Key ID (opcional)
            aws_secret_access_key: AWS Secret Access Key (opcional)
            aws_session_token: AWS Session Token para credenciales temporales
        """
        self.face_confidence_threshold = face_confidence_threshold
        self.face_occluded_threshold = face_occluded_threshold
        self.frontal_threshold = frontal_threshold
        self.dry_run = dry_run
        self.engine = engine
        
        # AWS Configuration - parameters override environment variables
        self.region = region or os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self._aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self._aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self._aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        
        self._client = None
        self._total_faces_detected = 0
        self._total_frames_processed = 0
        self._last_error: Optional[str] = None
        
        if self.engine == "local":
            model_path = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found. Ensure it is downloaded.")
            
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.face_confidence_threshold / 100.0
            )
            self._face_detector = mp_vision.FaceDetector.create_from_options(options)
            logger.info("Motor de detección local (Mediapipe Tasks) inicializado")
        elif not dry_run:
            self._init_client()
    
    def _init_client(self) -> bool:
        """
        Inicializa el cliente de AWS Rekognition.
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            # Build kwargs for boto3 client
            client_kwargs = {"region_name": self.region}
            
            # Only pass credentials if explicitly provided
            if self._aws_access_key_id and self._aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = self._aws_access_key_id
                client_kwargs["aws_secret_access_key"] = self._aws_secret_access_key
                if self._aws_session_token:
                    client_kwargs["aws_session_token"] = self._aws_session_token
                logger.debug("Using explicit AWS credentials")
            else:
                logger.debug("Using default AWS credential chain")
            
            self._client = boto3.client("rekognition", **client_kwargs)
            logger.info(f"Cliente Rekognition inicializado (región: {self.region})")
            return True
        except NoCredentialsError:
            logger.error(
                "No se encontraron credenciales de AWS. "
                "Configure AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY en .env, "
                "páselas como parámetros, o use --dry-run para modo simulación."
            )
            self._last_error = "No AWS credentials found"
            return False
        except Exception as e:
            logger.error(f"Error al inicializar cliente Rekognition: {e}")
            self._last_error = str(e)
            return False
    
    def _is_frontal_face(self, pose: Dict[str, float]) -> bool:
        """
        Determina si un rostro está orientado frontalmente.
        
        Args:
            pose: Diccionario con Yaw, Roll, Pitch
            
        Returns:
            True si el rostro está suficientemente frontal
        """
        yaw = abs(pose.get('Yaw', 0))
        roll = abs(pose.get('Roll', 0))
        pitch = abs(pose.get('Pitch', 0))
        
        return (
            yaw < self.frontal_threshold and
            roll < self.frontal_threshold and
            pitch < self.frontal_threshold
        )
    
    def _is_occluded(self, face_detail: Dict[str, Any]) -> bool:
        """
        Determina si un rostro está ocluido.
        
        Args:
            face_detail: Detalle del rostro de Rekognition
            
        Returns:
            True si el rostro está significativamente ocluido
        """
        occluded = face_detail.get('FaceOccluded', {})
        if occluded.get('Value', False):
            confidence = occluded.get('Confidence', 0)
            return confidence > self.face_occluded_threshold
        return False
    
    def count_faces(self, frame) -> List[DetectedFace]:
        """
        Detecta y cuenta rostros en un frame.
        
        Aplica filtros de calidad para retornar solo rostros válidos:
        - Confianza por encima del umbral
        - Orientación frontal
        - Sin oclusión significativa
        
        Args:
            frame: Frame BGR de OpenCV (numpy array)
            
        Returns:
            Lista de DetectedFace con rostros válidos
        """
        if frame is None:
            return []
        
        self._total_frames_processed += 1
        
        if self.dry_run:
            return self._simulate_detection(frame)
            
        if self.engine == "local":
            return self._local_detect(frame)
        
        if self._client is None:
            if not self._init_client():
                return []
        
        try:
            # Codificar frame a JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = buffer.tobytes()
            
            # Llamar a Rekognition
            response = self._client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"]
            )
            
            faces = []
            for face_detail in response.get("FaceDetails", []):
                confidence = face_detail.get('Confidence', 0)
                
                # Filtro de confianza
                if confidence < self.face_confidence_threshold:
                    continue
                
                pose = face_detail.get('Pose', {})
                is_frontal = self._is_frontal_face(pose)
                
                # Filtro de frontalidad
                if not is_frontal:
                    continue
                
                is_occluded = self._is_occluded(face_detail)
                
                # Filtro de oclusión
                if is_occluded:
                    continue
                
                quality = face_detail.get('Quality', {})
                
                detected_face = DetectedFace(
                    confidence=confidence,
                    bounding_box=face_detail.get('BoundingBox', {}),
                    pose=pose,
                    is_frontal=is_frontal,
                    is_occluded=is_occluded,
                    quality={
                        'brightness': quality.get('Brightness', 0),
                        'sharpness': quality.get('Sharpness', 0)
                    }
                )
                faces.append(detected_face)
            
            self._total_faces_detected += len(faces)
            
            if faces:
                logger.debug(f"Rostros detectados: {len(faces)}")
            
            return faces
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Error de Rekognition ({error_code}): {e}")
            self._last_error = str(e)
            return []
        except Exception as e:
            logger.error(f"Error en detección de rostros (AWS): {e}")
            self._last_error = str(e)
            return []
            
    def _local_detect(self, frame) -> List[DetectedFace]:
        """
        Detecta y extrae metadata de rostros usando Mediapipe (procesamiento local).
        No retorna error en caso de fallo, retorna lista vacía.
        """
        try:
            # Convertir a RGB para Mediapipe y crear mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detectar caras
            detection_result = self._face_detector.detect(mp_image)
            
            faces = []
            if detection_result.detections:
                height, width = frame.shape[:2]
                for detection in detection_result.detections:
                    confidence = detection.categories[0].score * 100.0
                    if confidence < self.face_confidence_threshold:
                        continue
                        
                    bbox = detection.bounding_box
                    # Convert to normalized rekognition format
                    bounding_box = {
                        'Left': max(0.0, bbox.origin_x / width),
                        'Top': max(0.0, bbox.origin_y / height),
                        'Width': min(1.0 - (bbox.origin_x / width), bbox.width / width),
                        'Height': min(1.0 - (bbox.origin_y / height), bbox.height / height)
                    }
                    
                    # Calcular calidad (Sharpness & Brightness)
                    x_px = int(bounding_box['Left'] * width)
                    y_px = int(bounding_box['Top'] * height)
                    w_px = int(bounding_box['Width'] * width)
                    h_px = int(bounding_box['Height'] * height)
                    
                    x_px = max(0, min(x_px, width - 1))
                    y_px = max(0, min(y_px, height - 1))
                    w_px = max(1, min(w_px, width - x_px))
                    h_px = max(1, min(h_px, height - y_px))
                    
                    roi = frame[y_px:y_px+h_px, x_px:x_px+w_px]
                    sharpness = 0
                    brightness = 0
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                        brightness = np.mean(gray_roi)
                    
                    detected_face = DetectedFace(
                        confidence=confidence,
                        bounding_box=bounding_box,
                        pose={'Yaw': 0, 'Roll': 0, 'Pitch': 0},
                        is_frontal=True,  # Asumimos que si MP lo detecta, es suficientemente frontal
                        is_occluded=False,
                        quality={'brightness': float(brightness), 'sharpness': float(sharpness)}
                    )
                    faces.append(detected_face)
                    
            self._total_faces_detected += len(faces)
            if faces:
                logger.debug(f"Rostros detectados (local): {len(faces)}")
            return faces
        except Exception as e:
            logger.error(f"Error en detección local mediapipe: {e}")
            self._last_error = str(e)
            return []
    
    def _simulate_detection(self, frame) -> List[DetectedFace]:
        """
        Simula detección de rostros para modo dry_run.
        
        En modo simulación, detecta rostros usando Haar Cascades de OpenCV
        como alternativa local sin necesidad de AWS.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Lista de DetectedFace simulados
        """
        try:
            # Usar detector de rostros de OpenCV como fallback
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            height, width = frame.shape[:2]
            faces = []
            
            for (x, y, w, h) in detected:
                # Convertir a formato de Rekognition (normalizado)
                bounding_box = {
                    'Left': x / width,
                    'Top': y / height,
                    'Width': w / width,
                    'Height': h / height
                }
                
                detected_face = DetectedFace(
                    confidence=95.0,  # Simulado
                    bounding_box=bounding_box,
                    pose={'Yaw': 0, 'Roll': 0, 'Pitch': 0},
                    is_frontal=True,
                    is_occluded=False,
                    quality={'brightness': 80, 'sharpness': 80}
                )
                faces.append(detected_face)
            
            self._total_faces_detected += len(faces)
            
            if faces:
                logger.debug(f"Rostros detectados (simulación): {len(faces)}")
            
            return faces
            
        except Exception as e:
            logger.error(f"Error en simulación de detección: {e}")
            return []
    
    def draw_faces(
        self,
        frame,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ):
        """
        Dibuja rectángulos alrededor de los rostros detectados.
        
        Args:
            frame: Frame BGR donde dibujar
            faces: Lista de DetectedFace
            color: Color BGR del rectángulo
            thickness: Grosor de la línea
            
        Returns:
            Frame con rostros marcados
        """
        output = frame.copy()
        height, width = frame.shape[:2]
        
        for face in faces:
            left, top, right, bottom = face.get_cv2_rect(width, height)
            cv2.rectangle(output, (left, top), (right, bottom), color, thickness)
            
            # Mostrar confianza
            label = f"{face.confidence:.1f}%"
            cv2.putText(
                output, label, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del contador.
        
        Returns:
            Diccionario con estadísticas de detección
        """
        return {
            "total_faces_detected": self._total_faces_detected,
            "total_frames_processed": self._total_frames_processed,
            "dry_run": self.dry_run,
            "last_error": self._last_error,
            "thresholds": {
                "confidence": self.face_confidence_threshold,
                "occluded": self.face_occluded_threshold,
                "frontal": self.frontal_threshold
            }
        }


if __name__ == "__main__":
    import numpy as np
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test frame (black)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("\nProbando FaceCounter en modo local (Mediapipe Tasks Vision)...")
    counter_local = FaceCounter(engine="local")
    faces_local = counter_local.count_faces(test_frame)
    print(f"Rostros en frame negro (local): {len(faces_local)}")
    
    # Stats
    
    print("\nPara probar con una cámara real, ejecute:")
    print("  python -c \"")
    print("  import cv2")
    print("  from detector import FaceCounter")
    print("  cap = cv2.VideoCapture(0)")
    print("  counter = FaceCounter(engine='local')")
    print("  ret, frame = cap.read()")
    print("  if ret:")
    print("      faces = counter.count_faces(frame)")
    print("      print(f'Rostros: {len(faces)}')")
    print("  cap.release()\"")
    
    print("\nTest completado")
