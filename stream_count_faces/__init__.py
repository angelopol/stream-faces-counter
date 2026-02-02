"""
stream_count_faces - Paquete para conteo de rostros en streams de video

Este paquete proporciona componentes modulares para:
- Captura de video no bloqueante (VideoStream)
- Detección de movimiento (MotionDetector)
- Almacenamiento local con SQLite (LocalBuffer)
- Conteo de rostros con AWS Rekognition (FaceCounter)
- Rastreo de pasajeros para deduplicación (FaceTracker)

Uso típico:
    from stream_count_faces import VideoStream, MotionDetector, LocalBuffer, FaceCounter, FaceTracker
"""

from .camera import VideoStream
from .motion import MotionDetector
from .storage import LocalBuffer
from .detector import FaceCounter
from .tracker import FaceTracker, extract_face_image

__version__ = "1.1.0"
__author__ = "angelopol"

__all__ = [
    "VideoStream",
    "MotionDetector", 
    "LocalBuffer",
    "FaceCounter",
    "FaceTracker",
    "extract_face_image",
]
