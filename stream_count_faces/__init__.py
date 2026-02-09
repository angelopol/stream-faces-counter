"""
stream_count_faces - Paquete para conteo de rostros en streams de video

Este paquete proporciona componentes modulares para:
- Captura de video no bloqueante (VideoStream)
- Detección de movimiento (MotionDetector)
- Almacenamiento local con SQLite (LocalBuffer)
- Cache de rostros para procesamiento offline (FaceCache)
- Almacenamiento de eventos de pasajeros (PassengerEventStore)
- Geolocalización GPS/IP (LocationProvider)
- Conteo de rostros con AWS Rekognition (FaceCounter)
- Rastreo de pasajeros para deduplicación (FaceTracker)
- Sincronización con servidor admin (CloudSync)

Uso típico:
    from stream_count_faces import VideoStream, MotionDetector, LocalBuffer, FaceCounter, FaceTracker
"""

from .camera import VideoStream
from .motion import MotionDetector
from .storage import LocalBuffer, FaceCache, PassengerEventStore
from .geolocation import LocationProvider, Location
from .detector import FaceCounter
from .tracker import FaceTracker, extract_face_image
from .sync import CloudSync, SyncManager, SyncResult, get_device_mac

__version__ = "1.4.0"
__author__ = "angelopol"

__all__ = [
    "VideoStream",
    "MotionDetector", 
    "LocalBuffer",
    "FaceCache",
    "PassengerEventStore",
    "LocationProvider",
    "Location",
    "FaceCounter",
    "FaceTracker",
    "extract_face_image",
    "CloudSync",
    "SyncManager",
    "SyncResult",
    "get_device_mac",
]
