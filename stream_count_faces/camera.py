"""
camera.py - Captura de video no bloqueante con threading

Proporciona la clase VideoStream para captura continua de frames
sin bloquear el hilo principal, optimizado para sistemas embebidos.
"""

import cv2
import threading
import time
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Captura de video en tiempo real usando un hilo dedicado.
    
    Esta clase evita cuellos de botella de I/O al leer frames
    continuamente en un hilo separado, permitiendo que el hilo
    principal procese frames sin esperar por la cámara.
    
    Atributos:
        source: Fuente de video (índice de cámara o URL)
        width: Ancho de frame objetivo
        height: Alto de frame objetivo
        
    Ejemplo:
        >>> stream = VideoStream(source=0, width=1280, height=720)
        >>> stream.start()
        >>> frame = stream.read()
        >>> if frame is not None:
        ...     # procesar frame
        >>> stream.stop()
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        reconnect_delay: float = 2.0
    ):
        """
        Inicializa el stream de video.
        
        Args:
            source: Índice de cámara (0, 1, ...) o URL RTSP/HTTP
            width: Ancho de frame objetivo
            height: Alto de frame objetivo
            reconnect_delay: Segundos a esperar antes de reconectar
        """
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._frame_count = 0
        self._last_frame_time = 0.0
        
    def _connect(self) -> bool:
        """
        Establece conexión con la fuente de video.
        
        Returns:
            True si la conexión fue exitosa, False en caso contrario
        """
        try:
            if self._capture is not None:
                self._capture.release()
                
            self._capture = cv2.VideoCapture(self.source)
            
            if not self._capture.isOpened():
                logger.error(f"No se pudo abrir la fuente de video: {self.source}")
                return False
            
            # Configurar resolución
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Configurar buffer mínimo para evitar frames antiguos
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Leer frame de prueba
            ret, frame = self._capture.read()
            if not ret or frame is None:
                logger.error("No se pudo leer frame inicial de la cámara")
                return False
            
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(
                f"Cámara conectada: {actual_width}x{actual_height} @ {fps:.1f} FPS"
            )
            
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error al conectar con la cámara: {e}")
            return False
    
    def _update(self) -> None:
        """
        Bucle principal del hilo de captura.
        
        Lee frames continuamente y los almacena en un buffer interno.
        Maneja reconexión automática en caso de pérdida de conexión.
        """
        while self._running:
            if not self._connected:
                logger.info("Intentando reconectar con la cámara...")
                if self._connect():
                    logger.info("Reconexión exitosa")
                else:
                    time.sleep(self.reconnect_delay)
                    continue
            
            try:
                ret, frame = self._capture.read()
                
                if not ret or frame is None:
                    logger.warning("Frame perdido, reintentando...")
                    self._connected = False
                    continue
                
                # Actualizar frame en buffer thread-safe
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error en captura de frame: {e}")
                self._connected = False
                time.sleep(0.1)
    
    def start(self) -> "VideoStream":
        """
        Inicia el hilo de captura de video.
        
        Returns:
            self para permitir encadenamiento
        """
        if self._running:
            logger.warning("El stream ya está corriendo")
            return self
        
        if not self._connect():
            logger.error("No se pudo iniciar el stream")
            return self
        
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        
        logger.info("Stream de video iniciado")
        return self
    
    def read(self) -> Optional[any]:
        """
        Obtiene el frame más reciente del buffer.
        
        Returns:
            Frame actual como numpy array, o None si no hay frame disponible
        """
        with self._frame_lock:
            if self._frame is None:
                return None
            # Retornar copia para evitar modificaciones concurrentes
            return self._frame.copy()
    
    def stop(self) -> None:
        """
        Detiene el stream y libera recursos.
        """
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        
        self._connected = False
        logger.info("Stream de video detenido")
    
    def is_running(self) -> bool:
        """
        Verifica si el stream está activo.
        
        Returns:
            True si el stream está corriendo
        """
        return self._running and self._connected
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del stream.
        
        Returns:
            Diccionario con estadísticas de captura
        """
        with self._frame_lock:
            return {
                "running": self._running,
                "connected": self._connected,
                "frame_count": self._frame_count,
                "last_frame_time": self._last_frame_time,
                "source": self.source,
                "resolution": f"{self.width}x{self.height}"
            }
    
    def __enter__(self) -> "VideoStream":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.INFO)
    
    print("Probando VideoStream...")
    with VideoStream(source=0, width=640, height=480) as stream:
        time.sleep(1)  # Esperar a que capture algunos frames
        
        for i in range(10):
            frame = stream.read()
            if frame is not None:
                print(f"Frame {i+1}: shape={frame.shape}")
            else:
                print(f"Frame {i+1}: None")
            time.sleep(0.1)
        
        print(f"Stats: {stream.get_stats()}")
    
    print("Test completado")
