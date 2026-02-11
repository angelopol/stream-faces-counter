"""
motion.py - Detección de movimiento por diferenciación de frames

Proporciona la clase MotionDetector para filtrar frames sin actividad,
optimizando el uso de recursos al evitar procesamiento innecesario.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Detector de movimiento basado en diferenciación de frames.
    
    Utiliza técnicas de visión por computador para detectar cambios
    significativos entre frames consecutivos:
    1. Conversión a escala de grises
    2. Desenfoque gaussiano para reducir ruido
    3. Diferencia absoluta entre frames
    4. Umbralización binaria
    5. Detección de contornos
    6. Filtrado por área mínima
    
    Atributos:
        min_area: Área mínima de contorno para considerar movimiento
        threshold: Umbral de diferencia binaria (0-255)
        blur_kernel: Tamaño del kernel gaussiano (debe ser impar)
        
    Ejemplo:
        >>> detector = MotionDetector(min_area=5000, threshold=25)
        >>> while True:
        ...     frame = camera.read()
        ...     if detector.detect(frame):
        ...         print("¡Movimiento detectado!")
    """
    
    def __init__(
        self,
        min_area: int = 5000,
        threshold: int = 25,
        blur_kernel: int = 21,
        cooldown_frames: int = 5
    ):
        """
        Inicializa el detector de movimiento.
        
        Args:
            min_area: Área mínima de contorno para activar detección (píxeles²)
            threshold: Umbral para binarización de diferencia (0-255)
            blur_kernel: Tamaño del kernel gaussiano (debe ser impar)
            cooldown_frames: Frames a mantener activo después de detectar movimiento
        """
        self.min_area = min_area
        self.threshold = threshold
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.cooldown_frames = cooldown_frames
        
        self._reference_frame: Optional[np.ndarray] = None
        self._cooldown_counter = 0
        self._motion_regions: List[Tuple[int, int, int, int]] = []
        self._total_motion_area = 0
        self._frame_count = 0
        
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesa un frame para comparación.
        
        Args:
            frame: Frame BGR de entrada
            
        Returns:
            Frame en escala de grises con desenfoque aplicado
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar desenfoque gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return blurred
    
    def detect(self, frame: np.ndarray) -> bool:
        """
        Detecta si hay movimiento significativo en el frame.
        
        Compara el frame actual con el frame de referencia.
        El primer frame siempre establece la referencia y retorna False.
        
        Args:
            frame: Frame BGR de la cámara
            
        Returns:
            True si se detectó movimiento significativo, False en caso contrario
        """
        if frame is None:
            return False
        
        self._frame_count += 1
        processed = self._preprocess(frame)
        
        # Primer frame: establecer referencia
        if self._reference_frame is None:
            self._reference_frame = processed
            logger.debug("Frame de referencia establecido")
            return False
        
        # Verificar si las dimensiones coinciden
        if processed.shape != self._reference_frame.shape:
            logger.warning(f"Cambio de resolución detectado: {self._reference_frame.shape} -> {processed.shape}. Reiniciando referencia.")
            self._reference_frame = processed
            return False
        
        # Si estamos en cooldown, mantener el estado de movimiento
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            # Actualizar referencia gradualmente durante cooldown
            self._reference_frame = cv2.addWeighted(
                self._reference_frame, 0.7, processed, 0.3, 0
            )
            return True
        
        # Calcular diferencia absoluta
        frame_delta = cv2.absdiff(self._reference_frame, processed)
        
        # Aplicar umbral binario
        _, thresh = cv2.threshold(
            frame_delta, self.threshold, 255, cv2.THRESH_BINARY
        )
        
        # Dilatar para rellenar huecos
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos por área
        self._motion_regions = []
        self._total_motion_area = 0
        motion_detected = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                motion_detected = True
                self._total_motion_area += area
                x, y, w, h = cv2.boundingRect(contour)
                self._motion_regions.append((x, y, w, h))
        
        if motion_detected:
            self._cooldown_counter = self.cooldown_frames
            logger.debug(
                f"Movimiento detectado: {len(self._motion_regions)} regiones, "
                f"área total: {self._total_motion_area}"
            )
        else:
            # Actualizar referencia gradualmente cuando no hay movimiento
            self._reference_frame = cv2.addWeighted(
                self._reference_frame, 0.95, processed, 0.05, 0
            )
        
        return motion_detected
    
    def get_motion_regions(self) -> List[Tuple[int, int, int, int]]:
        """
        Obtiene las regiones donde se detectó movimiento.
        
        Returns:
            Lista de tuplas (x, y, width, height) para cada región de movimiento
        """
        return self._motion_regions.copy()
    
    def get_motion_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Genera una máscara binaria de las áreas con movimiento.
        
        Args:
            frame: Frame BGR de referencia para dimensiones
            
        Returns:
            Máscara binaria donde blanco indica movimiento
        """
        if not self._motion_regions:
            return None
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in self._motion_regions:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask
    
    def draw_motion_regions(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Dibuja rectángulos en las regiones con movimiento.
        
        Args:
            frame: Frame BGR donde dibujar
            color: Color BGR del rectángulo
            thickness: Grosor de la línea
            
        Returns:
            Frame con rectángulos dibujados
        """
        output = frame.copy()
        for (x, y, w, h) in self._motion_regions:
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        return output
    
    def reset(self) -> None:
        """
        Reinicia el detector, descartando el frame de referencia.
        """
        self._reference_frame = None
        self._cooldown_counter = 0
        self._motion_regions = []
        self._total_motion_area = 0
        logger.debug("Detector de movimiento reiniciado")
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del detector.
        
        Returns:
            Diccionario con estadísticas de detección
        """
        return {
            "frame_count": self._frame_count,
            "has_reference": self._reference_frame is not None,
            "cooldown_active": self._cooldown_counter > 0,
            "motion_regions": len(self._motion_regions),
            "total_motion_area": self._total_motion_area,
            "min_area": self.min_area,
            "threshold": self.threshold
        }


if __name__ == "__main__":
    # Test con frames sintéticos
    logging.basicConfig(level=logging.DEBUG)
    
    print("Probando MotionDetector...")
    detector = MotionDetector(min_area=1000, threshold=25)
    
    # Frame negro (referencia)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    result1 = detector.detect(frame1)
    print(f"Frame 1 (negro): movimiento={result1}")  # False (establece referencia)
    
    # Frame idéntico
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    result2 = detector.detect(frame2)
    print(f"Frame 2 (negro): movimiento={result2}")  # False
    
    # Frame con cambio significativo (cuadrado blanco)
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame3, (200, 150), (400, 350), (255, 255, 255), -1)
    result3 = detector.detect(frame3)
    print(f"Frame 3 (con cuadrado): movimiento={result3}")  # True
    
    print(f"Regiones de movimiento: {detector.get_motion_regions()}")
    print(f"Stats: {detector.get_stats()}")
    
    print("Test completado")
