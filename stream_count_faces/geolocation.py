"""
geolocation.py - Proveedor de ubicación multiplataforma

Proporciona la clase LocationProvider para obtener coordenadas GPS
con múltiples fuentes y fallback automático.

Fuentes de ubicación (en orden de prioridad):
1. GPSD (Linux GPS daemon)
2. GPS Serial (Windows/Linux via puerto COM/tty)
3. IP Geolocation (fallback basado en IP pública)
"""

import logging
import os
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Location:
    """Representa una ubicación geográfica."""
    latitude: Optional[float]
    longitude: Optional[float]
    source: str  # 'gps', 'serial', 'ip', 'none'
    accuracy: Optional[float] = None  # metros
    timestamp: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Verifica si la ubicación es válida."""
        return self.latitude is not None and self.longitude is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp
        }


class LocationProvider:
    """
    Proveedor de ubicación con múltiples fuentes.
    
    Intenta obtener ubicación GPS primero, con fallback a IP geolocation.
    Compatible con Windows y Linux.
    
    Fuentes soportadas:
    - GPSD: Daemon GPS en Linux (requiere gpsd corriendo)
    - Serial GPS: Dispositivo GPS via puerto serial (COM en Windows)
    - IP Geolocation: Fallback basado en IP (requiere internet)
    
    Ejemplo:
        >>> provider = LocationProvider(serial_port="COM3")
        >>> location = provider.get_location()
        >>> if location.is_valid():
        ...     print(f"Lat: {location.latitude}, Lng: {location.longitude}")
    """
    
    def __init__(
        self,
        serial_port: Optional[str] = None,
        serial_baudrate: int = 9600,
        gpsd_host: str = "localhost",
        gpsd_port: int = 2947,
        use_ip_fallback: bool = True,
        cache_seconds: int = 10
    ):
        """
        Inicializa el proveedor de ubicación.
        
        Args:
            serial_port: Puerto serial del GPS (e.g., "COM3", "/dev/ttyUSB0")
            serial_baudrate: Baudrate del GPS serial (default: 9600)
            gpsd_host: Host del servicio GPSD (default: localhost)
            gpsd_port: Puerto del servicio GPSD (default: 2947)
            use_ip_fallback: Si usar IP geolocation como fallback
            cache_seconds: Segundos para cachear ubicación (evita llamadas excesivas)
        """
        self.serial_port = serial_port or os.getenv("GPS_SERIAL_PORT")
        self.serial_baudrate = serial_baudrate
        self.gpsd_host = gpsd_host
        self.gpsd_port = gpsd_port
        self.use_ip_fallback = use_ip_fallback
        self.cache_seconds = cache_seconds
        
        self._lock = threading.Lock()
        self._cached_location: Optional[Location] = None
        self._cache_time: Optional[datetime] = None
        
        # Estado de disponibilidad de fuentes
        self._gpsd_available = False
        self._serial_available = False
        self._serial_connection = None
        
        # Intentar inicializar fuentes
        self._init_sources()
        
        logger.info(
            f"LocationProvider inicializado: "
            f"gpsd={self._gpsd_available}, "
            f"serial={self._serial_available}, "
            f"ip_fallback={use_ip_fallback}"
        )
    
    def _init_sources(self) -> None:
        """Inicializa las fuentes de ubicación disponibles."""
        # Intentar GPSD (solo en Linux típicamente)
        try:
            from gps import gps, WATCH_ENABLE
            self._gpsd_available = True
            logger.debug("GPSD disponible")
        except ImportError:
            logger.debug("GPSD no disponible (gps module no instalado)")
        except Exception as e:
            logger.debug(f"Error inicializando GPSD: {e}")
        
        # Intentar GPS serial
        if self.serial_port:
            try:
                import serial
                self._serial_connection = serial.Serial(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate,
                    timeout=1
                )
                self._serial_available = True
                logger.debug(f"GPS Serial disponible en {self.serial_port}")
            except ImportError:
                logger.debug("PySerial no instalado")
            except Exception as e:
                logger.debug(f"Error abriendo puerto serial {self.serial_port}: {e}")
    
    def get_location(self) -> Location:
        """
        Obtiene la ubicación actual.
        
        Intenta múltiples fuentes en orden de prioridad.
        
        Returns:
            Location con coordenadas o (None, None) si no está disponible
        """
        with self._lock:
            # Verificar cache
            if self._is_cache_valid():
                return self._cached_location
            
            location = None
            
            # 1. Intentar GPSD
            if self._gpsd_available:
                location = self._get_from_gpsd()
                if location and location.is_valid():
                    logger.debug(f"Ubicación obtenida de GPSD: {location.latitude}, {location.longitude}")
                    self._update_cache(location)
                    return location
            
            # 2. Intentar GPS Serial
            if self._serial_available:
                location = self._get_from_serial()
                if location and location.is_valid():
                    logger.debug(f"Ubicación obtenida de GPS Serial: {location.latitude}, {location.longitude}")
                    self._update_cache(location)
                    return location
            
            # 3. Fallback a IP geolocation
            if self.use_ip_fallback:
                location = self._get_from_ip()
                if location and location.is_valid():
                    logger.debug(f"Ubicación obtenida de IP: {location.latitude}, {location.longitude}")
                    self._update_cache(location)
                    return location
            
            # No se pudo obtener ubicación
            location = Location(
                latitude=None,
                longitude=None,
                source="none",
                timestamp=datetime.now().isoformat()
            )
            self._update_cache(location)
            return location
    
    def _is_cache_valid(self) -> bool:
        """Verifica si el cache de ubicación es válido."""
        if self._cached_location is None or self._cache_time is None:
            return False
        
        elapsed = (datetime.now() - self._cache_time).total_seconds()
        return elapsed < self.cache_seconds
    
    def _update_cache(self, location: Location) -> None:
        """Actualiza el cache de ubicación."""
        self._cached_location = location
        self._cache_time = datetime.now()
    
    def _get_from_gpsd(self) -> Optional[Location]:
        """Obtiene ubicación desde GPSD."""
        try:
            from gps import gps, WATCH_ENABLE
            
            session = gps(host=self.gpsd_host, port=self.gpsd_port)
            session.stream(WATCH_ENABLE)
            
            # Intentar obtener un fix con timeout
            for _ in range(5):  # Máximo 5 intentos
                report = session.next()
                if report.get('class') == 'TPV':
                    lat = report.get('lat')
                    lon = report.get('lon')
                    if lat is not None and lon is not None:
                        return Location(
                            latitude=lat,
                            longitude=lon,
                            source="gps",
                            accuracy=report.get('epx', None),
                            timestamp=datetime.now().isoformat()
                        )
            return None
        except Exception as e:
            logger.debug(f"Error obteniendo ubicación de GPSD: {e}")
            self._gpsd_available = False
            return None
    
    def _get_from_serial(self) -> Optional[Location]:
        """Obtiene ubicación desde GPS serial (NMEA)."""
        try:
            if not self._serial_connection or not self._serial_connection.is_open:
                return None
            
            # Leer líneas NMEA buscando GGA o RMC
            for _ in range(20):  # Leer hasta 20 líneas
                line = self._serial_connection.readline().decode('ascii', errors='ignore').strip()
                
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    location = self._parse_nmea_gga(line)
                    if location:
                        return location
                
                elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    location = self._parse_nmea_rmc(line)
                    if location:
                        return location
            
            return None
        except Exception as e:
            logger.debug(f"Error leyendo GPS serial: {e}")
            return None
    
    def _parse_nmea_gga(self, sentence: str) -> Optional[Location]:
        """Parsea sentencia NMEA GGA."""
        try:
            parts = sentence.split(',')
            if len(parts) < 10:
                return None
            
            # Verificar fix quality (índice 6)
            if parts[6] == '0':  # No fix
                return None
            
            lat = self._nmea_to_decimal(parts[2], parts[3])
            lon = self._nmea_to_decimal(parts[4], parts[5])
            
            if lat is None or lon is None:
                return None
            
            return Location(
                latitude=lat,
                longitude=lon,
                source="serial",
                timestamp=datetime.now().isoformat()
            )
        except Exception:
            return None
    
    def _parse_nmea_rmc(self, sentence: str) -> Optional[Location]:
        """Parsea sentencia NMEA RMC."""
        try:
            parts = sentence.split(',')
            if len(parts) < 10:
                return None
            
            # Verificar status (índice 2)
            if parts[2] != 'A':  # A = Active, V = Void
                return None
            
            lat = self._nmea_to_decimal(parts[3], parts[4])
            lon = self._nmea_to_decimal(parts[5], parts[6])
            
            if lat is None or lon is None:
                return None
            
            return Location(
                latitude=lat,
                longitude=lon,
                source="serial",
                timestamp=datetime.now().isoformat()
            )
        except Exception:
            return None
    
    def _nmea_to_decimal(self, coord: str, direction: str) -> Optional[float]:
        """Convierte coordenada NMEA a decimal."""
        try:
            if not coord or not direction:
                return None
            
            # NMEA format: DDMM.MMMM or DDDMM.MMMM
            if direction in ['N', 'S']:
                degrees = float(coord[:2])
                minutes = float(coord[2:])
            else:
                degrees = float(coord[:3])
                minutes = float(coord[3:])
            
            decimal = degrees + minutes / 60.0
            
            if direction in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        except Exception:
            return None
    
    def _get_from_ip(self) -> Optional[Location]:
        """Obtiene ubicación aproximada desde IP pública."""
        try:
            import geocoder
            
            g = geocoder.ip('me')
            if g.ok and g.latlng:
                return Location(
                    latitude=g.latlng[0],
                    longitude=g.latlng[1],
                    source="ip",
                    accuracy=5000.0,  # ~5km precisión típica de IP
                    timestamp=datetime.now().isoformat()
                )
            return None
        except ImportError:
            logger.warning("geocoder no instalado, IP geolocation no disponible")
            return None
        except Exception as e:
            logger.debug(f"Error obteniendo ubicación de IP: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del proveedor."""
        return {
            "gpsd_available": self._gpsd_available,
            "serial_available": self._serial_available,
            "serial_port": self.serial_port,
            "ip_fallback_enabled": self.use_ip_fallback,
            "cache_seconds": self.cache_seconds,
            "cached_location": self._cached_location.to_dict() if self._cached_location else None
        }
    
    def close(self) -> None:
        """Cierra conexiones."""
        if self._serial_connection:
            try:
                self._serial_connection.close()
            except Exception:
                pass


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.DEBUG)
    
    print("Probando LocationProvider...")
    
    provider = LocationProvider(use_ip_fallback=True)
    location = provider.get_location()
    
    print(f"Ubicación obtenida:")
    print(f"  Latitud: {location.latitude}")
    print(f"  Longitud: {location.longitude}")
    print(f"  Fuente: {location.source}")
    print(f"  Válida: {location.is_valid()}")
    
    print(f"\nStats: {provider.get_stats()}")
    
    provider.close()
    print("\nTest completado")
