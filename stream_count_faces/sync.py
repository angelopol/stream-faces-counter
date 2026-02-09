"""
sync.py - Sincronización con servidor admin

Proporciona la clase CloudSync para enviar eventos de pasajeros
al sistema administrativo central.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


def get_device_mac() -> str:
    """
    Obtiene la dirección MAC del dispositivo.
    
    Returns:
        Dirección MAC en formato xx:xx:xx:xx:xx:xx
    """
    try:
        from getmac import get_mac_address
        mac = get_mac_address()
        if mac:
            return mac.lower()
    except ImportError:
        logger.warning("getmac no instalado, usando MAC dummy")
    except Exception as e:
        logger.warning(f"Error obteniendo MAC: {e}")
    
    # Fallback: usar MAC basado en hostname
    import socket
    import hashlib
    hostname = socket.gethostname()
    hash_bytes = hashlib.md5(hostname.encode()).hexdigest()[:12]
    return ':'.join(hash_bytes[i:i+2] for i in range(0, 12, 2))


@dataclass
class SyncResult:
    """Resultado de una sincronización."""
    success: bool
    synced_count: int
    error_message: Optional[str] = None
    server_time: Optional[str] = None


class CloudSync:
    """
    Sincroniza eventos locales con el servidor admin.
    
    Envía eventos de pasajeros (con geolocalización) al servidor
    central para análisis y dashboard.
    
    Atributos:
        api_url: URL base del servidor admin
        api_token: Token de autenticación del dispositivo
        device_mac: Dirección MAC del dispositivo
    
    Ejemplo:
        >>> sync = CloudSync(
        ...     api_url="https://admin.example.com/api/v1",
        ...     api_token="abc123...",
        ...     device_mac="aa:bb:cc:dd:ee:ff"
        ... )
        >>> events = [{"timestamp": "...", "passenger_count": 1, ...}]
        >>> result = sync.send_events(events)
        >>> if result.success:
        ...     print(f"Synced {result.synced_count} events")
    """
    
    def __init__(
        self,
        api_url: str,
        api_token: str,
        device_mac: Optional[str] = None,
        timeout: int = 30,
        retry_count: int = 3,
        retry_delay: float = 5.0
    ):
        """
        Inicializa el cliente de sincronización.
        
        Args:
            api_url: URL base del API (ej: https://admin.example.com/api/v1)
            api_token: Token de autenticación único del dispositivo
            device_mac: MAC del dispositivo (auto-detectado si no se provee)
            timeout: Timeout de conexión en segundos
            retry_count: Número de reintentos en caso de fallo
            retry_delay: Segundos entre reintentos
        """
        self.api_url = api_url.rstrip('/')
        self.api_token = api_token
        self.device_mac = device_mac or get_device_mac()
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({
            'X-Device-Token': self.api_token,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })
        
        # Estadísticas
        self._total_synced = 0
        self._total_failed = 0
        self._last_sync_at: Optional[datetime] = None
        self._last_error: Optional[str] = None
        
        logger.info(
            f"CloudSync inicializado: api={self.api_url}, "
            f"mac={self.device_mac}"
        )
    
    def send_events(self, events: List[Dict[str, Any]]) -> SyncResult:
        """
        Envía eventos de pasajeros al servidor.
        
        Args:
            events: Lista de eventos con formato:
                - timestamp: ISO 8601 timestamp
                - passenger_count: Número de pasajeros
                - latitude: Opcional, latitud GPS
                - longitude: Opcional, longitud GPS
                - location_source: 'gps', 'serial', 'ip', 'none'
                - face_id: Opcional, ID del rostro
                
        Returns:
            SyncResult con estado de la sincronización
        """
        if not events:
            return SyncResult(success=True, synced_count=0)
        
        payload = {
            'device_mac': self.device_mac,
            'events': events,
        }
        
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                response = self._session.post(
                    f"{self.api_url}/sync",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    synced = data.get('synced_count', len(events))
                    
                    with self._lock:
                        self._total_synced += synced
                        self._last_sync_at = datetime.now()
                        self._last_error = None
                    
                    logger.info(f"Sincronizados {synced} eventos al servidor")
                    
                    return SyncResult(
                        success=True,
                        synced_count=synced,
                        server_time=data.get('server_time')
                    )
                
                elif response.status_code in [401, 403]:
                    error = "Token inválido o dispositivo desactivado"
                    logger.error(f"Error de autenticación: {error}")
                    self._last_error = error
                    return SyncResult(success=False, synced_count=0, error_message=error)
                
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Error en sync (intento {attempt + 1}): {last_error}")
                    
            except requests.exceptions.Timeout:
                last_error = "Timeout de conexión"
                logger.warning(f"Timeout en sync (intento {attempt + 1})")
                
            except requests.exceptions.ConnectionError:
                last_error = "Sin conexión al servidor"
                logger.warning(f"Sin conexión (intento {attempt + 1})")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error en sync (intento {attempt + 1}): {e}")
            
            # Esperar antes de reintentar
            if attempt < self.retry_count - 1:
                time.sleep(self.retry_delay)
        
        # Todos los reintentos fallaron
        with self._lock:
            self._total_failed += len(events)
            self._last_error = last_error
        
        return SyncResult(
            success=False,
            synced_count=0,
            error_message=last_error
        )
    
    def get_server_status(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado del servidor.
        
        Returns:
            Dict con información del servidor o None si falla
        """
        try:
            response = self._session.get(
                f"{self.api_url}/status",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.warning(f"Error obteniendo status del servidor: {e}")
        
        return None
    
    def is_server_available(self) -> bool:
        """Verifica si el servidor está disponible."""
        status = self.get_server_status()
        return status is not None and status.get('success', False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cliente de sync."""
        with self._lock:
            return {
                'api_url': self.api_url,
                'device_mac': self.device_mac,
                'total_synced': self._total_synced,
                'total_failed': self._total_failed,
                'last_sync_at': self._last_sync_at.isoformat() if self._last_sync_at else None,
                'last_error': self._last_error,
            }


class SyncManager:
    """
    Gestor de sincronización que integra LocalBuffer con CloudSync.
    
    Procesa eventos pendientes del buffer local y los envía
    automáticamente al servidor en segundo plano.
    """
    
    def __init__(
        self,
        cloud_sync: CloudSync,
        local_buffer,  # PassengerEventStore
        batch_size: int = 50,
        sync_interval: float = 60.0
    ):
        """
        Inicializa el gestor de sincronización.
        
        Args:
            cloud_sync: Cliente de sincronización
            local_buffer: Buffer local de eventos (PassengerEventStore)
            batch_size: Eventos a enviar por lote
            sync_interval: Segundos entre sincronizaciones automáticas
        """
        self.cloud_sync = cloud_sync
        self.local_buffer = local_buffer
        self.batch_size = batch_size
        self.sync_interval = sync_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start_background_sync(self) -> None:
        """Inicia sincronización automática en segundo plano."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info("Sincronización en segundo plano iniciada")
    
    def stop_background_sync(self) -> None:
        """Detiene la sincronización en segundo plano."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Sincronización en segundo plano detenida")
    
    def _sync_loop(self) -> None:
        """Loop de sincronización."""
        while self._running:
            try:
                self.sync_pending()
            except Exception as e:
                logger.error(f"Error en sync loop: {e}")
            
            time.sleep(self.sync_interval)
    
    def sync_pending(self) -> SyncResult:
        """
        Sincroniza eventos pendientes del buffer local.
        
        Returns:
            SyncResult con total sincronizado
        """
        pending = self.local_buffer.get_pending_events(limit=self.batch_size)
        
        if not pending:
            return SyncResult(success=True, synced_count=0)
        
        # Convertir a formato de API
        events = [
            {
                'timestamp': event['timestamp'],
                'passenger_count': 1,
                'latitude': event.get('latitude'),
                'longitude': event.get('longitude'),
                'location_source': event.get('location_source', 'none'),
                'face_id': event.get('face_id'),
            }
            for event in pending
        ]
        
        result = self.cloud_sync.send_events(events)
        
        if result.success:
            # Marcar eventos como sincronizados
            event_ids = [event['id'] for event in pending]
            self.local_buffer.mark_synced(event_ids)
        
        return result


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Test CloudSync ===")
    print(f"Device MAC: {get_device_mac()}")
    
    # Test con servidor dummy (no conectará)
    sync = CloudSync(
        api_url="http://localhost:8000/api/v1",
        api_token="test_token_12345"
    )
    
    print(f"Stats: {sync.get_stats()}")
    
    # Intentar sync (fallará sin servidor)
    result = sync.send_events([
        {
            'timestamp': datetime.now().isoformat(),
            'passenger_count': 1,
            'latitude': 10.5,
            'longitude': -66.9,
            'location_source': 'ip',
        }
    ])
    
    print(f"Sync result: success={result.success}, error={result.error_message}")
    print(f"Final stats: {sync.get_stats()}")
