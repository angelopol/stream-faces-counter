"""
storage.py - Almacenamiento local con SQLite (Store-and-Forward)

Proporciona la clase LocalBuffer para persistencia local de eventos,
implementando el patrón Store-and-Forward para resiliencia ante fallas de red.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class LocalBuffer:
    """
    Buffer local para eventos usando SQLite.
    
    Implementa el patrón Store-and-Forward:
    1. Los eventos se guardan localmente en SQLite
    2. Un proceso de sincronización envía eventos pendientes a la nube
    3. Los eventos sincronizados se marcan para posterior limpieza
    
    Esto garantiza que no se pierdan eventos por fallas de red
    o desconexiones temporales del dispositivo.
    
    Atributos:
        db_path: Ruta al archivo de base de datos SQLite
        
    Ejemplo:
        >>> buffer = LocalBuffer("data/events.db")
        >>> buffer.save_event("face_count", {"count": 5, "timestamp": "..."})
        >>> pending = buffer.get_pending_events()
        >>> buffer.mark_synced([e["id"] for e in pending])
    """
    
    def __init__(self, db_path: str = "data/transport_events.db"):
        """
        Inicializa el buffer local.
        
        Args:
            db_path: Ruta al archivo SQLite. Usa ':memory:' para base en memoria.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._persistent_conn = None  # For in-memory databases
        
        # Crear directorio si no existe
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        else:
            # For in-memory databases, create a persistent connection
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        
        # Inicializar base de datos
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager para conexiones a la base de datos.
        
        Yields:
            Conexión SQLite configurada
        """
        if self._persistent_conn is not None:
            # Use persistent connection for in-memory databases
            yield self._persistent_conn
        else:
            # Create new connection for file-based databases
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def _init_database(self) -> None:
        """
        Inicializa las tablas de la base de datos.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabla principal de eventos
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS event_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        synced INTEGER DEFAULT 0,
                        sync_timestamp TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Índices para consultas eficientes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_synced 
                    ON event_queue(synced)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_timestamp 
                    ON event_queue(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_type 
                    ON event_queue(event_type)
                """)
                
                conn.commit()
                logger.info(f"Base de datos inicializada: {self.db_path}")
    
    def save_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> int:
        """
        Guarda un evento en el buffer local.
        
        Args:
            event_type: Tipo de evento (e.g., "face_count", "motion_detected")
            data: Datos del evento como diccionario
            timestamp: Timestamp ISO 8601 (opcional, usa hora actual si no se provee)
            
        Returns:
            ID del evento guardado
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO event_queue (timestamp, event_type, data)
                    VALUES (?, ?, ?)
                    """,
                    (timestamp, event_type, json.dumps(data))
                )
                conn.commit()
                event_id = cursor.lastrowid
                
                logger.debug(f"Evento guardado: id={event_id}, type={event_type}")
                return event_id
    
    def get_pending_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene eventos pendientes de sincronización.
        
        Args:
            limit: Número máximo de eventos a retornar
            event_type: Filtrar por tipo de evento (opcional)
            
        Returns:
            Lista de diccionarios con eventos pendientes
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if event_type:
                    cursor.execute(
                        """
                        SELECT id, timestamp, event_type, data, created_at
                        FROM event_queue
                        WHERE synced = 0 AND event_type = ?
                        ORDER BY timestamp ASC
                        LIMIT ?
                        """,
                        (event_type, limit)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, timestamp, event_type, data, created_at
                        FROM event_queue
                        WHERE synced = 0
                        ORDER BY timestamp ASC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                
                rows = cursor.fetchall()
                events = []
                for row in rows:
                    events.append({
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "event_type": row["event_type"],
                        "data": json.loads(row["data"]),
                        "created_at": row["created_at"]
                    })
                
                return events
    
    def mark_synced(self, event_ids: List[int]) -> int:
        """
        Marca eventos como sincronizados.
        
        Args:
            event_ids: Lista de IDs de eventos a marcar
            
        Returns:
            Número de eventos actualizados
        """
        if not event_ids:
            return 0
        
        sync_timestamp = datetime.now().isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(event_ids))
                cursor.execute(
                    f"""
                    UPDATE event_queue
                    SET synced = 1, sync_timestamp = ?
                    WHERE id IN ({placeholders})
                    """,
                    [sync_timestamp] + event_ids
                )
                conn.commit()
                updated = cursor.rowcount
                
                logger.debug(f"Eventos marcados como sincronizados: {updated}")
                return updated
    
    def cleanup_old_events(self, days: int = 30) -> int:
        """
        Elimina eventos sincronizados antiguos.
        
        Args:
            days: Días a mantener eventos sincronizados
            
        Returns:
            Número de eventos eliminados
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM event_queue
                    WHERE synced = 1 AND sync_timestamp < ?
                    """,
                    (cutoff,)
                )
                conn.commit()
                deleted = cursor.rowcount
                
                if deleted > 0:
                    logger.info(f"Eventos antiguos eliminados: {deleted}")
                return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del buffer.
        
        Returns:
            Diccionario con estadísticas de uso
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total de eventos
                cursor.execute("SELECT COUNT(*) as total FROM event_queue")
                total = cursor.fetchone()["total"]
                
                # Eventos pendientes
                cursor.execute(
                    "SELECT COUNT(*) as pending FROM event_queue WHERE synced = 0"
                )
                pending = cursor.fetchone()["pending"]
                
                # Eventos sincronizados
                synced = total - pending
                
                # Evento más antiguo pendiente
                cursor.execute(
                    """
                    SELECT timestamp FROM event_queue 
                    WHERE synced = 0 
                    ORDER BY timestamp ASC 
                    LIMIT 1
                    """
                )
                oldest_row = cursor.fetchone()
                oldest_pending = oldest_row["timestamp"] if oldest_row else None
                
                # Tipos de eventos
                cursor.execute(
                    """
                    SELECT event_type, COUNT(*) as count 
                    FROM event_queue 
                    GROUP BY event_type
                    """
                )
                event_types = {
                    row["event_type"]: row["count"] 
                    for row in cursor.fetchall()
                }
                
                return {
                    "total_events": total,
                    "pending_events": pending,
                    "synced_events": synced,
                    "oldest_pending": oldest_pending,
                    "event_types": event_types,
                    "db_path": self.db_path
                }
    
    def get_pending_count(self) -> int:
        """
        Obtiene el número de eventos pendientes.
        
        Returns:
            Número de eventos sin sincronizar
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) as count FROM event_queue WHERE synced = 0"
                )
                return cursor.fetchone()["count"]
    
    def clear_all(self) -> int:
        """
        Elimina todos los eventos (usar con precaución).
        
        Returns:
            Número de eventos eliminados
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM event_queue")
                conn.commit()
                deleted = cursor.rowcount
                
                logger.warning(f"Todos los eventos eliminados: {deleted}")
                return deleted


if __name__ == "__main__":
    # Test básico
    logging.basicConfig(level=logging.DEBUG)
    
    print("Probando LocalBuffer...")
    
    # Usar base de datos en memoria para test
    buffer = LocalBuffer(":memory:")
    
    # Guardar algunos eventos
    id1 = buffer.save_event("face_count", {"count": 3, "location": "entrance"})
    id2 = buffer.save_event("motion_detected", {"area": 5000})
    id3 = buffer.save_event("face_count", {"count": 1, "location": "entrance"})
    
    print(f"Eventos guardados: {id1}, {id2}, {id3}")
    
    # Obtener estadísticas
    stats = buffer.get_stats()
    print(f"Estadísticas: {stats}")
    
    # Obtener eventos pendientes
    pending = buffer.get_pending_events()
    print(f"Eventos pendientes: {len(pending)}")
    for event in pending:
        print(f"  - {event['event_type']}: {event['data']}")
    
    # Marcar como sincronizados
    buffer.mark_synced([id1, id2])
    
    # Verificar estado
    stats = buffer.get_stats()
    print(f"Después de sincronizar: pending={stats['pending_events']}")
    
    print("Test completado")
