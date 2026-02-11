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
from typing import List, Dict, Any, Optional, Tuple
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


class FaceCache:
    """
    Cache de rostros para procesamiento offline.
    
    Cuando AWS Rekognition no está disponible (sin internet),
    los rostros se almacenan localmente para su posterior
    procesamiento cuando la conexión se restablezca.
    
    Características:
    - Almacena imágenes de rostros como blobs SQLite
    - Soporta procesamiento por lotes cuando hay conexión
    - Evita duplicados mediante hash de imagen
    - Limpieza automática de rostros procesados
    
    Ejemplo:
        >>> cache = FaceCache("data/face_cache.db")
        >>> cache.store_pending(face_bytes, event_timestamp)
        >>> pending = cache.get_pending_faces(limit=10)
        >>> for face in pending:
        ...     # Procesar con AWS
        ...     cache.mark_processed(face["id"], is_new=True)
    """
    
    def __init__(self, db_path: str = "data/face_cache.db"):
        """
        Inicializa el cache de rostros.
        
        Args:
            db_path: Ruta al archivo SQLite. Usa ':memory:' para test.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._persistent_conn = None
        
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        else:
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Context manager para conexiones."""
        if self._persistent_conn is not None:
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def _init_database(self) -> None:
        """Inicializa las tablas de la base de datos."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabla de rostros pendientes
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pending_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_hash TEXT NOT NULL,
                        image_data BLOB NOT NULL,
                        timestamp TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        processed INTEGER DEFAULT 0,
                        is_new_passenger INTEGER DEFAULT NULL,
                        face_id TEXT DEFAULT NULL,
                        is_excluded INTEGER DEFAULT NULL,
                        error_message TEXT DEFAULT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        processed_at TEXT DEFAULT NULL
                    )
                """)
                
                # Índices
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pending_processed 
                    ON pending_faces(processed)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pending_hash 
                    ON pending_faces(image_hash)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pending_timestamp 
                    ON pending_faces(timestamp)
                """)
                
                conn.commit()
                logger.info(f"FaceCache inicializado: {self.db_path}")
    
    def _compute_hash(self, image_data: bytes) -> str:
        """Calcula hash de imagen para deduplicación."""
        import hashlib
        return hashlib.sha256(image_data).hexdigest()[:16]
    
    def store_pending(
        self, 
        image_data: bytes, 
        timestamp: Optional[str] = None
    ) -> Optional[int]:
        """
        Almacena un rostro pendiente de procesamiento.
        
        Args:
            image_data: Imagen del rostro en bytes (JPEG)
            timestamp: Timestamp del evento (opcional)
            
        Returns:
            ID del registro, o None si ya existe (duplicado)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        image_hash = self._compute_hash(image_data)
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Verificar si ya existe este rostro (por hash)
                cursor.execute(
                    "SELECT id FROM pending_faces WHERE image_hash = ? AND processed = 0",
                    (image_hash,)
                )
                existing = cursor.fetchone()
                if existing:
                    logger.debug(f"Rostro duplicado ignorado: hash={image_hash[:8]}")
                    return None
                
                # Insertar nuevo
                cursor.execute(
                    """
                    INSERT INTO pending_faces (image_hash, image_data, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (image_hash, image_data, timestamp)
                )
                conn.commit()
                face_id = cursor.lastrowid
                
                logger.debug(f"Rostro pendiente almacenado: id={face_id}, hash={image_hash[:8]}")
                return face_id
    
    def get_pending_faces(
        self, 
        limit: int = 50, 
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Obtiene rostros pendientes de procesamiento.
        
        Args:
            limit: Número máximo de rostros a retornar
            max_retries: Máximo de reintentos antes de descartar
            
        Returns:
            Lista de diccionarios con rostros pendientes
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, image_hash, image_data, timestamp, retry_count, created_at
                    FROM pending_faces
                    WHERE processed = 0 AND retry_count < ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (max_retries, limit)
                )
                
                rows = cursor.fetchall()
                faces = []
                for row in rows:
                    faces.append({
                        "id": row["id"],
                        "image_hash": row["image_hash"],
                        "image_data": row["image_data"],
                        "timestamp": row["timestamp"],
                        "retry_count": row["retry_count"],
                        "created_at": row["created_at"]
                    })
                
                return faces
    
    def increment_retry(self, face_id: int) -> None:
        """
        Incrementa el contador de reintentos.
        
        Args:
            face_id: ID del rostro
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE pending_faces SET retry_count = retry_count + 1 WHERE id = ?",
                    (face_id,)
                )
                conn.commit()
    
    def mark_processed(
        self,
        face_id: int,
        is_new: bool,
        tracked_face_id: Optional[str] = None,
        is_excluded: bool = False,
        error_message: Optional[str] = None
    ) -> None:
        """
        Marca un rostro como procesado.
        
        Args:
            face_id: ID del registro en cache
            is_new: True si es nuevo pasajero
            tracked_face_id: ID asignado por FaceTracker
            is_excluded: True si es personal autorizado
            error_message: Mensaje de error si falló
        """
        processed_at = datetime.now().isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE pending_faces 
                    SET processed = 1,
                        is_new_passenger = ?,
                        face_id = ?,
                        is_excluded = ?,
                        error_message = ?,
                        processed_at = ?
                    WHERE id = ?
                    """,
                    (
                        1 if is_new else 0,
                        tracked_face_id,
                        1 if is_excluded else 0,
                        error_message,
                        processed_at,
                        face_id
                    )
                )
                conn.commit()
                logger.debug(f"Rostro procesado: id={face_id}, is_new={is_new}, is_excluded={is_excluded}")
    
    def mark_failed(self, face_id: int, error_message: str) -> None:
        """
        Marca un rostro como fallido pero incrementa reintento.
        
        Args:
            face_id: ID del registro
            error_message: Descripción del error
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE pending_faces 
                    SET retry_count = retry_count + 1,
                        error_message = ?
                    WHERE id = ?
                    """,
                    (error_message, face_id)
                )
                conn.commit()
    
    def get_pending_count(self) -> int:
        """Obtiene el número de rostros pendientes."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) as count FROM pending_faces WHERE processed = 0"
                )
                return cursor.fetchone()["count"]
    
    def cleanup_processed(self, days: int = 7) -> int:
        """
        Elimina rostros procesados antiguos.
        
        Args:
            days: Días a mantener rostros procesados
            
        Returns:
            Número de registros eliminados
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM pending_faces
                    WHERE processed = 1 AND processed_at < ?
                    """,
                    (cutoff,)
                )
                conn.commit()
                deleted = cursor.rowcount
                
                if deleted > 0:
                    logger.info(f"Rostros procesados antiguos eliminados: {deleted}")
                return deleted
    
    def cleanup_failed(self, max_retries: int = 3) -> int:
        """
        Elimina rostros que excedieron reintentos.
        
        Args:
            max_retries: Máximo de reintentos permitidos
            
        Returns:
            Número de registros eliminados
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM pending_faces
                    WHERE processed = 0 AND retry_count >= ?
                    """,
                    (max_retries,)
                )
                conn.commit()
                deleted = cursor.rowcount
                
                if deleted > 0:
                    logger.warning(f"Rostros fallidos eliminados: {deleted}")
                return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) as total FROM pending_faces")
                total = cursor.fetchone()["total"]
                
                cursor.execute(
                    "SELECT COUNT(*) as pending FROM pending_faces WHERE processed = 0"
                )
                pending = cursor.fetchone()["pending"]
                
                cursor.execute(
                    "SELECT COUNT(*) as processed FROM pending_faces WHERE processed = 1"
                )
                processed = cursor.fetchone()["processed"]
                
                cursor.execute(
                    """
                    SELECT COUNT(*) as new_passengers 
                    FROM pending_faces 
                    WHERE processed = 1 AND is_new_passenger = 1
                    """
                )
                new_passengers = cursor.fetchone()["new_passengers"]
                
                cursor.execute(
                    """
                    SELECT SUM(LENGTH(image_data)) as total_bytes 
                    FROM pending_faces 
                    WHERE processed = 0
                    """
                )
                row = cursor.fetchone()
                pending_bytes = row["total_bytes"] if row["total_bytes"] else 0
                
                return {
                    "total_faces": total,
                    "pending_faces": pending,
                    "processed_faces": processed,
                    "new_passengers_found": new_passengers,
                    "pending_storage_mb": round(pending_bytes / (1024 * 1024), 2),
                    "db_path": self.db_path
                }
    
    def clear_all(self) -> int:
        """Elimina todos los registros."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM pending_faces")
                conn.commit()
                deleted = cursor.rowcount
                logger.warning(f"FaceCache limpiado: {deleted} registros")
                return deleted


class TrackerPersistence:
    """
    Persistencia del estado del FaceTracker (collages y rostros activos).
    Permite recuperar el estado después de un reinicio.
    """
    
    def __init__(self, db_path: str = "data/face_cache.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            
        self._init_database()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def _init_database(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabla de collages
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracker_collages (
                        idx INTEGER PRIMARY KEY,
                        image_data BLOB NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla de rostros rastreados
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracker_faces (
                        face_id TEXT PRIMARY KEY,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        collage_idx INTEGER,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info(f"TrackerPersistence inicializado: {self.db_path}")

    def save_state(self, collages: List[bytes], faces: List[Dict[str, Any]]) -> None:
        """
        Guarda el estado completo (sobrescribe lo anterior).
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Limpiar todo
                cursor.execute("DELETE FROM tracker_collages")
                cursor.execute("DELETE FROM tracker_faces")
                
                # Guardar collages
                for idx, img_data in enumerate(collages):
                    cursor.execute(
                        "INSERT INTO tracker_collages (idx, image_data) VALUES (?, ?)",
                        (idx, img_data)
                    )
                
                # Guardar rostros
                for face in faces:
                    cursor.execute(
                        """
                        INSERT INTO tracker_faces (face_id, first_seen, last_seen, collage_idx)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            face['face_id'], 
                            face['first_seen'], 
                            face['last_seen'], 
                            face['collage_idx']
                        )
                    )
                
                conn.commit()
                logger.info(f"Estado del tracker guardado: {len(collages)} collages, {len(faces)} rostros")

    def load_state(self) -> Tuple[Dict[int, bytes], List[Dict[str, Any]]]:
        """
        Carga el estado guardado.
        
        Returns:
            (collages_dict, faces_list)
        """
        collages = {}
        faces = []
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Cargar collages
                cursor.execute("SELECT idx, image_data FROM tracker_collages ORDER BY idx ASC")
                for row in cursor.fetchall():
                    collages[row[0]] = row[1]
                
                # Cargar rostros
                cursor.execute("SELECT face_id, first_seen, last_seen, collage_idx FROM tracker_faces")
                for row in cursor.fetchall():
                    faces.append({
                        'face_id': row[0],
                        'first_seen': row[1],
                        'last_seen': row[2],
                        'collage_idx': row[3]
                    })
                    
        return collages, faces
        
    def clear(self):
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tracker_collages")
                cursor.execute("DELETE FROM tracker_faces")
                conn.commit()



class PassengerEventStore:
    """
    Almacenamiento de eventos de abordaje de pasajeros con geolocalización.
    
    Guarda timestamp, face_id, y coordenadas GPS para cada pasajero
    que aborda el vehículo. Permite análisis estadístico de flujo
    de pasajeros por ubicación (paradas).
    
    Las coordenadas son opcionales - si no hay GPS disponible,
    se almacenan como NULL pero el timestamp siempre se guarda.
    
    Ejemplo:
        >>> store = PassengerEventStore("data/passengers.db")
        >>> store.record_boarding("abc123", 10.5, -66.9, "gps")
        >>> stats = store.get_location_stats()
    """
    
    def __init__(self, db_path: str = "data/passenger_events.db"):
        """
        Inicializa el almacenamiento de eventos.
        
        Args:
            db_path: Ruta al archivo SQLite
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._persistent_conn = None
        
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        else:
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Context manager para conexiones."""
        if self._persistent_conn is not None:
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def _init_database(self) -> None:
        """Inicializa las tablas de la base de datos."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabla de eventos de abordaje
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS passenger_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        face_id TEXT,
                        latitude REAL,
                        longitude REAL,
                        location_source TEXT,
                        location_accuracy REAL,
                        synced INTEGER DEFAULT 0,
                        sync_timestamp TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Índices para consultas eficientes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_passenger_timestamp 
                    ON passenger_events(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_passenger_location 
                    ON passenger_events(latitude, longitude)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_passenger_synced 
                    ON passenger_events(synced)
                """)
                
                conn.commit()
                logger.info(f"PassengerEventStore inicializado: {self.db_path}")
    
    def record_boarding(
        self,
        face_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_source: str = "none",
        location_accuracy: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Registra un evento de abordaje de pasajero.
        
        Args:
            face_id: ID del rostro del pasajero (opcional)
            latitude: Latitud GPS (puede ser None)
            longitude: Longitud GPS (puede ser None)
            location_source: Fuente de ubicación ('gps', 'serial', 'ip', 'none')
            location_accuracy: Precisión en metros (opcional)
            timestamp: Timestamp ISO 8601 (default: ahora)
            
        Returns:
            ID del evento registrado
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO passenger_events 
                    (timestamp, face_id, latitude, longitude, location_source, location_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, face_id, latitude, longitude, location_source, location_accuracy)
                )
                conn.commit()
                event_id = cursor.lastrowid
                
                logger.debug(
                    f"Abordaje registrado: id={event_id}, "
                    f"face={face_id[:8] if face_id else 'N/A'}, "
                    f"loc={f'{latitude:.4f},{longitude:.4f}' if latitude else 'None'}, "
                    f"source={location_source}"
                )
                return event_id
    
    def get_recent_events(
        self, 
        limit: int = 100,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene eventos de abordaje recientes.
        
        Args:
            limit: Número máximo de eventos
            hours: Filtrar por últimas N horas (opcional)
            
        Returns:
            Lista de eventos de abordaje
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if hours:
                    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                    cursor.execute(
                        """
                        SELECT * FROM passenger_events
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (cutoff, limit)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM passenger_events
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
    
    def get_pending_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene eventos pendientes de sincronizar."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM passenger_events
                    WHERE synced = 0
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (limit,)
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
    
    def mark_synced(self, event_ids: List[int]) -> int:
        """Marca eventos como sincronizados."""
        if not event_ids:
            return 0
        
        sync_timestamp = datetime.now().isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(event_ids))
                cursor.execute(
                    f"""
                    UPDATE passenger_events
                    SET synced = 1, sync_timestamp = ?
                    WHERE id IN ({placeholders})
                    """,
                    [sync_timestamp] + event_ids
                )
                conn.commit()
                return cursor.rowcount
    
    def get_location_stats(
        self, 
        precision: int = 3,
        hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene estadísticas de pasajeros agrupadas por ubicación.
        
        Agrupa eventos cercanos (mismo redondeo de coordenadas)
        para identificar "paradas" aproximadas.
        
        Args:
            precision: Decimales para agrupar (3 = ~100m)
            hours: Filtrar por últimas N horas
            
        Returns:
            Lista de ubicaciones con conteo de pasajeros
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Usar ROUND para agrupar ubicaciones cercanas
                if hours:
                    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                    cursor.execute(
                        f"""
                        SELECT 
                            ROUND(latitude, {precision}) as lat_group,
                            ROUND(longitude, {precision}) as lng_group,
                            COUNT(*) as passenger_count,
                            MIN(timestamp) as first_boarding,
                            MAX(timestamp) as last_boarding
                        FROM passenger_events
                        WHERE latitude IS NOT NULL 
                          AND longitude IS NOT NULL
                          AND timestamp >= ?
                        GROUP BY lat_group, lng_group
                        ORDER BY passenger_count DESC
                        """,
                        (cutoff,)
                    )
                else:
                    cursor.execute(
                        f"""
                        SELECT 
                            ROUND(latitude, {precision}) as lat_group,
                            ROUND(longitude, {precision}) as lng_group,
                            COUNT(*) as passenger_count,
                            MIN(timestamp) as first_boarding,
                            MAX(timestamp) as last_boarding
                        FROM passenger_events
                        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                        GROUP BY lat_group, lng_group
                        ORDER BY passenger_count DESC
                        """
                    )
                
                rows = cursor.fetchall()
                return [
                    {
                        "latitude": row["lat_group"],
                        "longitude": row["lng_group"],
                        "passenger_count": row["passenger_count"],
                        "first_boarding": row["first_boarding"],
                        "last_boarding": row["last_boarding"]
                    }
                    for row in rows
                ]
    
    def get_hourly_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Obtiene estadísticas de pasajeros por hora del día.
        
        Útil para identificar horas pico.
        
        Args:
            days: Días hacia atrás a considerar
            
        Returns:
            Lista con conteo por hora (0-23)
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                        COUNT(*) as passenger_count
                    FROM passenger_events
                    WHERE timestamp >= ?
                    GROUP BY hour
                    ORDER BY hour
                    """,
                    (cutoff,)
                )
                
                rows = cursor.fetchall()
                # Completar horas faltantes con 0
                hourly = {row["hour"]: row["passenger_count"] for row in rows}
                return [
                    {"hour": h, "passenger_count": hourly.get(h, 0)}
                    for h in range(24)
                ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total de eventos
                cursor.execute("SELECT COUNT(*) as total FROM passenger_events")
                total = cursor.fetchone()["total"]
                
                # Eventos con ubicación
                cursor.execute(
                    """
                    SELECT COUNT(*) as with_location 
                    FROM passenger_events 
                    WHERE latitude IS NOT NULL
                    """
                )
                with_location = cursor.fetchone()["with_location"]
                
                # Por fuente de ubicación
                cursor.execute(
                    """
                    SELECT location_source, COUNT(*) as count
                    FROM passenger_events
                    GROUP BY location_source
                    """
                )
                by_source = {row["location_source"]: row["count"] for row in cursor.fetchall()}
                
                # Pendientes de sincronizar
                cursor.execute(
                    "SELECT COUNT(*) as pending FROM passenger_events WHERE synced = 0"
                )
                pending = cursor.fetchone()["pending"]
                
                # Evento más reciente
                cursor.execute(
                    "SELECT timestamp FROM passenger_events ORDER BY timestamp DESC LIMIT 1"
                )
                latest_row = cursor.fetchone()
                latest = latest_row["timestamp"] if latest_row else None
                
                return {
                    "total_events": total,
                    "events_with_location": with_location,
                    "events_without_location": total - with_location,
                    "location_rate": round(with_location / total * 100, 1) if total > 0 else 0,
                    "by_source": by_source,
                    "pending_sync": pending,
                    "latest_event": latest,
                    "db_path": self.db_path
                }
    
    def cleanup_old_events(self, days: int = 90) -> int:
        """
        Elimina eventos antiguos sincronizados.
        
        Args:
            days: Días a mantener
            
        Returns:
            Número de eventos eliminados
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM passenger_events
                    WHERE synced = 1 AND timestamp < ?
                    """,
                    (cutoff,)
                )
                conn.commit()
                deleted = cursor.rowcount
                
                if deleted > 0:
                    logger.info(f"Eventos de pasajeros antiguos eliminados: {deleted}")
                return deleted
    
    def clear_all(self) -> int:
        """Elimina todos los eventos."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM passenger_events")
                conn.commit()
                deleted = cursor.rowcount
                logger.warning(f"PassengerEventStore limpiado: {deleted} eventos")
                return deleted


