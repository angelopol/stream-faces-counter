# Stream Count Faces

Librería Python para conteo de rostros en tiempo real en streams de video usando AWS Rekognition.

## Componentes

- **VideoStream** (`camera.py`): Captura de video no bloqueante con threading
- **MotionDetector** (`motion.py`): Detección de movimiento por diferenciación de frames
- **LocalBuffer** (`storage.py`): Almacenamiento SQLite con patrón Store-and-Forward
- **FaceCounter** (`detector.py`): Wrapper para AWS Rekognition con filtros de calidad
- **FaceTracker** (`tracker.py`): Rastreo de pasajeros para deduplicación con TTL configurable

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/angelopol/stream-faces-counter.git
cd stream-faces-counter

# Instalar como paquete editable (desarrollo)
pip install -e .

# O instalar directamente
pip install .
```

## Configuración de AWS

### Opción 1: Archivo .env (recomendado)

```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar con tus credenciales
```

Contenido del `.env`:
```
AWS_ACCESS_KEY_ID=tu_access_key_aqui
AWS_SECRET_ACCESS_KEY=tu_secret_key_aqui
AWS_REGION=us-east-1
```

### Opción 2: Parámetros directos

```python
counter = FaceCounter(
    aws_access_key_id="AKIA...",
    aws_secret_access_key="...",
    region="us-east-1"
)
```

### Opción 3: AWS CLI

```bash
aws configure
```

## Uso

```python
from stream_count_faces import VideoStream, MotionDetector, LocalBuffer, FaceCounter

# Captura de video
stream = VideoStream(source=0)
stream.start()

# Detección de movimiento
motion = MotionDetector(min_area=5000)

# Contador de rostros (usa credenciales de .env automáticamente)
counter = FaceCounter(dry_run=False)

# Modo desarrollo sin AWS
counter_dev = FaceCounter(dry_run=True)

# Buffer local
buffer = LocalBuffer("events.db")

# Bucle principal
while True:
    frame = stream.read()
    if motion.detect(frame):
        faces = counter.count_faces(frame)
        if faces:
            buffer.save_event("face_count", {"count": len(faces)})
```

## Licencia

MIT License
