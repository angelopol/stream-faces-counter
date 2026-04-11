# Flujo Funcional: Stream Faces Counter

El paquete `stream-faces-counter` es el motor empresarial integral (IoT) responsable de ejecutarse de forma perpetua a bordo de los dispositivos integrados (ej. Raspberry Pi o Jetson Nano) en las unidades de transporte. Su misión es detectar rostros desde un flujo de video (stream en vivo), descartar conductores, identificar pasajeros únicos, empaquetar estos eventos con geolocalización y sincronizarlos de forma asíncrona mediante SQLite hacia la nube.

A continuación, se detalla el ciclo de vida y flujo funcional del software en producción.

---

## 1. Arranque y Configuración (Inicialización)
Al encenderse el dispositivo abordo, el script principal se inicializa instanciando los diversos directores de módulos:
1. **Adquisición de Cámara:** `VideoStream` toma control asíncrono del dispositivo `/dev/video0` sin bloquear el hilo principal.
2. **Sistema de Cache Local:** `LocalBuffer` arma la estructura SQLite (Persistencia), cargando estados previos para prevenir que los datos se pierdan durante apagones o desconexiones.
3. **Carga de Exclusiones Biométrica:** El `FaceTracker` recibe un directorio o base de datos biométrica conteniendo los rostros de los "Dueños, Conductores o Colectores" (`_load_excluded_faces`). Estas caras se fusionan internamente protegidas para nunca ser contadas como pasajeros.
4. **Sincronización:** `CloudSync` utiliza el Token de Hardware (Dirección MAC y Secretos) provisto por el panel Administrador de Transporte ('Transport Admin') para autenticarse, creando un canal cliente-servidor silencioso.

## 2. Lectura y Detección de Movimiento
Para reducir los colapsos de red, el calor computacional de la placa (Thermal Throttling) y los altos cobros por API Cloud:
* El módulo `MotionDetector` extrae constantemente variaciones perimetrales y diferencias visuales entre el cuadro anterior y el de ahora.
* **Hibernación Condicional:** Si la unidad está detenida o no pasa nadie por la puerta (no existe movimiento), el motor omite el envío a las redes generativas e hiberna este segmento de frames.

## 3. Detección Inteligente y Rápida
Una vez que se activa el movimiento (alguien pasa el umbral de la cámara):
1. El analizador enruta la trama de imagen (`frame`) a un detector primario liviano, habitualmente `MediaPipe`.
2. Extrae las caras de los peatones, su confianza y métricas basales para posteriormente enviarlas al subsistema `FaceCounter`.
3. Adicionalmente, el subsistema se cruza con AWS Rekognition cuando los frames prometen ser cruciales, aplicando métricas de calidad y tamaño en base al motor cognitivo nativo.

## 4. Rastreo y Deduplicación (FaceTracker Avanzado)
El momento más importante ocurre cuando un rostro promete ser analizado como persona:
1. **Filtro de Exclusión Biométrico:** `FaceTracker` toma el rostro entrante y lo compara contra un collage de "Hardware Protegido" (El conductor y ayudantes pre cargados). Si el algoritmo empareja similitud, deniega el resto de flujos automáticamente y expulsa este rostro temporal garantizando que los conductores no suban las tarifas.
2. **Evaluación de Duplicados en Tiempo Difiere:** Como el Internet móvil en las rutas es inestable, las caras pueden recaer en un estado Offline. `FaceTracker` está re-escrito para almacenar la caché `FaceCache` si no hay Internet, impidiendo que choquen o cuelguen el script por desconexiones.
3. **Mosaico Condensado:** De estar Online (sin retrasos), la verificación de caras se hace contra todo el mosaico dinámico para saber si la cara entrante es o no un nuevo pasajero abordando la unidad (`is_new_passenger`).
4. **Limpieza Biológica (TTL Cache Expire):** Los pasajeros expiran luego de configurados minutos u horas (`cleanup_expired() / _evict_oldest()`), esto resetea la cara del pasajero para que, si aborda la buseta mañana desde la parada inicial, se vuelva a cobrar el pasaje exitosamente.

## 5. Evento de Abordaje y Geolocalización
1. **Atribución Espacial:** Al catalogar `+1 Pasajero Nuevo`, se gatilla el módulo `LocationProvider`. Empleando chips GPS por serial (`/dev/ttyUSB0`) o simuladores de Red Local/WebIP, el evento captura `(Latitud, Longitud)`.
2. **Almacenaje DB SQLite:** Esta entidad `{FaceID, TimeStamp, Location}` es insertada infaliblemente en la base de datos a base firme `PassengerEventStore`.

## 6. Sincronización en la Nube (CloudSync)
El mecanismo no detiene la cámara mediante guardados síncronos; de eso se encarga un sub-thread (Hilo secundario):
* La clase `CloudSync` cuenta con un `_sync_loop()`. Cada `'x'` segundos despierta comprobando si la unidad posee cobertura e Internet y si `PassengerEventStore` presenta registros pendientes por publicar (is_dirty).
* Ejecuta llamados JSON Batch al Endpoint del Transport Admin backend.
* Si el llamado resulta en `HTTP 200 OK`, el `SyncManager` borra finalmente estas líneas locales de la SQL e indica al sistema que han llegado exitosamente al tablero KPI en el panel administrativo del Dueño de Ruta.
