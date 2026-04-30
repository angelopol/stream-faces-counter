# 🧩 Arquitectura del Algoritmo: Stream Faces Counter (SDK)

Este directorio no es un script ejecutable directo, sino el **Paquete Base (Librería / SDK)** que provee las piezas modulares, algorítmicas y matemáticas sobre las cuales se construye todo el ecosistema Edge del transporte inteligente.

> [!TIP]
> Al separar la lógica en clases puras e independientes, se logra una arquitectura **Desacoplada**, permitiendo cambiar el motor de Inteligencia Artificial o la base de datos sin reescribir el orquestador principal.

---

## ⚙️ Componentes Modulares y sus Algoritmos

El pipeline de procesamiento se divide en varias clases especializadas que ejecutan algoritmos de visión por computadora y gestión de datos.

### 1. `VideoStream` (Subprocesamiento Asíncrono)
Leer una cámara (I/O) suele bloquear el hilo principal de Python, reduciendo los fotogramas drásticamente.
*   **Algoritmo:** Implementa un patrón Productor-Consumidor usando el módulo `threading`. Un hilo en segundo plano captura constantemente fotogramas de `cv2.VideoCapture` y los empuja a una variable compartida de estado o cola.
*   **Resultado:** El hilo principal que hace la inferencia facial siempre recibe el fotograma más reciente instantáneamente, sin latencias de hardware, garantizando fluidez en tiempo real.

### 2. `MotionDetector` (Sustracción de Fondo Dinámica)
Aplicar Inteligencia Artificial a fotogramas donde el autobús está vacío y estacionado es un desperdicio crítico de CPU y energía térmica.
*   **Algoritmo:** Utiliza la técnica de **Diferenciación de Cuadros (Frame Differencing)**.
    1.  Convierte el fotograma a Escala de Grises.
    2.  Aplica un Desenfocado Gaussiano (Blur Kernel) para suavizar imperfecciones e ignorar "ruido" como estática de la cámara o vibraciones menores del motor.
    3.  Calcula el delta absoluto (`cv2.absdiff`) entre el cuadro actual y el fondo de referencia.
    4.  Aplica un umbral binario (`cv2.threshold`) para volver blancos los píxeles que cambiaron drásticamente.
    5.  Calcula los Contornos (`cv2.findContours`). Si el área del polígono supera el umbral (ej. 15,000 píxeles), infiere que hay un humano moviéndose (Motion = True).

### 3. `FaceCounter` (Inferencia Abstrácta y Calidad de Imagen)
*   **Algoritmo:** Funciona como un adaptador (Wrapper). Dependiendo de la configuración del bus, dispara las matrices al motor local (MediaPipe TFLite) o las empaqueta hacia la Nube (AWS Rekognition).
*   Se encarga de ejecutar la primera limpieza matemática: Extraer las *Bounding Boxes* (Cajas Delimitadoras) y validar las métricas de frontalidad.
*   **Filtro de Varianza Laplaciana (Detección de Desenfoque / Sharpness):** Para garantizar la confiabilidad geométrica de los rostros y evitar procesar "fantasmas" por el movimiento del bus, el algoritmo mide la nitidez matemática de la imagen usando la varianza Laplaciana.
    1.  Se extrae la región de interés (el rostro) y se transfiere a escala de grises.
    2.  Se convoluciona la matriz de píxeles con el **Operador Laplaciano** (`cv2.Laplacian`). Esta operación matemática calcula la segunda derivada de la imagen, lo que resalta inmediatamente las regiones de cambios bruscos de intensidad (los bordes definidos como los ojos o la nariz).
    3.  Se calcula la **Varianza** sobre la matriz resultante. Una imagen bien enfocada tendrá bordes muy afilados, lo que se traduce en una varianza muy alta. Una imagen borrosa o movida carecerá de bordes finos, arrojando una varianza matemáticamente baja.
    4.  Si esta puntuación cae por debajo de la barrera de tolerancia (`blur_threshold`), el rostro se clasifica como *Poco Confiable* y se aborta su rastreo.

### 4. `FaceTracker` (Extracción de Vectores y Rastreo)
El componente más denso matemáticamente. Sustituye la lógica anticuada de "Collage" del script de video estático por algoritmos de vanguardia embebidos.
*   **Extracción de Características (Feature Extraction):** En lugar de guardar fotos en el disco duro, extrae un vector numérico (Embeddings de 128 o 512 dimensiones) que representa la topología del rostro.
*   **Algoritmo de Distancia Euclidiana / Similitud del Coseno:** Compara el vector entrante contra todos los vectores en la memoria caché RAM. Si la distancia vectorial es corta (Similitud > 80%), el algoritmo concluye algebraicamente que es el mismo pasajero abordando lentamente.
*   **Filtro de Exclusión (Staff Veto):** Cruza el vector entrante contra un subconjunto de vectores correspondientes a la tripulación autorizada. Si hace "Match", el estado devuelve `is_excluded = True`, bloqueando la facturación.

### 5. `LocalBuffer` y `PassengerEventStore` (Almacenamiento Perimetral)
*   **Algoritmo Store-and-Forward:** Motores basados en SQLite3 transaccional. Mantienen un índice de cola (Queue). Guardan inmediatamente cualquier evento de telemetría junto a las coordenadas devueltas por el `LocationProvider` (vía puerto serial GPS).

---

## 📊 Integración Final
Este paquete funciona como los "órganos" de la solución. Ninguno de estos algoritmos decide por sí solo cuándo ejecutarse, todos quedan a disposición de ser instanciados y llamados secuencialmente por el demonio (Daemon) principal del autobús.

---

**Fuente:** Documentación Técnica de Algoritmos (2026).
