# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # ⚠️ Etapa 1: funciones a implementar
│       │   ├── classifier_service.py   # ⚠️ Etapa 2: funciones a implementar
│       │   ├── detection_service.py    # ⚠️ Etapa 3: funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configurá tus modelos

Entrená tus modelos (Etapa 2) con `2-train.ipynb` (en Google Colab, con GPU) y guardá los checkpoints dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

### A implementar por el grupo

Funciones que la cátedra dejó para completar:

| Etapa | Función | Archivo |
|-------|---------|---------|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |

En las Etapas 1 y 3 el trabajo se limita a completar las funciones. En la Etapa 2, además, el grupo entrena sus propios modelos (Modelo A: fine-tuning de ResNet18/ResNet50; Modelo B: CNN propia, **ambos obligatorios**) y hace un estudio de resultados.

Ver el contrato de cada función, recomendaciones y errores frecuentes en [ENUNCIADO.md](ENUNCIADO.md#24-funciones-a-implementar).

### Implementado por el grupo

Las ocho funciones de la tabla están implementadas (no queda ningún `NotImplementedError`):

| Etapa | Función | Implementación |
|-------|---------|----------------|
| 1 | `extract_embedding(image)` | ResNet18 preentrenada en ImageNet (`IMAGENET1K_V1`) sin la capa final (`fc = Identity`): vector de 512 del average pooling, normalizado a norma 1. El modelo se carga una sola vez. |
| 1 | `search_similar_images(embedding, top_k)` | Búsqueda en pgvector (`store.search`) o recorrido del JSON (`store.all()`), score con la métrica configurada (`SIMILARITY_METRIC`), orden descendente y Top-K. |
| 1 | `predict_breed_from_neighbors(results)` | `unknown` si el vecino más parecido no supera `SIMILARITY_THRESHOLD`; si no, voto ponderado por similitud. El score es la proporción del voto de la raza ganadora. |
| 2 | `train_classifier()` | Entrena el modelo activo (ResNet18 fine-tuned o CNN propia) con los splits de `data/splits.csv`, augmentation solo en train, AdamW, scheduler coseno y label smoothing. Guarda el mejor checkpoint según la accuracy de valid (con la lista de razas en `model.classes`), corta con early stopping y guarda el historial en `output/`. |
| 2 | `evaluate_classifier()` | Accuracy y precision, recall, specificity y F1 macro (one-vs-rest, con `metrics.py`) sobre un split (`test` por defecto); guarda las predicciones por imagen en `last_predictions`. |
| 2 | `extract_custom_embedding(image)` | Embedding de los modelos entrenados: `model.embed()` en la CNN propia (capa de 512 antes del clasificador) y todas las capas menos `fc` en la ResNet18; normalizado a norma 1. |
| 3 | `detect_dogs(image)` | YOLOv8n preentrenado en COCO, cargado una sola vez, filtrando la clase `dog` (16) con `YOLO_CONF_THRESHOLD`. Devuelve cajas en píxeles enteros con su confianza, o una lista vacía. |
| 3 | `classify_detected_dog(crop)` | Modelo activo de la Etapa 2 con el mismo preprocesamiento del entrenamiento, en `eval()` y `no_grad()`; softmax, raza más probable y su probabilidad. |

Módulos auxiliares agregados:

- `src/lib/preprocessing.py`: preprocesamiento compartido por las notebooks y los servicios (BGR → RGB, resize a 224, normalización ImageNet) y el pipeline de data augmentation.
- `src/lib/training.py`: hiperparámetros por modelo (`DEFAULT_HPARAMS`), lectura de `data/splits.csv` y el `Dataset` de entrenamiento.
- `src/lib/models/cnn_custom.py`: la arquitectura del Modelo B (`CNNCustom`). Está en un módulo para que la app pueda cargar el checkpoint completo con `torch.load`.

Flujo de la app con la configuración entregada: la Etapa 1 busca con los embeddings del Modelo A (`EMBEDDING_MODEL=resnet18_finetuned`, umbral 0,822); la Etapa 2 clasifica con el modelo elegido en la interfaz (por defecto `resnet18_finetuned`); la Etapa 3 detecta con YOLOv8n y clasifica cada recorte con el Modelo A.

### Orden de trabajo sugerido

1. `1-pre-processing.ipynb`: EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y splits.
2. Etapa 1 en `similarity_service.py`. Indexá la galería con `python scripts/build_index.py --split train` y probá la pestaña de la Etapa 1.
3. Etapa 2 en `classifier_service.py`, entrenando y comparando los modelos con `2-train.ipynb`. Descargá los checkpoints a `models/`.
4. Etapa 3 en `detection_service.py` y validación end-to-end con `docker compose up`.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Dataset

Detalle completo en `1-pre-processing.ipynb`.

**Dataset y splits.** 70 Dog Breeds Image Dataset (Kaggle): 70 razas, imágenes JPEG RGB de 224×224. Se usaron los splits originales después del filtrado; los splits finales quedan en `data/splits.csv` (lo leen la notebook 2 y `train_classifier`) y las imágenes descartadas, con su motivo, en `data/descartadas.csv`.

| Split | Original | Después del filtrado | Imágenes por raza |
|-------|----------|----------------------|-------------------|
| train | 7946 | 7652 | 63 a 190 |
| valid | 700 | 685 | 8 a 10 |
| test | 700 | 696 | 8 a 10 |
| independiente | — | 184 (22 razas) | 5 a 10 |

El desbalance de train es moderado (la raza mayor tiene unas 3 veces más imágenes que la menor). Se reportan métricas macro y se probaron pesos por clase en la loss (sin mejora).

**Reglas de filtrado y calidad.**

| Regla | Criterio | Descartadas |
|-------|----------|-------------|
| 1. Corrupta | El archivo no se puede abrir | 0 |
| 2. Baja resolución | Menor a 224×224 | 0 |
| 3. Borrosa (solo train) | Varianza del Laplaciano < 50 | 19 |
| 4. Exposición extrema (solo train) | Brillo medio menor a 30 o mayor a 225, con contraste menor a 20 | 0 |
| 5. Duplicada | dHash de 64 bits (Hamming ≤ 10) verificado con correlación ≥ 0,90 sobre 64×64 en grises; se agrupan en clusters y se conserva una copia con prioridad test > valid > train | 289 (270 train, 15 valid, 4 test) |
| 6. Conflicto de etiqueta | La misma imagen con razas distintas | 10 (train; 5 también eran duplicadas) |

Se encontraron 327 pares de duplicados, 130 de ellos entre splits distintos (61 train-test, 59 train-valid, 10 valid-test): sin el filtrado habría fuga de datos. El umbral de borrosas es conservador porque un percentil habría descartado más del 20 % de razas de pelo claro (Bichon Frise, Maltese, Pomeranian); con 50 ninguna raza pierde más del 2,7 %. Las imágenes en escala de grises (30, el 0,3 %) se conservan. La etiqueta `American  Spaniel` (doble espacio en valid) se normalizó.

**Conjunto independiente de evaluación.** Imágenes de Wikimedia Commons con licencia Creative Commons o de dominio público; para cada una se registraron la URL de origen, el autor y la licencia en `data/dataset/independent/fuentes.csv`. No se suben al repositorio.

- 184 imágenes de un perro de 22 razas (entre 5 y 10 por raza): Beagle, Bichon Frise, Border Collie, Boston Terrier, Chihuahua, Cockapoo, Collie, Dalmation, French Bulldog, German Sheperd, Golden Retriever, Komondor, Labradoodle, Labrador, Lhasa, Malinois, Maltese, Pug, Rhodesian, Shih-Tzu, Siberian Husky y Vizsla.
- 16 escenas con varios perros, personas u objetos, y 21 imágenes sin perros, para la Etapa 3.
- Link de descarga (solo lectura): [carpeta tp2 en Google Drive](https://drive.google.com/drive/folders/1SnNhPsoO0mt7Gb1PWTMirqP_Ib5pN4IF?usp=sharing), archivo `independent.zip`. Se descomprime en `data/dataset/independent/`.

**Preprocesamiento.** Un único módulo compartido (`src/lib/preprocessing.py`): BGR → RGB, resize directo a 224×224 y normalización con la media y el desvío de ImageNet.

**Data augmentation (solo train, con albumentations).** Flip horizontal (p = 0,5); rotación ±15° con bordes reflejados (p = 0,5); recorte aleatorio de 75–100 % del área, proporción 3:4 a 4:3 (p = 0,5), que simula los recortes de YOLO; brillo y contraste ±20 % (p = 0,5); blur gaussiano de 3–5 px (p = 0,2); ruido gaussiano (p = 0,2) y escala de grises (p = 0,05). Se descartaron el flip vertical y los cambios de tono, porque alteran la apariencia de la raza.

## Modelos y entrenamiento

Detalle completo en `2-train.ipynb` (entrenada en Google Colab con GPU T4).

**Embeddings baseline (Etapa 1).** ResNet18 preentrenada en ImageNet sin la capa final: vector de 512 normalizado a norma 1 y similitud coseno. ImageNet incluye más de 100 razas de perros, así que sus features ya separan bien las razas (accuracy 0,932 en test por voto de vecinos) sin entrenar nada.

**Modelo A: ResNet18 con fine-tuning.** Se reemplazó la capa final por una de 70 salidas. Se compararon dos estrategias en valid: feature extraction (solo la capa nueva entrenable), con accuracy 0,931, y fine-tuning completo con learning rates diferenciados (0,001 para la capa nueva y 0,0001 para las preentrenadas, para no destruir lo aprendido en ImageNet), con 0,969. Se eligió el fine-tuning completo.

**Modelo B: CNN propia desde cero.** Cinco bloques convolucionales (dos convoluciones 3×3 con BatchNorm y ReLU, y max pooling) con 32, 64, 128, 256 y 256 filtros; Global Average Pooling; capa de embedding de 512 (lineal + BatchNorm + ReLU); dropout 0,3 y clasificador de 70 salidas. 2,52 millones de parámetros. El GAP evita una capa densa de 6,4 millones de parámetros, y el embedding de 512 hace que sea intercambiable con la ResNet18 en la base de vectores. Se compararon en valid la versión base (0,866) y una con pesos por clase en la loss (0,861); se eligió la base.

**Hiperparámetros** (en `DEFAULT_HPARAMS`, modificables desde la notebook; semilla 42 en ambos):

| | Modelo A | Modelo B |
|---|---|---|
| Épocas (máximo) | 10 | 40 |
| Batch | 64 | 64 |
| Optimizador | AdamW | AdamW |
| Learning rate | 0,001 (capa nueva) / 0,0001 (resto) | 0,001 |
| Weight decay | 0,0001 | 0,0001 |
| Scheduler | Coseno | Coseno |
| Label smoothing | 0,1 | 0,1 |
| Dropout | — | 0,3 |
| Early stopping (paciencia) | 3 | 6 |

En cada época se guarda el checkpoint solo si mejora la accuracy de valid.

**Detector (Etapa 3).** YOLOv8n preentrenado en COCO, clase `dog` (16), `YOLO_CONF_THRESHOLD = 0.25`. Se compararon 0,10, 0,25 y 0,50 con el conjunto independiente (sección 14 de `2-train.ipynb`): 0,25 es el único valor que detecta el perro en el 100 % de las fotos de un perro (con 0,50 se pierden 3 de 22) y reduce los falsos positivos respecto de 0,10 (de 5 a 3 imágenes sin perros con detecciones).

**Umbral de similitud.** `SIMILARITY_THRESHOLD = 0.822` con `EMBEDDING_MODEL = resnet18_finetuned`, similitud coseno y `TOP_K = 10`. Regla: percentil 2 del score del vecino más parecido en valid, es decir, se acepta rechazar como `unknown` al 2 % de los perros reales. Con los embeddings del baseline la misma regla da 0,836. Con 0,822 se rechaza el 81 % de las imágenes sin perros (17 de 21; el baseline rechazaba el 95 %).

**Checkpoints.** Link público de solo lectura: [carpeta tp2 en Google Drive](https://drive.google.com/drive/folders/1SnNhPsoO0mt7Gb1PWTMirqP_Ib5pN4IF?usp=sharing), subcarpeta `models/`. Contiene `resnet18_finetuned.pth` y `cnn_custom.pth` (los modelos elegidos) y los checkpoints de los dos experimentos del Modelo B.

Ubicación esperada: `models/resnet18_finetuned.pth` y `models/cnn_custom.pth`. Los pesos de YOLO (`models/yolov8n.pt`) se descargan solos en el primer uso.

## Resultados

**Etapa 1: búsqueda por similitud** (galería: train; consultas: test y conjunto independiente; TOP_K = 10; coseno).

| Embeddings | Dim | NDCG@10 test | Precision@10 test | Accuracy raza test | NDCG@10 independiente | Accuracy raza independiente |
|---|---|---|---|---|---|---|
| Baseline (ResNet18 ImageNet) | 512 | 0,873 | 0,860 | 0,932 | 0,474 | 0,565 |
| ResNet18 fine-tuned (Modelo A) | 512 | **0,950** | **0,949** | **0,963** | **0,537** | **0,598** |
| CNN propia (Modelo B) | 512 | 0,753 | 0,739 | 0,822 | 0,275 | 0,315 |

**Etapa 2: clasificación** (métricas macro; en el conjunto independiente se promedian solo sobre las 22 razas presentes).

| Métrica | Modelo A (ResNet18 fine-tuned) | Modelo B (CNN propia) |
|---|---|---|
| Accuracy (test) | **0,966** | 0,818 |
| Precision / Recall / F1 (test) | 0,967 / 0,966 / 0,965 | 0,835 / 0,817 / 0,816 |
| Specificity (test) | 1,000 | 0,997 |
| Accuracy / Recall (independiente) | **0,592** / 0,616 | 0,321 / 0,339 |
| Parámetros / Tamaño | 11,21 M / 44,9 MB | 2,52 M / 10,1 MB |
| Tiempo de inferencia en CPU | ~58 ms por imagen | ~82 ms por imagen |

Las matrices de confusión y las curvas de entrenamiento están en las secciones 6 y 7 de `2-train.ipynb`. La specificity queda cerca de 1 en todos los casos por la gran cantidad de verdaderos negativos de cada raza con 70 clases, así que no sirve para comparar modelos.

**Comparación entre enfoques.**

- *Fine-tuning contra CNN propia:* el Modelo A supera al B en todas las métricas de rendimiento (0,966 contra 0,818 en test) con el mismo costo de inferencia que el baseline. La CNN aprende todo con unas 110 imágenes por raza, mientras que la ResNet parte de lo aprendido con 1,2 millones de imágenes de ImageNet. El Modelo B solo gana en memoria (10 MB) y es más lento en CPU, porque aplica sus primeras convoluciones sobre la imagen completa (unos 2.800 millones de multiplicaciones por imagen, contra unos 1.800 millones de la ResNet18).
- *Búsqueda por similitud contra clasificación supervisada:* con los mismos embeddings del Modelo A dan prácticamente lo mismo (0,963 contra 0,966 en test): lo que define el rendimiento es la representación. La búsqueda no necesita reentrenar para agregar razas, muestra ejemplos parecidos y permite responder `unknown`, a cambio de guardar la galería y mantener una base de vectores; la clasificación tiene costo fijo y no necesita base de datos.
- *Cambio de dominio:* en el conjunto independiente el rendimiento cae mucho (Modelo A: de 0,941 en test con las mismas 22 razas a 0,592; Modelo B: de 0,790 a 0,321). Las fotos de Kaggle tienen el perro centrado ocupando todo el cuadro, mientras que las de Wikimedia tienen más fondo, otras poses y proporciones, y el resize directo las deforma.

**Elección para la app.** Clasificación: Modelo A. Búsqueda: embeddings del Modelo A (NDCG@10 de 0,873 a 0,950 sin costo adicional), aceptando un rechazo algo menor de imágenes sin perros (81 % contra 95 %), porque en la Etapa 3 YOLO ya filtra las imágenes sin perros.

**Análisis de errores (falsos positivos y falsos negativos).**

- Modelo A: los errores se concentran en pares de razas parecidas: Boston Terrier ↔ Bulldog (F1 de 0,67 y 0,73), American Spaniel ↔ Irish Spaniel, Malinois ↔ German Sheperd y Cockapoo ↔ Labradoodle. Los falsos negativos de Boston Terrier son claramente Boston Terriers clasificados como Bulldog, y entre los falsos positivos hay una imagen etiquetada como Bulldog con rasgos de Boston Terrier o Bulldog Francés: hipótesis de ruido en las etiquetas de esa clase.
- Modelo B: falla sobre todo en razas chicas de pelo largo (Cocker, Lhasa, Yorkie, Cockapoo) y de pelaje color arena (Labrador, Dingo), lo que indica que aprendió sobre todo color y textura. Se confirma en el t-SNE de los embeddings (sección 9).
- Los errores tienen scores más bajos que los aciertos (mediana de 0,88 en los aciertos del Modelo A), por lo que un score bajo es una señal útil de predicción dudosa.

**Etapa 3: detección.** Con `YOLO_CONF_THRESHOLD = 0.25`, el detector encuentra los perros aunque sean chicos y estén separados (por ejemplo, seis perros en una escena en la nieve). Los falsos positivos en imágenes sin perros son animales parecidos (un gato sin pelo y dos zorros; COCO no tiene la clase zorro) y los falsos negativos, perros muy chicos o lejanos, una foto antigua en blanco y negro con muchos perros superpuestos y perros en pinturas. Los recortes chicos se clasifican con scores bajos (0,09 a 0,20). Las capturas están en la sección 14 de `2-train.ipynb`.

## Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| Duplicados entre splits (130 pares) que producían fuga de datos | dHash de 64 bits verificado con correlación, agrupación en clusters y prioridad test > valid > train al conservar una copia. |
| La distancia de Hamming sola no distinguía duplicados de imágenes distintas entre 6 y 10 | Verificación de cada par con la correlación de las imágenes reducidas a 64×64 en grises (≥ 0,90). |
| El filtro de borrosas por percentil descartaba más del 20 % de razas de pelo claro | Umbral fijo y conservador (varianza del Laplaciano < 50) aplicado solo a train, midiendo el efecto por raza. |
| El criterio de escala de grises por media marcaba perros blancos y negros fotografiados a color | Percentil 99 de la diferencia entre canales. |
| La etiqueta `American  Spaniel` con doble espacio aparecía como una raza 71 | Normalización de los nombres de las razas. |
| Falta de memoria al pasar todo el dataset a YOLO en un solo llamado | Inferencia en batches de 32 imágenes. |
| El kernel de Jupyter seguía usando versiones viejas de los servicios después de editarlos | Reinicio del kernel o `importlib.reload` de los módulos. |
| En Colab, el zip del conjunto independiente creado en Windows tenía rutas con `\` y las imágenes no se encontraban | Extracción propia que reemplaza `\` por `/`. |
| Colab se desconecta y borra los archivos locales | Checkpoints, historiales y embeddings guardados en Google Drive y recuperados sin reentrenar; caché de embeddings en `output/`. |
| Métricas macro hundidas en el conjunto independiente (solo 22 de 70 razas) | Recalcular las métricas macro solo sobre las razas presentes y comparar contra test filtrado a las mismas razas. |
| El umbral de similitud calibrado con el baseline no valía para otros embeddings | Recalibración con la misma regla (percentil 2 en valid) para los embeddings del Modelo A. |
| Mediciones de tiempo en CPU distorsionadas por el calentamiento (el primer modelo medido tardaba el triple) | Calentamiento más largo y mediciones intercaladas en varias rondas, reportando la mediana. |
| La imagen Docker del backend ocupa varios GB porque `torch` instala las librerías de CUDA, y el build falló por falta de espacio en disco | Liberar espacio en disco. No se modificaron el `Dockerfile` ni `requirements.txt`; una alternativa sería instalar PyTorch solo para CPU, ya que el contenedor corre sin GPU. |

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** a definir por la cátedra. Solo se consideran los commits hasta esa fecha y hora; se recomienda trabajar de forma incremental.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
