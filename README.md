<![CDATA[<div align="center">

# 🌸 Proyecto H — Túnel de Flores

**Sistema de visión artificial para detección, clasificación y conteo automático de plantas y flores en carros logísticos**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Detectron2](https://img.shields.io/badge/Detectron2-Meta-4267B2?logo=meta&logoColor=white)](https://github.com/facebookresearch/detectron2)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## 📋 Índice

- [Descripción del proyecto](#-descripción-del-proyecto)
- [Arquitectura del sistema](#-arquitectura-del-sistema)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Requisitos e instalación](#-requisitos-e-instalación)
- [Pipeline de datos](#-pipeline-de-datos)
- [Scripts — Referencia completa](#-scripts--referencia-completa)
- [Configuración de las redes](#-configuración-de-las-redes)
- [Aplicación web (Streamlit)](#-aplicación-web-streamlit)
- [Workflow de uso](#-workflow-de-uso)

---

## 🎯 Descripción del proyecto

El sistema fotografía **carros de plantas** desde dos ángulos:
- **Frontal (F)**: se ven las baldas, los tickets de pedido, las flores y las plantas.
- **Trasera (B)**: se ven los tallos que sobresalen por detrás.

A partir de ambas imágenes, el sistema:

1. **Detecta y segmenta** (Mask R-CNN) las 5 clases: `Flores`, `Planta`, `Balda`, `ticket`, `tallo_grupo`.
2. **Asigna tickets a baldas** mediante cruce espacial de bounding boxes.
3. **Cuenta items** cruzando detecciones frontales y traseras (tallos ↔ masas).
4. **Clasifica la especie** (ConvNeXt) de cada flor/planta detectada, identificando el producto.
5. **Genera un JSON** con el inventario estructurado por ticket, balda y producto.

### JSON de salida (ejemplo)

```json
{
    "Items": {
        "Ticket_1": {
            "Balda_2": {
                "143992": {
                    "tipo": "flores",
                    "detecciones": 2,
                    "tallos_totales": 2,
                    "unidades_totales": 2,
                    "confianza_media": 0.927
                },
                "99338": {
                    "tipo": "flores",
                    "detecciones": 1,
                    "tallos_totales": 1,
                    "unidades_totales": 1,
                    "confianza_media": 0.918
                }
            }
        }
    }
}
```

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────┐    ┌─────────────────────┐
│   Imagen Frontal    │    │   Imagen Trasera    │
│   (99F.png)         │    │   (99B.png)         │
└────────┬────────────┘    └────────┬────────────┘
         │                          │
         ▼                          ▼
┌────────────────────────────────────────────────┐
│              MASK R-CNN (Detectron2)           │
│  Segmentación de instancias — 5 clases:        │
│  Flores · Planta · Balda · ticket · tallo_grupo│
└────────┬───────────────────────────┬───────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐       ┌──────────────────────┐
│ Cruce Espacial  │       │  Mapeo en Espejo     │
│ Ticket ↔ Balda  │       │  Trasera → Frontal   │
│ (por posición Y)│       │  (Flip X + match Y)  │
└────────┬────────┘       └──────────┬───────────┘
         │                           │
         ▼                           ▼
┌────────────────────────────────────────────────┐
│         CONVNEXT TINY (timm)                   │
│  Clasificación por especie — N clases          │
│  (crop de cada masa → producto_id)             │
└────────────────────────┬───────────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │   JSON final    │
              │   (inventario)  │
              └─────────────────┘
```

---

## 📁 Estructura del repositorio

```
py_PROYECTO_H/
│
├── configs/                        # ⚙️ Configuraciones YAML
│   ├── config1.yaml                # Config Mask R-CNN (anchors, solver, augmentation)
│   ├── config_convnext.yaml        # Config ConvNeXt (modelo, data, training)
│   └── config_manager.py           # Parser YAML → Detectron2 cfg
│
├── scripts/                        # 🔧 Pipeline principal (ejecutar en orden numérico)
│   ├── 00_cropping.py              # Extrae crops de flores/plantas para clasificación
│   ├── 00_trust_fix_coco.py        # Visualiza ground truth para verificar anotaciones
│   ├── 01_fix_coco.py              # Unifica datasets Roboflow → formato COCO estándar
│   ├── 02_train_maskrcnn.py        # Entrena Mask R-CNN (segmentación de instancias)
│   ├── 03_eval_maskrcnn.py         # Evalúa Mask R-CNN (métricas COCO + visualizaciones)
│   ├── 04_seg_tickets.py           # Cruce espacial tickets ↔ baldas (versión visual)
│   ├── 05_conteo.py                # Pipeline completo: detección + clasificación + conteo
│   ├── 06_conteo_masivo.py         # Conteo en lote sobre múltiples pares de imágenes
│   ├── 07_train_convnext.py        # Entrena ConvNeXt Tiny (clasificación por especie)
│   ├── 08_eval_convnext.py         # Evalúa ConvNeXt (confusion matrix, errores)
│   ├── conteo_module.py            # Wrapper para importar funciones de conteo en la app
│   └── upload_to_roboflow.py       # Sube crops clasificados a Roboflow
│
├── test_area/                      # 🖥️ Aplicación web
│   ├── app.py                      # Streamlit app con 3 módulos (MRCNN / ConvNeXt / Conteo)
│   └── view_gt.py                  # Visualizador de ground truth
│
├── data/                           # 📊 Datasets (excluido de Git, ver .gitignore)
│   ├── dataset_final/              # Pares F/B para producción
│   ├── coco_unified/               # Dataset COCO unificado para Mask R-CNN
│   └── crops_clasificacion/        # Crops para entrenar ConvNeXt
│
├── models/                         # 🧠 Modelos entrenados (excluido de Git, 19 GB)
│   ├── maskrcnn/                   # Versiones de Mask R-CNN
│   └── convnext/                   # Versiones de ConvNeXt
│
├── .gitignore                      # Archivos excluidos del versionado
├── WORKFLOW.md                     # Referencia rápida del flujo de trabajo
└── README.md                       # Este archivo
```

---

## 🛠️ Requisitos e instalación

### Requisitos de hardware

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU | NVIDIA con CUDA | NVIDIA RTX 3060+ |
| VRAM | 6 GB | 8+ GB |
| RAM | 16 GB | 32 GB |
| Disco | 30 GB | 50 GB |

### Dependencias principales

```
Python >= 3.10
PyTorch >= 2.0 (con CUDA)
Detectron2 (compilado contra tu versión de PyTorch)
timm >= 0.9 (para ConvNeXt)
torchvision
OpenCV (opencv-python, NO headless si necesitas GUI)
Streamlit
scikit-learn
PyYAML
Pillow
matplotlib
numpy
```

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/alexdme1/py_PROYECTO_H.git
cd py_PROYECTO_H

# 2. Activar entorno virtual
source /home/servi2/Enviroments/main_venv/bin/activate

# 3. Instalar dependencias (si no están ya)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install timm streamlit scikit-learn pyyaml opencv-python matplotlib
```

---

## 🔄 Pipeline de datos

### Flujo de preparación

```
Roboflow Export (COCO format)
        │
        ▼
┌─ 01_fix_coco.py ──────────────────────────────┐
│  • Lee splits train/valid/test de Roboflow     │
│  • Rota imágenes 90° CCW (formato vertical)    │
│  • Rota bounding boxes y segmentaciones        │
│  • Unifica categorías dispersas:               │
│      Balda1, Balda2, Balda3 → "Balda" (id=2)   │
│      0 → "ticket" (id=1)                       │
│  • Genera train.json, valid.json, test.json    │
└────────────────────────────────────────────────┘
        │
        ▼
   data/coco_unified/
   ├── images/          (todas las imágenes rotadas)
   └── annotations/     (train.json, valid.json, test.json)
```

### Clases del modelo de segmentación

| ID | Clase | Descripción |
|----|-------|-------------|
| 0 | `Flores` | Masas de flores visibles desde el frente |
| 1 | `ticket` | Etiquetas de pedido pegadas en las baldas |
| 2 | `Balda` | Estantes/niveles del carro |
| 3 | `Planta` | Plantas en maceta visibles desde el frente |
| 4 | `tallo_grupo` | Grupos de tallos visibles por detrás |

---

## 📜 Scripts — Referencia completa

### `00_cropping.py` — Extracción de crops

**Propósito**: Recorta las bounding boxes de `Flores` y `Planta` del dataset de segmentación para crear un dataset de clasificación por especie.

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `ROBOFLOW_DIR` | `data/Proyecto_H.v4i.coco(no_aug)/` | Carpeta del export Roboflow (sin augmentation) |
| `TARGET_CATEGORY_NAMES` | `{"Flores", "Planta"}` | Qué clases recortar |
| `MIN_CROP_SIZE` | `10` | Tamaño mínimo del crop (px). Descarta ruido |

**Funciones:**

| Función | Descripción |
|---------|-------------|
| `process_split(roboflow_dir, split, ...)` | Procesa un split (train/valid/test). Lee el JSON COCO, extrae las bboxes de las categorías objetivo, recorta la imagen, rota 90° CCW, y guarda el crop en `OUTPUT_DIR/{clase}/` |
| `main()` | Orquesta la extracción para todos los splits y muestra estadísticas |

**Salida**: `data/crops_clasificacion/{Flores,Planta}/*.png`

---

### `00_trust_fix_coco.py` — Verificador visual de anotaciones

**Propósito**: Dibuja las anotaciones ground truth (bboxes + masks translúcidas) sobre N imágenes aleatorias para verificar visualmente que el dataset está bien anotado.

**Funciones:**

| Función | Descripción |
|---------|-------------|
| `get_color_for_id(cat_id)` | Devuelve un color BGR único por categoría |
| `draw_ground_truth(json_path, images_dir, output_dir, num_samples)` | Carga el JSON COCO, selecciona N imágenes aleatorias, dibuja bboxes + segmentaciones translúcidas con etiquetas de clase, y guarda las visualizaciones |

---

### `01_fix_coco.py` — Unificador de datasets

**Propósito**: Convierte los exports de Roboflow (con nombres de categoría inconsistentes como `Balda1`, `Balda2`, `0`) en un dataset COCO limpio y unificado.

**Transformaciones clave:**
- Rota **todas las imágenes** 90° CCW (de paisaje a retrato)
- Rota las **bounding boxes** y **segmentaciones poligonales** acorde
- Unifica categorías: `Balda1/2/3 → Balda`, `0 → ticket`
- Genera **un JSON por split** (`train.json`, `valid.json`, `test.json`)

**Funciones:**

| Función | Descripción |
|---------|-------------|
| `to_float(val)` | Conversión segura a float (evita crashes por strings en anotaciones) |
| `process_bbox(bbox)` | Valida y parsea una bbox `[x, y, w, h]` |
| `process_segmentation(seg)` | Valida polígonos de segmentación (mín. 3 puntos = 6 coords) |
| `fix_and_merge_dataset(input_dir, output_dir)` | Pipeline completo: lee Roboflow → rota → remapea categorías → genera COCO unificado |

**Mapeo de categorías:**

```python
CATEGORY_MAP = {
    "Flores": 0,      # Masas de flores
    "0": 1,            # Tickets (en Roboflow se llaman "0")
    "Balda1": 2,       # ─┐
    "Balda2": 2,       # ─┤ Todas se unifican como "Balda"
    "Balda3": 2,       # ─┘
    "Planta": 3,       # Plantas en maceta
    "tallo_grupo": 4   # Grupos de tallos
}
```

---

### `02_train_maskrcnn.py` — Entrenamiento Mask R-CNN

**Propósito**: Entrena un modelo Mask R-CNN con backbone ResNet-50 FPN para segmentación de instancias.

**Arquitectura**: `mask_rcnn_R_50_FPN_3x` (pretrained COCO → fine-tuned en nuestro dataset)

**Clase `CustomEvaluatorTrainer`:**

| Método | Descripción |
|--------|-------------|
| `build_evaluator(cfg, dataset_name)` | Inyecta `COCOEvaluator` para evaluación periódica durante entrenamiento |
| `build_train_loader(cfg)` | Crea el dataloader con augmentaciones en runtime (brillo, contraste, flips). Lee los parámetros de `AUGMENTATIONS_EXTRA` del YAML |

**Función `main()`:**
1. Lee `config1.yaml` y aplica los parámetros al cfg de Detectron2
2. Registra el dataset COCO unificado (`flores_train`, `flores_valid`, `flores_test`)
3. Configura solver, anchors, RPN, ROI heads según el YAML
4. Entrena con checkpoints periódicos
5. Ejecuta evaluación final en test
6. Guarda el modelo en `models/maskrcnn/{run_name}/`

---

### `03_eval_maskrcnn.py` — Evaluación Mask R-CNN

**Propósito**: Evalúa un modelo entrenado con métricas COCO estándar (AP, AP50, AP75) y genera visualizaciones filtradas.

**Función `main(model_path, min_area)`:**

| Paso | Descripción |
|------|-------------|
| 1 | Carga el modelo con config del YAML (anchors, etc.) |
| 2 | Ejecuta `COCOEvaluator` sobre el split de test |
| 3 | Genera visualizaciones filtrando por área mínima de máscara |

| Parámetro | Descripción |
|-----------|-------------|
| `model_path` | Ruta al `.pth` del modelo |
| `min_area` | Área mínima en píxeles para dibujar detecciones (default: 5000). Elimina ruido visual |

---

### `04_seg_tickets.py` — Cruce espacial (visual)

**Propósito**: Versión visual del cruce espacial ticket ↔ balda con dibujo de líneas y zonas. Útil para debug y presentación.

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `asignar_tickets_a_baldas(detecciones, total_baldas)` | Asigna cada ticket a las baldas que domina (por posición Y). Cada ticket se "expande" hacia las baldas adyacentes que no tienen ticket propio |
| `procesar_pareja_imagenes(det_f, det_b)` | Traslada la asignación del espacio frontal al trasero. Las baldas traseras heredan las zonas Y de las frontales |
| `contar_articulos(det_f, det_b, asignacion)` | Cuenta Flores/Plantas por balda cruzando ambas vistas |
| `extract_detections(outputs)` | Convierte `outputs["instances"]` de Detectron2 a lista de dicts `{class, bbox}` |

---

### `05_conteo.py` — Pipeline completo de conteo

**Propósito**: Script principal que ejecuta la detección + clasificación + conteo completo. Genera el JSON final de inventario.

**Funciones del módulo (importables por la app):**

| Función | Descripción |
|---------|-------------|
| `asignar_tickets_a_baldas(det, n)` | Asignación espacial ticket → baldas |
| `procesar_pareja_imagenes(det_f, det_b)` | Traslado frontal → trasera |
| `contar_articulos(det_f, det_b, asig, img_f, clasificador)` | Conteo con clasificación por especie |

**Lógica de `contar_articulos`:**

```
1. Separar baldas frontales y traseras, ordenar por Y
2. Ubicar cada Flor/Planta en su balda (por centro Y)
3. Ubicar cada tallo_grupo en su balda trasera
4. MAPEO EN ESPEJO: para cada balda trasera → inventir X (flip)
   → encontrar la masa frontal más cercana en Y → sumar tallos
5. Para cada masa frontal con ConvNeXt disponible:
   a. Crop de la bbox
   b. Rotar 90° CCW
   c. Clasificar → producto_id + confianza
6. Agrupar items por producto_id en cada balda:
   → sumar detecciones, tallos, unidades
   → promediar confianza
```

**Variables de configuración (`__main__`):**

| Variable | Descripción |
|----------|-------------|
| `MRCNN_MODEL_PATH` | Ruta al `.pth` de Mask R-CNN a usar |
| `SCORE_THRESH` | Umbral mínimo de confianza para detecciones (default: 0.10) |
| `CONVNEXT_RUN_DIR` | Carpeta del run de ConvNeXt |
| `CONVNEXT_MODEL_PATH` | Ruta al `.pth` de ConvNeXt (best_model o model_final) |
| `IMAGE_FRONTAL_PATH` | Imagen frontal de prueba |
| `IMAGE_TRASERA_PATH` | Imagen trasera de prueba |

---

### `06_conteo_masivo.py` — Conteo en lote

**Propósito**: Ejecuta el pipeline de conteo sobre **todos los pares** de imágenes en `data/dataset_final/`. Genera un JSON global y visualizaciones para cada par.

**Mismo flujo que `05_conteo.py` pero iterando sobre todos los pares `{N}F.png` / `{N}B.png`.**

---

### `07_train_convnext.py` — Entrenamiento ConvNeXt

**Propósito**: Entrena un modelo ConvNeXt Tiny para clasificación de especies vegetales.

**Arquitectura**: `convnext_tiny.fb_in22k` (pretrained ImageNet-22K → fine-tuned)

**Funciones:**

| Función | Descripción |
|---------|-------------|
| `load_config(path)` | Carga `config_convnext.yaml` |
| `build_transforms(cfg)` | Construye augmentaciones para train (flip, rotación, color jitter, erasing) y preprocesamiento para val/test (resize + normalize) |
| `FilteredImageFolder` | Subclase de `ImageFolder` que excluye carpetas de clases no deseadas (ej. `borrar`) |
| `build_dataloaders(cfg, ...)` | Crea DataLoaders con train/val/test split |
| `build_model(cfg)` | Crea ConvNeXt Tiny con `timm`. Opción de congelar backbone (`FREEZE_BACKBONE`) |
| `build_scheduler(optimizer, cfg, steps)` | Cosine Annealing con warmup lineal |
| `train_one_epoch(...)` | Loop de entrenamiento con logging a TensorBoard |
| `evaluate(...)` | Evaluación sobre un split con loss y accuracy |
| `main()` | Orquesta todo: carga config → builds → train loop → test final → guarda modelo |

**Salidas** (en `models/convnext/{run_name}/`):

| Archivo | Descripción |
|---------|-------------|
| `best_model.pth` | Modelo con mejor validación accuracy |
| `model_final.pth` | Modelo al final del entrenamiento |
| `class_names.txt` | Lista de clases en el orden del modelo |
| `config_used.yaml` | Copia de la config usada para reproducibilidad |
| `events.out.tfevents.*` | Logs de TensorBoard |

---

### `08_eval_convnext.py` — Evaluación ConvNeXt

**Propósito**: Evaluación detallada del clasificador con métricas por clase.

**Genera:**
- Accuracy global y por clase
- Classification report completo (precision, recall, F1)
- Matriz de confusión (guardada como PNG)
- Imágenes de errores de clasificación (para análisis visual)

---

### `conteo_module.py` — Wrapper de importación

**Propósito**: Permite importar las funciones de `05_conteo.py` desde la app Streamlit sin ejecutar el bloque `__main__`.

Lee el source de `05_conteo.py`, extrae solo las definiciones de funciones (sin el main), y las re-exporta como módulo independiente.

---

### `upload_to_roboflow.py` — Subida a Roboflow

**Propósito**: Sube los crops clasificados a un proyecto de Roboflow para crear datasets de clasificación anotados.

> ⚠️ **API Key**: Se lee de la variable de entorno `ROBOFLOW_API_KEY`. Configura con:
> ```bash
> export ROBOFLOW_API_KEY="tu_api_key_aqui"
> ```

---

## ⚙️ Configuración de las redes

### `config1.yaml` — Mask R-CNN

```yaml
MODEL_INFO:
  SUFIJO_VERSION: "_anchors_v4_hope"    # Sufijo para el nombre del run
  OUTPUT_DIR_BASE: "models/maskrcnn"     # Carpeta base de salida
  BASE_WEIGHTS_YAML: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
```

#### Anchors (cómo escanea la red)

```yaml
MODEL:
  ANCHOR_GENERATOR:
    SIZES: [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512]]
    ASPECT_RATIOS: [[0.25, 0.5, 1.0, 2.5, 4.0]]
```

| Parámetro | Qué controla |
|-----------|-------------|
| `SIZES` | Tamaños de las anclas en cada nivel del FPN (5 niveles). Anclas pequeñas detectan tallos; grandes detectan baldas |
| `ASPECT_RATIOS` | Proporciones altura/ancho. `0.25` = muy horizontal (baldas), `4.0` = muy vertical (tallos) |

#### RPN (Region Proposal Network)

```yaml
  RPN:
    NMS_THRESH: 0.65           # IoU para suprimir propuestas duplicadas
    POST_NMS_TOPK_TRAIN: 2000  # Máx propuestas tras NMS en training
    POST_NMS_TOPK_TEST: 2000   # Máx propuestas tras NMS en test
```

#### ROI Heads (clasificación final)

```yaml
  ROI_HEADS:
    NUM_CLASSES: 5                # Flores, ticket, Balda, Planta, tallo_grupo
    BATCH_SIZE_PER_IMAGE: 512     # Regiones evaluadas por imagen (calidad vs velocidad)
    NMS_THRESH_TEST: 0.65         # NMS en las detecciones finales
    SCORE_THRESH_TEST: 0.20       # Confianza mínima para aceptar una detección
```

#### Solver (entrenamiento)

```yaml
SOLVER:
  IMS_PER_BATCH: 2        # Batch size (limitado por VRAM)
  BASE_LR: 0.00025        # Learning rate base
  MAX_ITER: 28000          # Iteraciones totales (~12-14 epochs)
  STEPS: [18000, 24000]    # Puntos donde el LR baja ×0.1
  WARMUP_ITERS: 1000       # Iteraciones de warmup lineal
  WEIGHT_DECAY: 0.0005     # Regularización L2
```

#### Data Augmentation

```yaml
INPUT:
  MIN_SIZE_TRAIN: [1000, 1080, 1150]  # Multi-scale training
  MAX_SIZE_TRAIN: 2000
  RANDOM_FLIP: "horizontal"

  AUGMENTATIONS_EXTRA:
    ACTIVAS: True
    PROBABILIDAD: 0.50
    BRILLO_MIN: 0.7
    BRILLO_MAX: 1.1
```

---

### `config_convnext.yaml` — ConvNeXt

#### Modelo

```yaml
MODEL:
  NAME: "convnext_tiny.fb_in22k"  # Pretrained en ImageNet-22K
  NUM_CLASSES: null                 # Auto-detectado del dataset
  PRETRAINED: true
  FREEZE_BACKBONE: false            # false = fine-tune completo
  EXCLUDE_CLASSES: ["borrar"]       # Clases a ignorar del dataset
```

#### Dataset

```yaml
DATA:
  ROOT: "data/Proyecto_H_clas.v1i.folder"
  IMG_SIZE: 224
  BATCH_SIZE: 16
  NUM_WORKERS: 4
```

#### Training

```yaml
TRAINING:
  EPOCHS: 30
  OPTIMIZER: "AdamW"
  LR: 0.0001
  WEIGHT_DECAY: 0.01
  SCHEDULER: "cosine"
  WARMUP_EPOCHS: 3
  LABEL_SMOOTHING: 0.1
```

---

### `config_manager.py` — Parser de configuración

Convierte el YAML en la estructura `CfgNode` que espera Detectron2.

| Función | Descripción |
|---------|-------------|
| `parse_yaml_config(yaml_path)` | Lee el YAML y devuelve un dict Python |
| `apply_custom_config_to_cfg(cfg, config_data)` | Mapea cada sección del dict a los campos equivalentes de `cfg` (Detectron2). Maneja ANCHOR_GENERATOR, RPN, ROI_HEADS, INPUT, SOLVER, DATALOADER, TEST |

---

## 🖥️ Aplicación web (Streamlit)

### `test_area/app.py`

Aplicación con 3 módulos seleccionables:

#### 1. Mask R-CNN (Segmentación)
- Selecciona run + checkpoint
- Sube una imagen
- Visualiza detecciones con masks coloreadas

#### 2. ConvNeXt (Clasificación)
- Selecciona run + checkpoint
- Sube un crop de flor/planta
- Muestra especie predicha + top-5 con barras de confianza

#### 3. Conteo (Pipeline completo)
- Selecciona modelos de **ambas** redes (Mask R-CNN + ConvNeXt)
- Sube par Frontal + Trasera
- Ejecuta pipeline completo
- Muestra visualizaciones de detección + JSON de inventario

### Lanzar la app

```bash
cd py_PROYECTO_H
/home/servi2/Enviroments/main_venv/bin/python test_area/app.py
# Abre http://localhost:8501
```

---

## 🔁 Workflow de uso

### Entrenamiento completo (desde cero)

```bash
# 1. Preparar dataset (convertir Roboflow → COCO unificado)
python scripts/01_fix_coco.py

# 2. Verificar visualmente las anotaciones
python scripts/00_trust_fix_coco.py

# 3. Entrenar Mask R-CNN
python scripts/02_train_maskrcnn.py

# 4. Evaluar Mask R-CNN
python scripts/03_eval_maskrcnn.py

# 5. Extraer crops para clasificación
python scripts/00_cropping.py

# 6. (Opcional) Subir crops a Roboflow para anotar por especie
python scripts/upload_to_roboflow.py

# 7. Entrenar ConvNeXt
python scripts/07_train_convnext.py

# 8. Evaluar ConvNeXt
python scripts/08_eval_convnext.py

# 9. Ejecutar conteo sobre un par de imágenes
python scripts/05_conteo.py

# 10. Lanzar la app web
python test_area/app.py
```

### Conteo rápido (con modelos ya entrenados)

```bash
# Editar rutas en scripts/05_conteo.py (líneas 375-390)
python scripts/05_conteo.py
```

---

<div align="center">

**Proyecto H** · Verdnatura · 2026

</div>
]]>
