# WORKFLOW – PoC túnel de flores (entorno local, scripts secuenciales)

## 0. Visión general

Este directorio es para una **PoC local**, no para un proyecto “bonito” de GitHub.
La prioridad es:

- Tener **datos COCO** en una carpeta clara.
- Tener **scripts de Python numerados** que pueda ejecutar uno detrás de otro.
- Tener **carpetas de salida** donde se vea rápido qué ha hecho cada fase.

Objetivo funcional de la PoC:

- El túnel captura imágenes 4K de carros de flores/plantas.
- Un modelo **Mask R‑CNN (Detectron2)** detecta y segmenta:
  - baldas (Balda1/2/3, etc.),
  - productos (Flores, Planta, tallo_grupo…).
- Un modelo **EfficientNet‑B4** clasifica la **especie** de cada producto a partir de crops 380×380.
- Un script de lógica Python:
  - agrupa por balda,
  - asigna a tickets JSON,
  - estima unidades (área + tallos),
  - compara contra el pedido esperado y genera alertas.

Todo esto se ejecutará **en local**, en Linux, con una GPU tipo RTX 3060.

---

## 0.1 Carpeta legacy de experimentos con YOLO

En el mismo nivel que este proyecto existe ya una carpeta llamada
`project_legacy_yolo/` (nombre orientativo) que contiene:

- La estructura antigua cuando se probó resolver el problema con YOLO.
- Versiones anteriores del **dataset** (imágenes + labels) en los formatos
  que usaba YOLO.
- Scripts, notebooks y utilidades que se emplearon en esos experimentos.

Puntos importantes:

- Este nuevo proyecto **NO** va a usar directamente la estructura de código
  ni la forma de entrenamiento de `project_legacy_yolo/`.
- **SÍ** podemos reutilizar cosas de esa carpeta:
  - las **imágenes originales** del túnel,
  - los **datasets ya anotados**, que se pueden convertir o exportar a COCO,
  - cualquier documentación/notebook que ayude a entender casos límite,
    errores típicos, etc.
- A efectos prácticos, cuando falten datos en `data/coco/`, la fuente oficial
  será:
  1. La herramienta de anotación actual (Roboflow/CVAT) exportando a COCO.
  2. La carpeta `project_legacy_yolo/`, de donde se pueden recuperar imágenes
     y, si hace falta, re-anotar para Mask R‑CNN.

Este WORKFLOW describe solo la **nueva línea de trabajo** basada en Detectron2
+ EfficientNet‑B4, pero se asume que la carpeta legacy está ahí y se puede
tirar de ella cuando haga falta más dataset o ejemplos históricos.

---

## 1. Estructura de directorios (simple y práctica)

```text
proyecto_tunel_flores/
├── data/
│   └── coco/
│       ├── images/           # Imágenes del túnel (frontal/trasera, etc.)
│       └── annotations/      # Anotaciones COCO (_annotations.coco_fixed.json)
│
├── outputs/
│   ├── maskrcnn_eval/        # Visualizaciones de Mask R-CNN
│   ├── crops_especies/       # Crops 380x380 por instancia
│   └── verificacion/         # Resultados de verificación por carro/ticket
│
├── models/
│   ├── maskrcnn/             # Pesos .pth de Detectron2
│   └── efficientnet/         # Pesos EfficientNet-B4
│
├── scripts/
│   ├── 01_fix_coco.py
│   ├── 02_train_maskrcnn.py
│   ├── 03_eval_maskrcnn.py
│   ├── 04_gen_crops.py
│   ├── 05_clasificar_especie.py
│   └── 06_verificar_pedido.py
│
├── WORKFLOW.md               # Este documento
└── requirements.txt          # (opcional) lista rápida de dependencias
```

Notas:

No hace falta empaquetar esto como librería ni seguir estándares estrictos.

La numeración de scripts (01, 02, …) define el orden natural de trabajo.

requirements.txt es solo para recordar qué instalar en el entorno local
(PyTorch, Detectron2, EfficientNet, OpenCV, etc.).

## 2. Qué quiero hacer en cada fase (scripts secuenciales)
### 2.1. 01_fix_coco.py – limpieza del dataset
Objetivo:
Partir de _annotations.coco.json exportado de la herramienta de anotación
(Roboflow/CVAT).
Arreglar problemas típicos:
- valores numéricos como cadenas ("327.7" → 327.7),
- segmentations con menos de 6 coordenadas,
- anotaciones sin máscara válida.

Entrada:
`data/coco/annotations/_annotations.coco.json`

Salida:
`data/coco/annotations/_annotations.coco_fixed.json` (dataset limpio listo
para Detectron2).

Uso previsto:
Ejecutar cuando haya cambios grandes en anotaciones o nuevos lotes de imágenes.
Si se reaprovechan imágenes de project_legacy_yolo/, se re-exportan a COCO
y pasan por este script antes de entrenar.

### 2.2. 02_train_maskrcnn.py – entrenamiento Mask R‑CNN (Detectron2)
Objetivo:
Usar Detectron2 para hacer fine‑tuning de Mask R‑CNN sobre mi COCO local,
con las clases: Balda1, Balda2, Balda3, Flores, Planta, tallo_grupo, etc.
Conseguir un modelo capaz de:
- detectar cada ramo/planta por separado,
- segmentar correctamente con máscara.

Entrada:
Imágenes: `data/coco/images/`
Anotaciones: `data/coco/annotations/_annotations.coco_fixed.json`.

Salida:
Pesos entrenados en `models/maskrcnn/model_final.pth`
(y opcionalmente checkpoints intermedios).

Uso previsto:
Se ejecuta cuando quiera actualizar el modelo con más datos o cambios de clases.

### 2.3. 03_eval_maskrcnn.py – evaluación visual del detector
Objetivo:
Cargar `models/maskrcnn/model_final.pth`.
Pasar un conjunto de imágenes de `data/coco/images/`.
Dibujar:
- máscaras,
- cajas,
- opcionalmente etiquetas de clase/score.

Entrada:
Modelo entrenado (`models/maskrcnn/model_final.pth`).
Imágenes de `data/coco/images/`.

Salida:
Imágenes anotadas en `outputs/maskrcnn_eval/` para inspección visual.

Uso previsto:
Comprobar cómo de bien separa ramos/baldas,
decidir si hace falta más anotación o más iteraciones de entrenamiento.

### 2.4. 04_gen_crops.py – generación de crops para EfficientNet‑B4
Objetivo:
Usar el modelo Mask R‑CNN entrenado para generar crops 380×380 por
instancia de producto (Flores, Planta, tallo_grupo).
Cada crop debe contener principalmente el ramo/planta, idealmente usando
la máscara para recortar.

Entrada:
Modelo Mask R‑CNN (`models/maskrcnn/model_final.pth`).
Imágenes de `data/coco/images/`.

Salida:
Carpeta `outputs/crops_especies/` con imágenes 380×380.
Opcionalmente un CSV/JSON que asocie cada crop con:
- id de instancia,
- especie (si ya la conozco),
- imagen original.

Uso previsto:
Crear dataset de entrenamiento para EfficientNet‑B4.
Más adelante, también servir como paso de inferencia para el pipeline final.

### 2.5. 05_clasificar_especie.py – clasificación con EfficientNet‑B4
Objetivo:
Cargar EfficientNet‑B4 (preentrenada y/o fine‑tuneada) desde `models/efficientnet/`.
Clasificar los crops generados en `outputs/crops_especies/` en especies:
rosa, gerbera, crisantemo, etc. (10+ clases).

Entrada:
Pesos EfficientNet‑B4 en `models/efficientnet/`.
Imágenes 380×380 en `outputs/crops_especies/`.

Salida:
Archivo (CSV/JSON) con:
- id de crop / id de instancia,
- species_id,
- score de clasificación.

Uso previsto:
Servir de base para la lógica de verificación; saber qué especie ve la IA
para cada instancia detectada.

### 2.6. 06_verificar_pedido.py – lógica de negocio de verificación
Objetivo:
Integrar salidas de Mask R‑CNN y EfficientNet‑B4 con los tickets JSON del carro.
Hacer todo el trabajo “de negocio”:
- agrupar instancias por balda (usando bbox/mask y rangos de Y),
- asignar productos a tickets según el JSON de la colección,
- contar unidades combinando área/tallos,
- aplicar tolerancias (±k unidades, umbrales de confianza),
- generar una estructura de alertas.

Entrada:
Detecciones de Mask R‑CNN (se pueden leer de disco o recalcular).
Clasificación de especies (salida de `05_clasificar_especie.py`).
JSON de tickets del carro (colección de 10 tickets, 2 carros × 5 baldas).

Salida:
Un JSON/CSV en `outputs/verificacion/` con:
- por ticket/balda/especie: esperado vs detectado,
- lista de alertas (mezcla, cantidad incorrecta, especie errónea, etc.).
Opcionalmente una imagen anotada (reutilizando lo de `03_eval_maskrcnn.py`)
marcando errores.

Uso previsto:
Ser la pieza final del pipeline de túnel.
Lo que se enseñará en la demo PoC al operador/supervisor.

## 3. Forma de trabajo recomendada
En este proyecto no hace falta estructurar paquetes, módulos ni tooling de GitHub.
La idea es trabajar así:

1. Preparar datos COCO y limpiarlos:
   Poner imágenes y JSON en `data/coco/` (copiando o reaprovechando material
   de project_legacy_yolo/ cuando convenga).
   Ejecutar `scripts/01_fix_coco.py`.

2. Entrenar o reentrenar Mask R‑CNN:
   Ejecutar `scripts/02_train_maskrcnn.py`.

3. Comprobar visualmente el detector:
   Ejecutar `scripts/03_eval_maskrcnn.py` y mirar `outputs/maskrcnn_eval/`.

4. Generar crops para clasificación:
   Ejecutar `scripts/04_gen_crops.py` y revisar `outputs/crops_especies/`.

5. Entrenar/usar EfficientNet‑B4 y clasificar:
   Ejecutar `scripts/05_clasificar_especie.py`.

6. Verificar pedidos de un carro:
   Ejecutar `scripts/06_verificar_pedido.py` y revisar `outputs/verificacion/`.

La prioridad es que, en el futuro, con solo abrir este directorio y leer
WORKFLOW.md, quede claro:
- qué hace cada script,
- en qué orden se ejecutan,
- dónde están los datos de entrada,
- dónde se escriben los resultados,
- y que existe una carpeta legacy con experimentos YOLO de la que se pueden
  reaprovechar datos si hace falta.
