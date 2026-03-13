# Diseño Árbol de Decisión de Conteo por Detección

## 1. Objetivo del modelo

Árbol (o Random Forest) que, a partir de **detecciones individuales** (planta/flor) con su contexto espacial, predice **cuántas unidades reales aporta cada detección** dentro de un ticket.
El conteo final por ticket/sku se obtiene sumando las unidades predichas de sus detecciones válidas.

- Unidad de modelo: detección individual de tipo `planta` o `flor`.
- Output por detección: `unidades_pred_d ∈ {0,1,2,…}`.
- Agregación:
  - Por ticket×sku: `unidades_ticket_sku = Σ_d unidades_pred_d`.
  - Opcionalmente por ticket×balda×sku, si se quiere mantener granularidad por balda como en la verificación actual.

El árbol **no** decide tickets ni baldas: se asume que la lógica existente de asignación ticket→baldas funciona bien y se aplica antes del modelo.

---

## 2. Punto de entrada en el pipeline

Orden de ejecución:

1. Mask R‑CNN: detecciones de `Flores`, `Planta`, `tallogrupo` en frontal y trasera.
2. EfficientNet‑B4 / ConvNeXt: clasificación de especie (sku) para detecciones de `Flores` y `Planta`.
3. Lógica geométrica actual:
   - Asignación de tickets a baldas (REGLA‑A01–A04).
   - Asignación de cada detección a una balda y un ticket.
4. Construcción de features por detección (este documento).
5. Árbol de decisión: predicción `unidades_pred_d` por detección de tipo `planta` o `flor`.
6. Agregación de unidades por ticket/sku y comparación contra el pedido esperado (lógica de verificación actual).

En esta primera fase, **no se modifican** las reglas B04–B05 de reparto de tallos; el modelo aprende directamente del contexto de detecciones sin necesitar una asignación explícita tallo→masa.

---

## 3. Unidad de entrenamiento y target

**Unidad de entrenamiento**: una detección `d` de tipo `planta` o `flor`, ya con:

- `ticket_d` asignado.
- `balda_d` asignada.
- Orden relativo dentro de la balda (posición en X normalizada).

Detecciones de `tallogrupo`:

- Se usan como **contexto/vecinos** en las features, pero no tienen target en este modelo (no se entrena salida para tallos).

**Target por detección**:

- `unidades_label_d` = número de unidades reales de ese sku que representa esa detección.
  - Ejemplos:
    - Planta vista delante y detrás, pero es la misma unidad → una de las dos detecciones etiquetada con `unidades_label_d = 1`, la otra con `0`.
    - Ramo de 10 flores en un solo bulto frontal → detección frontal etiquetada con `unidades_label_d = 10`.

En inferencia:

- `unidades_pred_d = f(features_d)` para cada detección de tipo `planta` o `flor`.
- Conteo final por ticket×sku: suma de `unidades_pred_d` filtrando por `ticket` y `sku`.

---

## 4. Features por detección

Para cada detección `d` (planta/flor/tallo) se construirán features con cuatro niveles.

### 4.1. Atributos propios de la detección

| Campo | Tipo | Notas |
|---|---|---|
| `tipo_d` | one‑hot | flor / planta / tallo |
| `sku_d` | str o int | vacío/0 si no aplica |
| `ticket_d`, `balda_d` | int | solo trazabilidad |
| `lado_d` | one‑hot | F / B |
| `pos_rel_d` | float [0,1] | posición relativa en balda |
| `volumen_d` | float | área de máscara en px (no bbox) |
| `score_mrcnn_d` | float [0,1] | confianza Mask R‑CNN |
| `score_convnext_d` | float [0,1] | confianza top‑1 ConvNeXt; 0.0 si tallo |
| `es_superpuesto_d` | 0/1 | si tiene algún artículo superpuesto |

### 4.2. Vecinos en la misma vista

**Vecino derecha:**

| Campo | Tipo | Notas |
|---|---|---|
| `vec_der_existe_d` | 0/1 | |
| `vec_der_tipo` | one‑hot | flor / planta / tallo / none |
| `vec_der_sku` | str | |
| `vec_der_pos_rel` | float | -1 si no existe |
| `vec_der_volumen` | float | 0 si no existe |
| `vec_der_score_mrcnn` | float | 0 si no existe |
| `vec_der_score_convnext` | float | 0 si no existe |
| `vec_der_superpuesto` | 0/1 | |
| `vec_der_superpuesto2` | 0/1 | si ese vecino tiene otro superpuesto |

**Vecino izquierda:** mismos campos con prefijo `vec_izq_`.

### 4.3. Corresponsal en la vista contraria ("otro lado")

Se elige la detección del mismo tipo en la otra vista con `pos_rel` más cercana. Si no existe, `otro_lado_existe_d = 0` y valores neutros.

| Campo | Tipo | Notas |
|---|---|---|
| `otro_lado_existe_d` | 0/1 | |
| `otro_lado_tipo` | one‑hot | flor / planta / tallo / none |
| `otro_lado_sku` | str | |
| `otro_lado_pos_rel` | float | -1 si no existe |
| `otro_lado_volumen` | float | 0 si no existe |
| `otro_lado_score_mrcnn` | float | 0 si no existe |
| `otro_lado_score_convnext` | float | 0 si no existe |
| `otro_lado_dist_pos_rel` | float | \|pos_rel_d − otro_lado_pos_rel\| |
| `otro_lado_superpuesto` | 0/1 | |
| `otro_lado_superpuesto2` | 0/1 | |

Vecinos del otro lado: mismos campos que 4.2 con prefijos `otro_lado_vec_der_` y `otro_lado_vec_izq_`.

### 4.4. Agregados de contexto local

| Campo | Tipo | Notas |
|---|---|---|
| `num_mismo_tipo_misma_balda_misma_vista` | int | |
| `num_mismo_tipo_misma_balda_otro_lado` | int | |
| `num_tallos_misma_balda` | int | total tallogrupo en esa balda, ambas vistas |

---

## 5. Normalización y one‑hot

La normalización y codificación se hace **en el pipeline de entrenamiento** (sklearn `ColumnTransformer`), no al generar el CSV.

- **Numéricas** (dejar tal cual o escalado simple): `pos_rel_d`, `volumen_d`, `score_*`, `vec_*_pos_rel`, `vec_*_volumen`, `otro_lado_*`, `num_*`.
- **One‑hot**: `tipo_d`, `lado_d`, `vec_der_tipo`, `vec_izq_tipo`, `otro_lado_tipo`, `otro_lado_vec_*_tipo` → categorías `{flor, planta, tallo, none}`.
- **sku_d y *_sku**: fase 1, no incluir como feature; opcionalmente solo flag `mismo_sku_que_otro_lado`.
- **Binarias** (0/1): dejar tal cual.

---

## 6. Uso del modelo en el conteo final

En inferencia:

1. Se generan todas las detecciones con las features anteriores.
2. El árbol predice `unidades_pred_d` solo para detecciones `tipo_d ∈ {flor, planta}`.
3. Para cada combinación `(ticket, sku)`:
   - `unidades_ticket_sku = Σ_d unidades_pred_d`
4. Este conteo se compara con el pedido esperado, siguiendo la lógica actual de verificación.

La lógica de tickets→baldas, generación de JSON y verificación pedido vs detectado permanece igual; el árbol sustituye la parte de "cuántas unidades reales representa este conjunto de detecciones".

---

## 7. Modelo de tallos (fase 2, pendiente)

Fase futura no bloqueante para la PoC:

- Modelo separado por detección de tipo `tallogrupo`.
- Objetivo inicial: clasificar cada tallo como `tallo_valido` vs `tallo_duplicado/ruido`, usando:
  - Volumen/máscara de tallo, score de Mask R‑CNN.
  - Posición relativa en balda y lado.
  - Vecinos izquierda/derecha y su solapamiento.
  - Distancia a la masa de planta/flor más cercana.
  - Densidad local de tallos en la balda.
- Integración: pre‑filtro de tallos antes de que sus conteos se usen como features en el modelo principal.

Se abordará una vez maduro el modelo principal y cuando exista dataset etiquetado específico de tallos.

---

## 8. Dataclass `DetectionFeat`

En `scripts/05_logica_conteo_tallos/decision_tree_features.py`:

```python
from dataclasses import dataclass

@dataclass
class DetectionFeat:
    carro_id: str
    image_pair_id: str
    detection_id: int
    ticket_idx: int
    balda_idx: int
    tipo_d: str
    sku_d: str
    lado_d: str
    pos_rel_d: float
    volumen_d: float
    score_mrcnn_d: float
    score_convnext_d: float
    es_superpuesto_d: int
    # Vecino derecha
    vec_der_existe_d: int
    vec_der_tipo: str
    vec_der_sku: str
    vec_der_pos_rel: float
    vec_der_volumen: float
    vec_der_score_mrcnn: float
    vec_der_score_convnext: float
    vec_der_superpuesto: int
    vec_der_superpuesto2: int
    # Vecino izquierda
    vec_izq_existe_d: int
    vec_izq_tipo: str
    vec_izq_sku: str
    vec_izq_pos_rel: float
    vec_izq_volumen: float
    vec_izq_score_mrcnn: float
    vec_izq_score_convnext: float
    vec_izq_superpuesto: int
    vec_izq_superpuesto2: int
    # Otro lado
    otro_lado_existe_d: int
    otro_lado_tipo: str
    otro_lado_sku: str
    otro_lado_pos_rel: float
    otro_lado_volumen: float
    otro_lado_score_mrcnn: float
    otro_lado_score_convnext: float
    otro_lado_dist_pos_rel: float
    otro_lado_superpuesto: int
    otro_lado_superpuesto2: int
    # Vecinos del otro lado
    otro_lado_vec_der_existe: int
    otro_lado_vec_der_tipo: str
    otro_lado_vec_der_pos_rel: float
    otro_lado_vec_der_volumen: float
    otro_lado_vec_der_score_mrcnn: float
    otro_lado_vec_der_score_convnext: float
    otro_lado_vec_izq_existe: int
    otro_lado_vec_izq_tipo: str
    otro_lado_vec_izq_pos_rel: float
    otro_lado_vec_izq_volumen: float
    otro_lado_vec_izq_score_mrcnn: float
    otro_lado_vec_izq_score_convnext: float
    # Agregados
    num_mismo_tipo_misma_balda_misma_vista: int
    num_mismo_tipo_misma_balda_otro_lado: int
    num_tallos_misma_balda: int
    # Etiquetas
    unidades_label_d: int       # -1 si sin etiquetar
    unidades_pred_regla: int    # 0 si no aplica

def detection_to_dict(d: DetectionFeat) -> dict:
    return d.__dict__.copy()
```

---

## 9. Firmas de funciones en `decision_tree_features.py`

Antigravity debe buscar en `05_conteo.py` las funciones que ya obtienen detecciones, tickets y baldas, y reutilizarlas aquí.

```python
from typing import List

def construir_detecciones_enriquecidas(
    det_front: list,
    det_back: list,
    asign_tickets,
    baldas_f,
    baldas_b,
    carro_id: str,
    image_pair_id: str,
) -> List[DetectionFeat]:
    """
    Asigna ticket_idx y balda_idx a cada detección.
    Calcula pos_rel_d (posición X relativa a la balda, [0,1]),
    volumen_d (área de máscara) y scores de Mask R‑CNN y ConvNeXt.
    Devuelve lista de DetectionFeat con campos básicos rellenados.
    """
    ...

def enlazar_vecinos_misma_vista(dets: List[DetectionFeat]) -> None:
    """
    Agrupa por (ticket_idx, balda_idx, lado_d), ordena por pos_rel_d
    y rellena vec_der_* y vec_izq_* en cada DetectionFeat.
    """
    ...

def enlazar_corresponsales_otra_vista(dets: List[DetectionFeat]) -> None:
    """
    Para cada DetectionFeat, busca en la vista opuesta
    (mismo ticket/balda y tipo) la detección con pos_rel más cercana.
    Rellena campos otro_lado_* y otro_lado_vec_der_*, otro_lado_vec_izq_*.
    """
    ...

def calcular_agregados_contexto(dets: List[DetectionFeat]) -> None:
    """
    Para cada DetectionFeat calcula agregados por (ticket_idx, balda_idx):
    num_mismo_tipo_misma_balda_misma_vista,
    num_mismo_tipo_misma_balda_otro_lado,
    num_tallos_misma_balda.
    """
    ...

def exportar_features_dataset_final() -> None:
    """
    Recorre data/dataset_final en busca de pares *_F.* / *_B.*.
    Para cada par:
      - Carga modelos (reutilizar lógica de 05_conteo.py).
      - Ejecuta detección + clasificación.
      - Construye lista de DetectionFeat.
      - Llama a enlazar_vecinos_misma_vista,
        enlazar_corresponsales_otra_vista,
        calcular_agregados_contexto.
      - Vuelca a detections_raw.csv (append).
    Guarda en scripts/05_logica_conteo_tallos/detections_raw.csv.
    """
    ...
```

---

## 10. Gestión de CSVs con Pandas (`decision_tree_data.py`)

En `scripts/05_logica_conteo_tallos/decision_tree_data.py`:

```python
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
RAW_CSV    = BASE_DIR / "detections_raw.csv"
LABELS_CSV = BASE_DIR / "detections_labels.csv"
MERGED_CSV = BASE_DIR / "detections_labeled.csv"
KEY_COLS   = ["image_pair_id", "detection_id"]

def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)

def load_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV)
    return pd.DataFrame(columns=KEY_COLS + ["unidades_label_d"])

def save_labels(df_labels: pd.DataFrame) -> None:
    df_labels.to_csv(LABELS_CSV, index=False)

def merge_raw_labels() -> pd.DataFrame:
    df_raw = load_raw()
    df_lab = load_labels()
    df = df_raw.merge(df_lab, on=KEY_COLS, how="left")
    df["unidades_label_d"] = df["unidades_label_d"].fillna(-1).astype(int)
    df.to_csv(MERGED_CSV, index=False)
    return df
```

`detections_labels.csv` contiene solo `image_pair_id`, `detection_id`, `unidades_label_d`.

---

## 11. Pipeline de entrenamiento (`decision_tree_train.py`)

En `scripts/05_logica_conteo_tallos/decision_tree_train.py`:

1. Leer `detections_labeled.csv` con `decision_tree_data.merge_raw_labels()`.
2. Filtrar:
   - `tipo_d ∈ {flor, planta}`.
   - `unidades_label_d >= 0`.
3. Construir `X` (features) e `y` (`unidades_label_d`).
4. Aplicar `ColumnTransformer`:
   - One‑hot sobre columnas categóricas (`tipo_d`, `lado_d`, `*_tipo`).
   - Escalado opcional sobre numéricas.
5. Entrenar `DecisionTreeRegressor` o `RandomForestRegressor` (sklearn).
6. Guardar modelo en `models/tree_conteo.pkl`.

---

## 12. Módulos de etiquetado en `test_area/app.py` (Antigravity)

### 12.1. Selector de módulo

Añadir al `st.sidebar.selectbox` existente:

```python
mode = st.sidebar.selectbox(
    "Módulo",
    [
        "Mask R-CNN",
        "ConvNeXt",
        "Conteo Pipeline",
        "Árbol – Etiquetar detecciones",
        "Árbol – Ver anotaciones",
    ]
)

if mode == "Árbol – Etiquetar detecciones":
    modulo_etiquetado_arbol()
elif mode == "Árbol – Ver anotaciones":
    modulo_visualizacion_arbol()
```

### 12.2. `modulo_etiquetado_arbol()`

1. **Selección de imagen:**
   - Escanear `data/dataset_final`, listar pares `*_F.*` / `*_B.*`.
   - Extraer `image_pair_id` y mostrar selector Streamlit.

2. **Inferencia:**
   - Cargar modelos existentes (mismos paths que `05_conteo.py`).
   - Ejecutar detección + clasificación.
   - Llamar a `construir_detecciones_enriquecidas`, `enlazar_vecinos_misma_vista`, `enlazar_corresponsales_otra_vista`, `calcular_agregados_contexto`.
   - Filtrar `tipo_d ∈ {flor, planta}` para la UI.

3. **UI:**
   - Imagen frontal con bboxes/máscaras de flor/planta y `detection_id` dibujado encima.
   - Opcional: imagen trasera al lado en segunda columna.
   - Panel lateral con tabla de detecciones:
     - campos: `detection_id`, `tipo_d`, `sku_d`, `ticket_idx`, `balda_idx`, `lado_d`, `pos_rel_d`, `volumen_d`, `score_mrcnn_d`, `score_convnext_d`.
     - `st.number_input` por detección para `unidades_label_d` (por defecto valor guardado o 0).

4. **Persistencia:**
   - Leer estado actual con `decision_tree_data.load_labels()`.
   - Al pulsar "Guardar anotaciones": actualizar/insertar filas por (`image_pair_id`, `detection_id`).
   - Llamar a `decision_tree_data.save_labels(df_labels)`.
   - No modificar `detections_raw.csv` desde la app.

### 12.3. `modulo_visualizacion_arbol()`

1. **Cargar datos:**
   - Llamar a `decision_tree_data.merge_raw_labels()`.

2. **Selectores:**
   - `image_pair_id` y `ticket_idx` disponibles (y opcionalmente `sku_d`).

3. **Visualización:**
   - Imagen frontal con bultos de ese ticket resaltados y texto `unidades_label_d` + `sku_d`.
   - Tabla resumen por `sku_d` (y opcionalmente `balda_idx`):
     - columnas: `sku_d`, `tipo_d`, `unidades_totales_ticket_sku`, `num_detecciones_sku`.

4. **UX:**
   - Botón para filtrar y resaltar detecciones con `unidades_label_d < 0` (sin etiquetar).
   - Navegación entre pares de imágenes con selectores.

---

## 13. Notas para Antigravity

- Respetar arquitectura descrita en `README.md` y `WORKFLOW.md`: Mask R‑CNN + EfficientNet‑B4, scripts numerados, app bajo `test_area/`.
- No modificar la lógica existente de:
  - Asignación de tickets a baldas.
  - Conteo actual en `05_conteo.py` (solo reutilizar funciones).
- Nuevos archivos:
  - `scripts/05_logica_conteo_tallos/decision_tree_features.py`
  - `scripts/05_logica_conteo_tallos/decision_tree_data.py`
  - `scripts/05_logica_conteo_tallos/decision_tree_train.py`
- CSVs generados en `scripts/05_logica_conteo_tallos/`:
  - `detections_raw.csv` (generado por `exportar_features_dataset_final()`).
  - `detections_labels.csv` (actualizado desde la app).
  - `detections_labeled.csv` (merge de los dos anteriores para entrenamiento).
- Toda la nueva funcionalidad debe ser parametrizable por rutas y no romper los modos actuales de la app Streamlit.
- Añadir en `README.md` un apartado breve "Árbol de Conteo – Entrenamiento y Etiquetado" describiendo los scripts, CSVs y uso de la app.
