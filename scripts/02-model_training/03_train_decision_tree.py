"""
03_train_decision_tree.py
Pipeline de entrenamiento del árbol de conteo.
Ver documentation/ARBOL_CONTEO.md sección 11.

Uso:
    /home/servi2/Enviroments/main_venv/bin/python3 scripts/02-model_training/03_train_decision_tree.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# Asegurar imports locales (decision_tree_data está en 05-logica_conteo_tallos)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONTEO_DIR = os.path.join(_PROJECT_ROOT, "scripts", "05-logica_conteo_tallos")
if _CONTEO_DIR not in sys.path:
    sys.path.insert(0, _CONTEO_DIR)

import decision_tree_data as dtd

# Columnas que NO son features
EXCLUDE_COLS = [
    "carro_id", "image_pair_id", "detection_id",
    "ticket_idx", "balda_idx",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "unidades_label_d", "unidades_pred_regla",
]

# Columnas categóricas (one-hot)
CAT_COLS = [
    "tipo_d", "lado_d",
    "vec_der_tipo", "vec_izq_tipo",
    "otro_lado_tipo",
    "otro_lado_vec_der_tipo", "otro_lado_vec_izq_tipo",
]

# Columnas string que se excluyen de features
SKU_COLS = ["sku_d", "vec_der_sku", "vec_izq_sku", "otro_lado_sku"]


def train_model(output_dir=None, n_estimators=100, max_depth=None, random_state=42):
    """
    Entrena un RandomForestRegressor desde detections_labeled.csv.
    Guarda en models/tree_conteo/tree_conteo.pkl.
    """
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "models", "tree_conteo")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar datos
    df = dtd.merge_raw_labels()
    if df.empty:
        print("[ERROR] No hay datos. Ejecuta primero 04_build_tree_features.py.")
        return ""

    # 2. Filtrar solo flor/planta con etiqueta
    df_train = df[
        (df["tipo_d"].isin(["flor", "planta"])) &
        (df["unidades_label_d"] >= 0)
    ].copy()

    if len(df_train) < 5:
        print(f"[ERROR] Solo {len(df_train)} muestras etiquetadas. Mínimo 5.")
        return ""

    print(f"[*] Muestras: {len(df_train)}")
    print(f"    Distribución unidades_label_d:")
    print(df_train["unidades_label_d"].value_counts().sort_index().to_string())

    # 3. Preparar X e y
    y = df_train["unidades_label_d"].values.astype(float)
    drop_cols = [c for c in EXCLUDE_COLS + SKU_COLS if c in df_train.columns]
    X = df_train.drop(columns=drop_cols)

    cat_cols_present = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols_present]

    # 4. ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_present),
        ("num", "passthrough", num_cols),
    ])

    X_transformed = preprocessor.fit_transform(X)
    feature_names = (
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols_present).tolist()
        + num_cols
    )
    print(f"[*] Features totales: {len(feature_names)}")

    # 5. Entrenar
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1,
    )

    if len(df_train) >= 20:
        cv = cross_val_score(model, X_transformed, y, cv=5, scoring="neg_mean_absolute_error")
        print(f"[*] CV MAE (5-fold): {-cv.mean():.3f} ± {cv.std():.3f}")

    model.fit(X_transformed, y)

    y_pred = model.predict(X_transformed)
    print(f"[*] Train MAE: {mean_absolute_error(y, y_pred):.3f} | R²: {r2_score(y, y_pred):.3f}")

    # Top 15 features
    imp = model.feature_importances_
    top = np.argsort(imp)[::-1][:15]
    print("\n[*] Top 15 features:")
    for i in top:
        print(f"    {feature_names[i]:45s} {imp[i]:.4f}")

    # 6. Guardar
    model_path = os.path.join(output_dir, "tree_conteo.pkl")
    joblib.dump({
        "model": model, "preprocessor": preprocessor,
        "feature_names": feature_names,
        "cat_cols": cat_cols_present, "num_cols": num_cols,
    }, model_path)
    print(f"\n[+] Modelo guardado: {model_path}")
    return model_path


if __name__ == "__main__":
    train_model()
