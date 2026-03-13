"""
decision_tree_data.py
Gestión de CSVs para el árbol de conteo.
CSVs en data/arbol_conteo/.
Ver documentation/ARBOL_CONTEO.md sección 10.
"""

import os
import pandas as pd
from pathlib import Path

# CSVs en data/arbol_conteo/ (raíz del proyecto)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = _PROJECT_ROOT / "data" / "arbol_conteo"
RAW_CSV    = DATA_DIR / "detections_raw.csv"
LABELS_CSV = DATA_DIR / "detections_labels.csv"
MERGED_CSV = DATA_DIR / "detections_labeled.csv"
KEY_COLS   = ["image_pair_id", "detection_id"]


def load_raw() -> pd.DataFrame:
    """Carga detections_raw.csv. Devuelve DataFrame vacío si no existe."""
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    return pd.DataFrame()


def load_labels() -> pd.DataFrame:
    """Carga detections_labels.csv. Si no existe devuelve DataFrame vacío."""
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV)
    return pd.DataFrame(columns=KEY_COLS + ["unidades_label_d"])


def save_labels(df_labels: pd.DataFrame) -> None:
    """Guarda etiquetas a detections_labels.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_labels.to_csv(LABELS_CSV, index=False)
    print(f"[+] Labels guardadas: {LABELS_CSV} ({len(df_labels)} filas)")


def merge_raw_labels() -> pd.DataFrame:
    """
    Join de raw + labels por (image_pair_id, detection_id).
    Rellena unidades_label_d = -1 si no hay etiqueta.
    Guarda detections_labeled.csv y devuelve el DataFrame.
    """
    df_raw = load_raw()
    if df_raw.empty:
        print("[!] detections_raw.csv vacío o no existe.")
        return df_raw

    df_lab = load_labels()

    if "unidades_label_d" in df_raw.columns:
        df_raw = df_raw.drop(columns=["unidades_label_d"])

    if df_lab.empty:
        df = df_raw.copy()
        df["unidades_label_d"] = -1
    else:
        df = df_raw.merge(df_lab[KEY_COLS + ["unidades_label_d"]], on=KEY_COLS, how="left")
        df["unidades_label_d"] = df["unidades_label_d"].fillna(-1).astype(int)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)
    print(f"[+] Merged CSV: {MERGED_CSV} ({len(df)} filas, {(df['unidades_label_d'] >= 0).sum()} etiquetadas)")
    return df
