"""
upload_to_roboflow.py — Sube los crops de Flores/Planta a Roboflow.

Sube todas las imágenes de data/crops_clasificacion/ al proyecto
"proyecto_h_clas" del workspace "floresverdnatura", usando el tag
de la subcarpeta (Flores o Planta) para identificarlas en Roboflow.
"""

import os
import glob
from roboflow import Roboflow

# =============================================================================
# CONFIGURACIÓN (MODIFICAR AQUÍ)
# =============================================================================

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "EN7iQhMMAT3BFZUkJhUb")  # Configura: export ROBOFLOW_API_KEY="tu_key"
WORKSPACE = "floresverdnatura"
PROJECT = "proyecto_h_clas"

# Carpeta con los crops organizados en subcarpetas por clase
CROPS_DIR = os.path.join("data", "crops_clasificacion_v3")


# =============================================================================
# LÓGICA DE SUBIDA
# =============================================================================

def main():
    print("=" * 60)
    print(" UPLOAD TO ROBOFLOW — Subida de crops")
    print("=" * 60)

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    # Recorrer cada subcarpeta (Flores, Planta)
    subfolders = [d for d in os.listdir(CROPS_DIR)
                  if os.path.isdir(os.path.join(CROPS_DIR, d))]

    if not subfolders:
        print(f"[!] No se encontraron subcarpetas en '{CROPS_DIR}'.")
        return

    total_uploaded = 0
    total_errors = 0

    for folder_name in sorted(subfolders):
        folder_path = os.path.join(CROPS_DIR, folder_name)
        images = glob.glob(os.path.join(folder_path, "*.png"))
        images += glob.glob(os.path.join(folder_path, "*.jpg"))
        images += glob.glob(os.path.join(folder_path, "*.jpeg"))

        print(f"\n[*] Subiendo {len(images)} imágenes de '{folder_name}'...")

        for i, img_path in enumerate(images):
            try:
                project.upload(
                    img_path,
                    tag_names=[folder_name],  # Tag = nombre de la subcarpeta
                    batch_name=f"crops_{folder_name}"
                )
                total_uploaded += 1
                # Progreso cada 50 imágenes
                if (i + 1) % 50 == 0:
                    print(f"    -> {i + 1}/{len(images)} subidas...")
            except Exception as e:
                total_errors += 1
                print(f"    [ERROR] {os.path.basename(img_path)}: {e}")

        print(f"    -> {folder_name}: completado")

    print("\n" + "=" * 60)
    print(" RESUMEN DE SUBIDA")
    print("=" * 60)
    print(f"  Total subidas correctamente: {total_uploaded}")
    print(f"  Total errores: {total_errors}")
    print(f"\n  Revisa tu proyecto en:")
    print(f"  https://app.roboflow.com/{WORKSPACE}/{PROJECT}/upload")


if __name__ == "__main__":
    main()
