"""
download_models.py — Descarga los modelos entrenados desde Hugging Face Hub.

Uso:
    python scripts/download_models.py

Los modelos se descargan a la carpeta models/ del proyecto.
"""
import os
import sys

# Configuración del repositorio en Hugging Face
HF_REPO_ID = "alexdme/proyecto-h-models"

# Mapeo: nombre del archivo en HF → ruta local relativa (desde la raíz del proyecto)
FILES_TO_DOWNLOAD = {
    # Mask R-CNN
    "maskrcnn/model_final.pth": os.path.join("models", "maskrcnn", "default", "model_final.pth"),

    # ConvNeXt
    "convnext/best_model.pth": os.path.join("models", "convnext", "v1_baseline", "best_model.pth"),
    "convnext/class_names.txt": os.path.join("models", "convnext", "v1_baseline", "class_names.txt"),
    "convnext/config_used.yaml": os.path.join("models", "convnext", "v1_baseline", "config_used.yaml"),
}


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[!] huggingface_hub no está instalado.")
        print("    Instálalo con: pip install huggingface_hub")
        sys.exit(1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"[*] Descargando modelos desde Hugging Face: {HF_REPO_ID}")
    print(f"    Destino: {os.path.join(project_root, 'models')}\n")

    for hf_filename, local_path in FILES_TO_DOWNLOAD.items():
        abs_path = os.path.join(project_root, local_path)

        # Si ya existe, saltar
        if os.path.exists(abs_path):
            print(f"  [✓] Ya existe: {local_path}")
            continue

        # Crear directorio destino
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        print(f"  [↓] Descargando: {hf_filename} → {local_path} ...", end=" ", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_filename,
                local_dir=os.path.join(project_root, "models", "_hf_cache"),
                local_dir_use_symlinks=False,
            )
            # Mover a la ubicación final
            import shutil
            shutil.move(downloaded, abs_path)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n[✓] Descarga completada.")
    print("    Ahora puedes ejecutar: python test_area/app.py")


if __name__ == "__main__":
    main()
