"""
00_cropping.py — Extrae crops de Flores y Plantas desde el export de Roboflow.

Lee directamente de la carpeta de Roboflow (train/, valid/, test/), donde cada
split tiene su _annotations.coco.json y las imágenes dentro de la propia carpeta.

Filtra solo las categorías "Flores" y "Planta", y recorta las bounding boxes.
Los crops se guardan SIN padding ni resize para verlos claros en Roboflow.

Salida:
    data/crops_clasificacion/
    ├── Flores/
    │   ├── train_img001_crop_0.png
    │   └── ...
    └── Planta/
        ├── valid_img042_crop_7.png
        └── ...
"""

import os
import json
import cv2

# =============================================================================
# CONFIGURACIÓN (MODIFICAR AQUÍ)
# =============================================================================

# Carpeta raíz del export de Roboflow (contiene train/, valid/, test/)
ROBOFLOW_DIR = os.path.join("data", "Proyecto_H-TRASERAS_V2.coco")

# Splits a procesar (cada uno tiene sus imágenes + _annotations.coco.json dentro)
SPLITS = ["train", "valid", "test"]

# Carpeta de salida para los crops
OUTPUT_DIR = os.path.join("data", "crops_clasificacion_v2")

# Nombres de categorías que queremos recortar (tal cual aparecen en Roboflow)
TARGET_CATEGORY_NAMES = {"Flores", "Planta"}

# IDs de categoría a IGNORAR (ej. superclases duplicadas de Roboflow)
# En FRONTALES_V2, id=0 es una superclase "Flores" que no queremos
EXCLUDE_CATEGORY_IDS = {0}

# Tamaño mínimo del crop en píxeles (ancho o alto). Si el crop es más pequeño
# que esto en cualquier dimensión, se descarta (probablemente es ruido).
MIN_CROP_SIZE = 1


# =============================================================================
# LÓGICA DE CROPPING
# =============================================================================

def process_split(roboflow_dir, split, output_dir, target_names, min_crop_size, crop_id_start):
    """
    Procesa un split individual de Roboflow.
    Las imágenes están DENTRO de la carpeta del split junto al JSON.
    """
    split_dir = os.path.join(roboflow_dir, split)
    json_path = os.path.join(split_dir, "_annotations.coco.json")

    if not os.path.exists(json_path):
        print(f"  [-] {split}/_annotations.coco.json no encontrado. Omitiendo.")
        return {}, 0, 0, 0, crop_id_start

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Mapeo de IDs originales de Roboflow → nombre de categoría
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Indexar imágenes por ID
    images_by_id = {img["id"]: img for img in data.get("images", [])}

    # Estadísticas para este split
    stats = {name: 0 for name in target_names}
    skipped_small = 0
    skipped_missing = 0
    skipped_read_error = 0

    # Cache de imágenes leídas (para no leer la misma imagen N veces)
    img_cache = {}
    crop_global_id = crop_id_start

    for ann in data.get("annotations", []):
        # Saltar IDs excluidos (superclases duplicadas)
        cat_id = ann.get("category_id")
        if cat_id in EXCLUDE_CATEGORY_IDS:
            continue

        # Resolver nombre de categoría desde el ID de Roboflow
        cat_name = cat_id_to_name.get(cat_id, "")
        if cat_name not in target_names:
            continue

        # Buscar la imagen
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            skipped_missing += 1
            continue

        file_name = img_info["file_name"]
        # Las imágenes están DENTRO de la carpeta del split
        img_path = os.path.join(split_dir, file_name)

        # Leer imagen (con cache)
        if img_path not in img_cache:
            img = cv2.imread(img_path)
            if img is None:
                skipped_read_error += 1
                img_cache[img_path] = None
                continue
            img_cache[img_path] = img
        else:
            img = img_cache[img_path]
            if img is None:
                continue

        img_h, img_w = img.shape[:2]

        # Extraer bbox COCO: [x, y, width, height]
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue

        x, y, w, h = [int(round(float(v))) for v in bbox]

        # Clampear la bbox a los límites de la imagen
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Descartar crops demasiado pequeños
        if crop_w < min_crop_size or crop_h < min_crop_size:
            skipped_small += 1
            continue

        # Recortar y rotar 90° a la izquierda (counter-clockwise)
        crop = img[y1:y2, x1:x2]
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Nombre del crop: split_nombreBase_cropID.png
        base_name = os.path.splitext(file_name)[0]
        crop_name = f"{split}_{base_name}_crop_{crop_global_id}.png"
        crop_path = os.path.join(output_dir, cat_name, crop_name)

        cv2.imwrite(crop_path, crop)

        stats[cat_name] += 1
        crop_global_id += 1

    return stats, skipped_small, skipped_missing, skipped_read_error, crop_global_id


def main():
    print("=" * 60)
    print(" 00_CROPPING — Extracción de crops Flores/Planta")
    print("=" * 60)
    print(f"\n[*] Leyendo dataset de: {ROBOFLOW_DIR}")

    # Crear subcarpetas de salida
    for cat_name in TARGET_CATEGORY_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, cat_name), exist_ok=True)

    # Acumuladores globales
    total_stats = {name: 0 for name in TARGET_CATEGORY_NAMES}
    total_skipped_small = 0
    total_skipped_missing = 0
    total_skipped_read_error = 0
    crop_id = 0

    for split in SPLITS:
        print(f"\n[*] Procesando split: {split}")
        stats, sk_small, sk_miss, sk_err, crop_id = process_split(
            ROBOFLOW_DIR, split, OUTPUT_DIR,
            TARGET_CATEGORY_NAMES, MIN_CROP_SIZE, crop_id
        )
        for name in TARGET_CATEGORY_NAMES:
            total_stats[name] += stats.get(name, 0)
        total_skipped_small += sk_small
        total_skipped_missing += sk_miss
        total_skipped_read_error += sk_err

        split_total = sum(stats.get(n, 0) for n in TARGET_CATEGORY_NAMES)
        print(f"    -> {split_total} crops extraídos en este split")

    print("\n" + "=" * 60)
    print(" RESUMEN DE EXTRACCIÓN")
    print("=" * 60)
    for cat_name, count in total_stats.items():
        print(f"  {cat_name:>15}: {count} crops guardados")
    print(f"  {'TOTAL':>15}: {sum(total_stats.values())} crops")
    print(f"\n  Descartados (muy pequeños): {total_skipped_small}")
    print(f"  Descartados (imagen no encontrada): {total_skipped_missing}")
    print(f"  Descartados (error al leer imagen): {total_skipped_read_error}")
    print(f"\n  Carpeta de salida: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
