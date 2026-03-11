import os
import json
import shutil
import re
import cv2

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE CATEGORÍAS
# -----------------------------------------------------------------------------
CATEGORY_MAP = {
    "Flores": 0,
    "0": 1,
    "Balda": 2,
    "Balda1": 2,
    "Balda2": 2,
    "Balda3": 2,
    "Planta": 3,
    "tallo_grupo": 4
}

# IDs de categoría a IGNORAR (superclases de Roboflow)
SKIP_CATEGORY_IDS = {0}

FINAL_CATEGORIES = [
    {"id": 0, "name": "Flores", "supercategory": "PRODUCTO"},
    {"id": 1, "name": "ticket", "supercategory": "CARRO"},
    {"id": 2, "name": "Balda", "supercategory": "CARRO"},
    {"id": 3, "name": "Planta", "supercategory": "PRODUCTO"},
    {"id": 4, "name": "tallo_grupo", "supercategory": "PRODUCTO"}
]

def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def process_bbox(bbox):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    valid_bbox = []
    for val in bbox:
        fval = to_float(val)
        if fval is None:
            return None
        valid_bbox.append(fval)
    return valid_bbox

def process_segmentation(seg):
    if not seg or not isinstance(seg, list):
        return None
    valid_seg = []
    for poly in seg:
        if not isinstance(poly, list):
            continue
        new_poly = []
        for val in poly:
            fval = to_float(val)
            if fval is None:
                break
            new_poly.append(fval)
        else: 
            if len(new_poly) >= 6:
                valid_seg.append(new_poly)
    if not valid_seg:
        return None
    return valid_seg

def fix_and_merge_dataset(input_dir, output_dir):
    # Extraer el nombre entre paréntesis (ej. big_aug, lil_aug, no_aug)
    model_name = "baseline"
    dirname = os.path.basename(os.path.normpath(input_dir))
    match = re.search(r'\((.*?)\)', dirname)
    if match:
        model_name = match.group(1)
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Modelo detectado por paréntesis: {model_name}")
    with open(os.path.join(output_dir, "model_name.txt"), "w") as f:
        f.write(model_name)
    
    out_img_dir = os.path.join(output_dir, "images")
    out_ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)
    
    splits = ["train", "valid", "test"]
    stats_total = {"images": 0, "annotations_kept": 0, "annotations_dropped": 0}

    for split in splits:
        split_dir = os.path.join(input_dir, split)
        json_path = os.path.join(split_dir, "_annotations.coco.json")
        
        if not os.path.exists(json_path):
            print(f"[-] Split {split} no encontrado. Omitiendo.")
            continue
            
        print(f"[*] Procesando split: {split}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        final_info = data.get("info", {"description": f"Dataset - Split {split}"})
        final_licenses = data.get("licenses", [])
        original_categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        
        final_images = []
        image_orig_widths = {} # Para la rotación de anotaciones

        for img in data.get("images", []):
            old_filename = img["file_name"]
            new_filename = f"{split}_{old_filename}"
            src_img_path = os.path.join(split_dir, old_filename)
            dst_img_path = os.path.join(out_img_dir, new_filename)
            
            orig_w = img.get("width")
            orig_h = img.get("height")

            if os.path.exists(src_img_path):
                # Leer imagen, rotar 90 grados a la izquierda, y guardar
                image = cv2.imread(src_img_path)
                if image is not None:
                    if orig_w is None or orig_h is None:
                        orig_h, orig_w = image.shape[:2]
                    rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(dst_img_path, rotated_img)
                else:
                    print(f"[-] Aviso: No se pudo leer {src_img_path} con cv2")
                    continue
            else:
                continue
                
            if orig_w is not None:
                image_orig_widths[img["id"]] = orig_w
                
            img["file_name"] = new_filename
            # Intercambiar dimensiones en el JSON para el nuevo formato
            img["width"] = orig_h
            img["height"] = orig_w
            
            final_images.append(img)
            stats_total["images"] += 1
            
        final_annotations = []
        dropped = 0
        kept = 0
        
        for ann in data.get("annotations", []):
            # Saltar superclases por ID
            raw_cat_id = ann.get("category_id")
            if raw_cat_id in SKIP_CATEGORY_IDS:
                dropped += 1
                continue

            original_cat_name = original_categories.get(raw_cat_id, "Unknown")
            if original_cat_name not in CATEGORY_MAP:
                print(f"  [!] Categoría desconocida '{original_cat_name}' (id={raw_cat_id}). Descartando.")
                dropped += 1
                continue
            
            new_cat_id = CATEGORY_MAP[original_cat_name]
            
            new_bbox = process_bbox(ann.get("bbox"))
            if new_bbox is None:
                dropped += 1
                continue
                
            new_seg = process_segmentation(ann.get("segmentation"))
            if new_seg is None:
                # Si no hay segmentación, generar un polígono rectangular desde la bbox
                x, y, w, h = new_bbox
                new_seg = [[x, y, x+w, y, x+w, y+h, x, y+h]]

            orig_w = image_orig_widths.get(ann["image_id"])
            if orig_w is None:
                dropped += 1
                continue

            # Rotar bbox 90 grados a la izquierda (CCW)
            # bbox = [x_min, y_min, w, h] => [y_min, orig_w - x_min - w, h, w]
            x_min, y_min, bw, bh = new_bbox
            rot_bbox = [y_min, orig_w - x_min - bw, bh, bw]
            
            # Asegurarnos de que las coordenadas no sean negativas por errores de flotantes
            rot_bbox = [max(0, cord) for cord in rot_bbox]

            # Rotar segmentación 90 grados a la izquierda (CCW)
            # punto (x, y) => (y, orig_w - x)
            rot_seg = []
            for poly in new_seg:
                r_poly = []
                for i in range(0, len(poly), 2):
                    px = poly[i]
                    py = poly[i+1]
                    r_poly.extend([py, orig_w - px])
                rot_seg.append(r_poly)
            
            ann["category_id"] = new_cat_id
            ann["bbox"] = rot_bbox
            ann["segmentation"] = rot_seg
            ann["iscrowd"] = 0 
            ann["area"] = to_float(ann.get("area")) or (rot_bbox[2] * rot_bbox[3])
                
            final_annotations.append(ann)
            kept += 1
            
        stats_total["annotations_kept"] += kept
        stats_total["annotations_dropped"] += dropped
        print(f"    - Anotaciones guardadas: {kept}, descartadas: {dropped}")

        # Guardar archivo json separado por cada split (train.json, valid.json, test.json)
        # de esta forma, test sigue siendo completamente apartado y los splits son limpios.
        unified_json = {
            "info": final_info,
            "licenses": final_licenses,
            "categories": FINAL_CATEGORIES,
            "images": final_images,
            "annotations": final_annotations
        }
        
        out_json_path = os.path.join(out_ann_dir, f"{split}.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(unified_json, f, indent=2)

if __name__ == "__main__":
    # --- RUTAS PRINCIPALES (MODIFICAR AQUÍ PARA CAMBIAR DE DATASET DE ENTRADA) ---
    input_roboflow_dir = os.path.join("data", "Proyecto_H.v5i.coco(aug_mas_data)")
    output_unified_dir = os.path.join("data", "coco_unified")
    
    fix_and_merge_dataset(input_roboflow_dir, output_unified_dir)
