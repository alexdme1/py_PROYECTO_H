import os
import cv2
import json
import random
import numpy as np

# Colores BGR distintos para asignar a cada categoría según su ID
# Si hay más IDs, se repetirá el color por módulo.
PALETTE = [
    (0, 0, 255),    # Rojo
    (255, 255, 0),  # Cian
    (0, 255, 255),  # Amarillo
    (0, 255, 0),    # Verde
    (255, 0, 255),  # Magenta
    (255, 128, 0),  # Azul claro
    (128, 0, 255),  # Rosa
]

def get_color_for_id(cat_id):
    return PALETTE[cat_id % len(PALETTE)]

def draw_ground_truth(json_path, images_dir, output_dir, num_samples=20):
    if not os.path.exists(json_path):
        print(f"[!] No se encontró '{json_path}'. Ejecuta 01_fix_coco.py primero.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # LEER CATEGORÍAS DIRECTAMENTE DEL JSON CREADO POR FIX_COCO
    categories = data.get("categories", [])
    category_names = {cat["id"]: cat["name"] for cat in categories}

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    
    # Crear un diccionario rápido para buscar anotaciones por image_id
    ann_by_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    if not images:
        print("[!] No hay imágenes en el JSON.")
        return

    samples = random.sample(images, min(len(images), num_samples))
    print(f"[*] Dibujando {len(samples)} imágenes aleatorias...")

    for img_info in samples:
        img_name = img_info["file_name"]
        img_id = img_info["id"]
        img_path = os.path.join(images_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"    [WARNING] No se pudo leer la imagen '{img_path}'.")
            continue

        # Crear una copia para la máscara superpuesta (efecto translúcido)
        overlay = img.copy()
        
        # Obtener todas las anotaciones de esta imagen
        anns = ann_by_image.get(img_id, [])

        for ann in anns:
            cat_id = ann.get("category_id", -1)
            cat_name = category_names.get(cat_id, f"Unknown_{cat_id}")
            color = get_color_for_id(cat_id)

            # 1. Dibujar Segmentación (Máscara translúcida)
            if "segmentation" in ann and ann["segmentation"]:
                for poly in ann["segmentation"]:
                    if len(poly) >= 6:
                        # Convertir lista plana [x1, y1, x2, y2...] a numpy array de pares [[x1, y1], [x2, y2]...]
                        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], color)
                        # También dibujar el borde del polígono un poco más fuerte
                        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            # 2. Dibujar Bounding Box
            if "bbox" in ann and len(ann["bbox"]) == 4:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Etiqueta de clase sobre la caja
                text = f"{cat_name}"
                # Pequeño fondo para que el texto se lea bien
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x, y - th - 5), (x + tw, y), color, -1)
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Fusionar la máscara translúcida (overlay) con la imagen base
        alpha = 0.4  # Transparencia de la máscara
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        out_path = os.path.join(output_dir, f"gt_{img_name}")
        cv2.imwrite(out_path, img)

    print(f"\n[+] Proceso finalizado. Visualizaciones guardadas en '{output_dir}'.")
    print(" Si todo cuadra (colores, clases y máscaras ceñidas a los objetos), el dataset unificado está perfecto.")

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    NUM_IMAGES_TO_DRAW = 20
    
    json_train = os.path.join("data", "coco_unified", "annotations", "train.json")
    images_folder = os.path.join("data", "coco_unified", "images")
    output_folder = os.path.join("data", "coco_unified", "trust_vis")
    
    draw_ground_truth(json_train, images_folder, output_folder, NUM_IMAGES_TO_DRAW)
