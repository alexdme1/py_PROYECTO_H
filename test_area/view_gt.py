import os
import cv2
import json
import argparse
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def main():
    parser = argparse.ArgumentParser(description="Visor de etiquetas COCO unificadas en bruto")
    parser.add_argument("--image", type=str, default="", help="Ruta o nombre del archivo de la imagen a buscar")
    args = parser.parse_args()

    # Carpetas de datos locales
    images_dir = "data/coco_unified/images"
    annotations_dir = "data/coco_unified/annotations"
    
    # Mapeo de colores fijos para no confundir
    metadata = MetadataCatalog.get("coco_visor_temp")
    metadata.thing_classes = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]
    metadata.thing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]

    # Unir todas las anotaciones de las particiones (train, test, valid)
    data = {"images": [], "annotations": [], "categories": []}
    for json_name in ["train.json", "valid.json", "test.json"]:
        path = os.path.join(annotations_dir, json_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                d = json.load(f)
                if not data["categories"]:
                    data["categories"] = d.get("categories", [])
                data["images"].extend(d.get("images", []))
                data["annotations"].extend(d.get("annotations", []))

    if not data["images"]:
        print("[!] No se han encontrado archivos JSON de COCO en data/coco_unified/annotations")
        return

    # Buscar la imagen solicitada (o elegir una al azar si no se pasa nada específico)
    target_img = None
    if args.image:
        for img in data["images"]:
            # Buscar coincidencia parcial (por si solo pones el nombre de foto y no la ruta en COCO_unfied)
            if args.image in img["file_name"] or img["file_name"] in args.image:
                target_img = img
                break
        if not target_img:
            print(f"[!] No se encontró ninguna imagen en los JSON que coincida con: {args.image}")
            return
    else:
        # Píllate una que sepas que tiene "tickets", por ejemplo
        imgs_with_tickets = [ann["image_id"] for ann in data["annotations"] if ann["category_id"] == 1]
        if imgs_with_tickets:
            random_id = random.choice(list(set(imgs_with_tickets)))
            target_img = next((i for i in data["images"] if i["id"] == random_id), None)
        else:
            target_img = random.choice(data["images"])

    print(f"[*] Visualizando Anotaciones de Ground Truth (COCO Puro)")
    print(f"[*] Imagen: {target_img['file_name']} (ID: {target_img['id']})")
    
    img_path = os.path.join(images_dir, target_img["file_name"])
    if not os.path.exists(img_path):
         print(f"[!] Ups, el archivo real no existe en disco: {img_path}")
         return

    # =========================================================================
    # CONFIGURACIÓN DEL USUARIO: FILTRO DE ETIQUETAS
    # Comenta o descomenta las clases que quieres que se dibujen en la imagen
    # =========================================================================
    CLASES_A_MOSTRAR = ["ticket"]
    
    # Crear diccionario inverso para buscar ID por nombre
    nombres_por_id = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Extraer las anotaciones de ESA imagen específica filtrando por nombre
    img_anns = []
    for ann in data["annotations"]:
        if ann["image_id"] == target_img["id"]:
            nombre_clase = nombres_por_id.get(ann["category_id"])
            if nombre_clase in CLASES_A_MOSTRAR:
                img_anns.append(ann)
    
    # Detectron2 usa internamente el mapeo continuo de ID's desde 0.
    # COCO original nuestro empieza las category_id en 1, así que restamos 1.
    for ann in img_anns:
        ann["category_id"] = ann["category_id"] - 1
        # Convertimos las 'categories' y 'bbox_mode' (Detectron asume mode=1 que es xywh)
        ann["bbox_mode"] = 1

    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"[!] No se pudo leer la imagen física con OpenCV: {img_path}")
        return

    # Dibuja usando la utilidad de Visión Computacional nativa de M-RCNN
    # Convertir BGR (OpenCV) a RGB (Detectron2 Visualizer)
    v = Visualizer(img_cv[:, :, ::-1], metadata, scale=1.0)
    
    try:
        out = v.draw_dataset_dict({"annotations": img_anns})
    except Exception as e:
        print(f"[!] Error dibujando polígonos/boxes nativas de COCO: {str(e)}")
        # Intento de bypass de fallos en COCO por puntos rotos
        print("[*] Intentando dibujar solo Bounding Boxes (saltando máscaras rotas)...")
        for ann in img_anns:
             ann.pop("segmentation", None) # Elimina máscaras y deja las cajas puras.
        
        v = Visualizer(img_cv[:, :, ::-1], metadata, scale=1.0)
        out = v.draw_dataset_dict({"annotations": img_anns})

    # Convertir RGB de vuelta a BGR para mostrar en OpenCV puro
    result_img = out.get_image()[:, :, ::-1].copy()

    # OpenCV: Reajustar tamaño si es masivo
    h, w = result_img.shape[:2]
    if h > 800:
        scale = 800 / h
        result_img = cv2.resize(result_img, (int(w*scale), int(h*scale)))

    cv2.imshow(f"GT: {target_img['file_name']}", result_img)
    print("\n[+] Mostrando por pantalla. Presiona CUALQUIER TECLA (o 'q') sobre la imagen de OpenCV para cerrarla.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
