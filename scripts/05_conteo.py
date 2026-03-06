"""
04_conteo.py
Módulo para el cruce espacial de tickets y baldas en imágenes frontales,
y su herencia simétrica hacia imágenes traseras en el sistema PoC.H.
"""

def asignar_tickets_a_baldas(detecciones_frontales, total_baldas=3):
    """
    Cruce espacial de tickets y baldas en la imagen frontal.
    
    Args:
        detecciones_frontales (list): Lista de diccionarios de la forma 
            {'class': 'ticket'/'balda', 'bbox': [x1, y1, x2, y2]}
        total_baldas (int): Número teórico esperado de baldas (default 3)
        
    Returns:
        dict: Mapeo {ticket_idx: [lista_de_indices_de_balda]}
    """
    baldas = []
    tickets = []
    
    # 1. Separar detecciones conservando su índice original para identificadores
    for i, det in enumerate(detecciones_frontales):
        clase = det.get('class', '').lower()
        if clase == 'balda':
            baldas.append((i, det))
        elif clase == 'ticket':
            tickets.append((i, det))
            
    # 2. Ordenar baldas por coordenada Y física (de arriba hacia abajo)
    # y1 es la coordenada superior de la bounding box [x1, y1, x2, y2]
    # Índice 0 será la balda más alta (ej. Balda 3), Índice n-1 será la más baja (ej. Balda 1)
    baldas.sort(key=lambda x: x[1]['bbox'][1])
    
    # 3. Asignar cada ticket a la balda en la que recae físicamente
    balda_a_ticket = {}
    
    for t_idx, t_det in tickets:
        t_y1, t_y2 = t_det['bbox'][1], t_det['bbox'][3]
        t_y_center = (t_y1 + t_y2) / 2.0
        
        mejor_balda = -1
        max_overlap = -1
        
        for b_order_idx, (b_idx, b_det) in enumerate(baldas):
            b_y1, b_y2 = b_det['bbox'][1], b_det['bbox'][3]
            
            # Verificación de inclusión del centro del ticket en la balda
            if b_y1 <= t_y_center <= b_y2:
                mejor_balda = b_order_idx
                break
                
            # Fallback a overlap por si el centro queda fuera de las Y pero hay solapamiento
            overlap = max(0, min(t_y2, b_y2) - max(t_y1, b_y1))
            if overlap > max_overlap and overlap > 0:
                max_overlap = overlap
                mejor_balda = b_order_idx
                
        if mejor_balda != -1 and mejor_balda not in balda_a_ticket:
            balda_a_ticket[mejor_balda] = t_idx

    # Si no detectamos tickets emparejados, no podemos asignar nada
    if not balda_a_ticket:
        return {}
        
    # 4. Lógica de dominancia y herencia (Reglas de negocio)
    resultado = {t_idx: [] for t_idx, _ in tickets}
    
    # Encontrar el ticket "top" (situado más arriba) que asume el control inicialmente
    indices_con_ticket = sorted(list(balda_a_ticket.keys()))
    ticket_dominante = balda_a_ticket[indices_con_ticket[0]]
    
    for b_order_idx in range(len(baldas)):
        # Si la balda tiene un ticket propio, este asume la dominancia a partir de ahora (hacia abajo)
        if b_order_idx in balda_a_ticket:
            ticket_dominante = balda_a_ticket[b_order_idx]
            
        # Asignar la balda actual al ticket dominante
        resultado[ticket_dominante].append(b_order_idx)
        
    # Filtrar tickets que no asumieron dominio
    resultado = {k: v for k, v in resultado.items() if v}
    
    return resultado

def procesar_pareja_imagenes(det_frontal, det_trasera):
    """
    Wrapper que demuestra cómo la asignación frontal se aplica al espacio Y 
    de la imagen trasera para futuras detecciones de plantas.
    
    Args:
        det_frontal (list): Detecciones bounding boxes imagen Frontal.
        det_trasera (list): Detecciones bounding boxes imagen Trasera.
        
    Returns:
        dict: Contiene la asignación lógica y las zonas Y útiles en Trasera.
    """
    # 1. Calculamos la asignación de tickets a baldas usando las reglas de la imagen Frontal
    asignacion = asignar_tickets_a_baldas(det_frontal)
    
    # 2. Localizamos las baldas en la Trasera ordenándolas por su Y de arriba a abajo
    baldas_trasera = [det for det in det_trasera if det.get('class', '').lower() == 'balda']
    baldas_trasera.sort(key=lambda x: x['bbox'][1])
    
    # 3. Aplicar al ESPACIO Y de la Trasera para que futuras detecciones sepan a quién pertenecen
    espacio_y_trasera_por_ticket = {}
    
    for ticket_idx, baldas_dominadas in asignacion.items():
        # Tomar baldas válidas para evitar IndexError en caso de fallos de Mask R-CNN
        indices_validos = [i for i in baldas_dominadas if i < len(baldas_trasera)]
        
        if indices_validos:
            # El rango Y dominado va desde el tope superior de su balda más alta 
            # hasta el tope inferior de su balda más baja
            y_min = min(baldas_trasera[i]['bbox'][1] for i in indices_validos)
            y_max = max(baldas_trasera[i]['bbox'][3] for i in indices_validos)
            
            espacio_y_trasera_por_ticket[ticket_idx] = {
                'indices_baldas_heredadas': indices_validos,
                'rango_y_min': y_min,
                'rango_y_max': y_max
            }
            
    return {
        "asignacion_base": asignacion,
        "zonas_evaluacion_trasera": espacio_y_trasera_por_ticket
    }

def contar_articulos(det_frontal, det_trasera, asignacion_base, img_frontal=None, clasificador=None):
    """
    Cuenta los artículos por balda realizando un mapeo en espejo (Flip X) 
    de las detecciones traseras sobre el sistema de coordenadas de las baldas frontales.
    
    Args:
        det_frontal: Detecciones frontales.
        det_trasera: Detecciones traseras.
        asignacion_base: Mapeo ticket -> baldas.
        img_frontal: (Opcional) Imagen frontal BGR para hacer crops y clasificar.
        clasificador: (Opcional) Tupla (model, transform, class_names, device) del ConvNeXt.
    """
    import math
    import cv2

    # Separar y ordenar baldas de arriba (Y menor) hacia abajo (Y mayor)
    baldas_f = [d for d in det_frontal if d.get('class', '').lower() == 'balda']
    baldas_f.sort(key=lambda x: x['bbox'][1])
    
    baldas_b = [d for d in det_trasera if d.get('class', '').lower() == 'balda']
    baldas_b.sort(key=lambda x: x['bbox'][1])

    # Utilidad para encontrar en qué balda cae el centro de una bbox dada
    def ubicar_en_balda(bbox, lista_baldas):
        y1, y2 = bbox[1], bbox[3]
        y_center = (y1 + y2) / 2.0
        mejor_balda, max_overlap = -1, -1
        
        for b_idx, b_det in enumerate(lista_baldas):
            b_y1, b_y2 = b_det['bbox'][1], b_det['bbox'][3]
            if b_y1 <= y_center <= b_y2:
                return b_idx
            # Fallback por solapamiento
            overlap = max(0, min(y2, b_y2) - max(y1, b_y1))
            if overlap > max_overlap and overlap > 0:
                max_overlap = overlap
                mejor_balda = b_idx
        return mejor_balda

    # Variables para agrupar las masas verdes (Flores y Plantas) y los Tallos por cada balda
    # Estructura: dict[b_idx] = lista_de_detecciones
    masas_por_balda = {i: [] for i in range(len(baldas_f))}
    tallos_por_balda = {i: [] for i in range(len(baldas_b))}

    # Llenar las dicciones basándonos en la vista de cada cámara
    for det in det_frontal:
        clase = det.get('class', '').lower()
        if clase in ['flores', 'planta']:
            b_idx = ubicar_en_balda(det['bbox'], baldas_f)
            if b_idx != -1:
                det['tallos_asociados'] = 0 # Inicializamos contador
                masas_por_balda[b_idx].append(det)

    for det in det_trasera:
        clase = det.get('class', '').lower()
        if clase in ['tallo_grupo']:
            b_idx = ubicar_en_balda(det['bbox'], baldas_b)
            if b_idx != -1:
                tallos_por_balda[b_idx].append(det)

    # =========================================================================
    # LÓGICA CORE: MAPEO EN ESPEJO (TRASERO -> FRONTAL)
    # -------------------------------------------------------------------------
    for b_idx in range(len(baldas_f)):
        # Si falta la balda opuesta, nos saltamos el mapeo geométrico fino
        if b_idx >= len(baldas_b):
            continue
            
        b_f = baldas_f[b_idx]['bbox']  # [x1, y1, x2, y2]
        b_b = baldas_b[b_idx]['bbox']
        
        ancho_f = b_f[2] - b_f[0]
        ancho_b = b_b[2] - b_b[0]
        
        lista_masas = masas_por_balda[b_idx]
        lista_tallos = tallos_por_balda[b_idx].copy()
        
        # 1. Anotamos la proyección (cx, cy) frontal dentro de cara tallo
        for tallo in lista_tallos:
            t_box = tallo['bbox']
            cx_tallo_b = (t_box[0] + t_box[2]) / 2.0
            
            porcentaje_x_invertido = 1.0 - ((cx_tallo_b - b_b[0]) / ancho_b)
            tallo['cx_proyectado_f'] = b_f[0] + (porcentaje_x_invertido * ancho_f)
            tallo['cy_proyectado_f'] = (t_box[1] + t_box[3]) / 2.0

        # Algoritmo de Reparto Justo en Dos Fases:
        import math
        
        # FASE 1: Garantizar que ninguna flor/planta se quede con 0 tallos, 
        # asignando el tallo MÁS CERCANO posible a cada flor vacía.
        while lista_tallos and any(m['tallos_asociados'] == 0 for m in lista_masas):
            mejor_distancia = float('inf')
            mejor_par = None # (masa_index, tallo)
            
            for m_idx, masa in enumerate(lista_masas):
                if masa['tallos_asociados'] > 0:
                    continue # Esta flor ya tiene su tallo
                    
                m_box = masa['bbox']
                cx_masa = (m_box[0] + m_box[2]) / 2.0
                cy_masa = (m_box[1] + m_box[3]) / 2.0
                
                for tallo in lista_tallos:
                    dist = math.hypot(tallo['cx_proyectado_f'] - cx_masa, tallo['cy_proyectado_f'] - cy_masa)
                    if dist < mejor_distancia:
                        mejor_distancia = dist
                        mejor_par = (masa, tallo)
                        
            if mejor_par is not None:
                # Damos el tallo a la flor/planta y quitamos el tallo del pool disponible
                mejor_par[0]['tallos_asociados'] += 1
                lista_tallos.remove(mejor_par[1])
                
        # FASE 2: Los tallos que sobren (porque había más tallos que flores en la balda)
        # se asignan ahora sí libremente a la flor/planta más cercana sin restricciones, engordando sus números.
        for tallo in lista_tallos:
            mejor_masa = None
            distancia_min = float('inf')
            
            for masa in lista_masas:
                m_box = masa['bbox']
                cx_masa = (m_box[0] + m_box[2]) / 2.0
                cy_masa = (m_box[1] + m_box[3]) / 2.0
                dist = math.hypot(tallo['cx_proyectado_f'] - cx_masa, tallo['cy_proyectado_f'] - cy_masa)
                
                if dist < distancia_min:
                    distancia_min = dist
                    mejor_masa = masa
                    
            if mejor_masa is not None:
                mejor_masa['tallos_asociados'] += 1
                
    # =========================================================================
    
    # --- DIAGNÓSTICO: mostrar qué se encontró en cada balda ---
    print("\n[DEBUG] Resumen de detecciones por balda frontal:")
    baldas_cubiertas = set()
    for t_idx, b_indices in asignacion_base.items():
        for b_idx in b_indices:
            baldas_cubiertas.add(b_idx)
    
    for b_idx in range(len(baldas_f)):
        n_masas = len(masas_por_balda[b_idx])
        en_ticket = "✓" if b_idx in baldas_cubiertas else "✗ SIN TICKET"
        items_str = ", ".join([f"{m['class']}" for m in masas_por_balda[b_idx]])
        print(f"  Balda idx={b_idx} (Y: {baldas_f[b_idx]['bbox'][1]:.0f}-{baldas_f[b_idx]['bbox'][3]:.0f}): "
              f"{n_masas} items [{items_str}] {en_ticket}")
    
    # Items que no cayeron en ninguna balda
    total_flores_planta = sum(1 for d in det_frontal if d.get('class','').lower() in ['flores','planta'])
    total_asignadas = sum(len(v) for v in masas_por_balda.values())
    if total_asignadas < total_flores_planta:
        print(f"  [!] {total_flores_planta - total_asignadas} flores/plantas no cayeron en ninguna balda")
    print()
    
    # Preparar el JSON final estructurado por Tickets y sus Baldas dominadas
    t_idx_order = sorted(asignacion_base.keys(), key=lambda idx: det_frontal[idx]['bbox'][1], reverse=True)
    ticket_mapping = {t_idx: i + 1 for i, t_idx in enumerate(t_idx_order)}
    total_baldas = len(baldas_f)
    
    resultado_json = {}
    
    for t_idx, b_indices in asignacion_base.items():
        ticket_key = f"Ticket_{ticket_mapping[t_idx]}"
        resultado_json[ticket_key] = {}
        
        # Ordenamos las baldas de abajo a arriba para la salida visual
        b_indices_sorted = sorted(b_indices, reverse=True)
        
        for b_idx in b_indices_sorted:
            balda_key = f"Balda_{total_baldas - b_idx}"
            
            # Recolectar datos individuales de cada masa en esta balda
            items_raw = []  # Lista de (producto_id, tipo, tallos, unidades, confianza)
            contador_indices = {'Flores': 1, 'Planta': 1}
            
            for masa in masas_por_balda[b_idx]:
                clase_str = masa['class'].capitalize() # "Flores" o "Planta"
                
                tallos = masa['tallos_asociados']
                unidades_finales = max(1, tallos)
                
                # --- Clasificación por especie con ConvNeXt ---
                producto_id = None
                confianza = None
                if img_frontal is not None and clasificador is not None:
                    cnx_model, cnx_transform, cnx_classes, cnx_device = clasificador
                    bbox = masa['bbox']  # [x1, y1, x2, y2]
                    x1 = max(0, int(bbox[0]))
                    y1 = max(0, int(bbox[1]))
                    x2 = min(img_frontal.shape[1], int(bbox[2]))
                    y2 = min(img_frontal.shape[0], int(bbox[3]))
                    
                    crop = img_frontal[y1:y2, x1:x2]
                    if crop.size > 0:
                        from PIL import Image as PILImage
                        import torch as _torch
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        pil_crop = PILImage.fromarray(crop_rgb)
                        input_t = cnx_transform(pil_crop).unsqueeze(0).to(cnx_device)
                        
                        with _torch.no_grad():
                            output = cnx_model(input_t)
                            probs = _torch.softmax(output, dim=1)[0]
                            top_prob, top_idx = probs.max(0)
                        
                        producto_id = cnx_classes[top_idx.item()]
                        confianza = top_prob.item()
                
                # Si no hay ConvNeXt, usamos Flores_1/Planta_1 como ID
                if producto_id is None:
                    producto_id = f"{clase_str}_{contador_indices[clase_str]}"
                    contador_indices[clase_str] += 1
                
                items_raw.append((producto_id, clase_str.lower(), tallos, unidades_finales, confianza))
            
            # Agrupar por producto_id
            items_en_esta_balda = {}
            for prod_id, tipo, tallos, unidades, conf in items_raw:
                if prod_id not in items_en_esta_balda:
                    items_en_esta_balda[prod_id] = {
                        "tipo": tipo,
                        "detecciones": 0,
                        "tallos_totales": 0,
                        "unidades_totales": 0,
                        "_confianzas": []  # temporal, se promediará después
                    }
                items_en_esta_balda[prod_id]["detecciones"] += 1
                items_en_esta_balda[prod_id]["tallos_totales"] += tallos
                items_en_esta_balda[prod_id]["unidades_totales"] += unidades
                if conf is not None:
                    items_en_esta_balda[prod_id]["_confianzas"].append(conf)
            
            # Calcular confianza media y limpiar campo temporal
            for prod_id, data in items_en_esta_balda.items():
                confs = data.pop("_confianzas")
                if confs:
                    data["confianza_media"] = round(sum(confs) / len(confs), 3)
            
            resultado_json[ticket_key][balda_key] = items_en_esta_balda
            
    return resultado_json, ticket_mapping

if __name__ == "__main__":
    import os
    import sys
    import cv2
    import yaml
    import torch

    # Asegurar que configs/ está en el path para importar config_manager
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "configs"))
    os.chdir(PROJECT_ROOT)

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    import timm
    from torchvision import transforms

    # ======================================================================
    # CONFIGURACIÓN DEL USUARIO — MODIFICAR AQUÍ
    # ======================================================================
    
    # --- Mask R-CNN (segmentación) ---
    # Ruta al .pth del modelo Mask R-CNN a usar
    MRCNN_MODEL_PATH = os.path.join("models", "maskrcnn", "big_aug_anchors_v4", "model_final.pth")
    # Umbral de confianza mínimo para las detecciones
    SCORE_THRESH = 0.10
    
    # --- ConvNeXt (clasificación por especie) ---
    # Carpeta del run de ConvNeXt (debe contener class_names.txt y config_used.yaml)
    CONVNEXT_RUN_DIR = os.path.join("models", "convnext", "v1_baseline")
    # Checkpoint a usar (best_model.pth o model_final.pth)
    CONVNEXT_MODEL_PATH = os.path.join(CONVNEXT_RUN_DIR, "best_model.pth")
    
    # --- Imágenes de prueba ---
    IMAGE_FRONTAL_PATH = os.path.join("data", "dataset_final", "15F.png")
    IMAGE_TRASERA_PATH = os.path.join("data", "dataset_final", "15B.png")
    # ======================================================================

    print("[*] Iniciando prueba con Detectron2 + ConvNeXt...")
    
    # 1. Configurar y Cargar Mask R-CNN
    #    Se aplica config1.yaml para que las anchors, RPN, ROI, etc.
    #    coincidan con las del modelo entrenado.
    from config_manager import parse_yaml_config, apply_custom_config_to_cfg
    custom_cfg = parse_yaml_config()
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
    cfg.MODEL.WEIGHTS = MRCNN_MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH 
    
    try:
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print(f"[-] Error al cargar Mask R-CNN: {e}")
        exit()

    # 2. Cargar ConvNeXt para clasificación por especie
    clasificador = None
    if os.path.exists(CONVNEXT_MODEL_PATH):
        print("[*] Cargando clasificador ConvNeXt...")
        cnx_config_path = os.path.join(CONVNEXT_RUN_DIR, "config_used.yaml")
        cnx_class_names_path = os.path.join(CONVNEXT_RUN_DIR, "class_names.txt")
        
        # Leer config
        cnx_cfg = {}
        if os.path.exists(cnx_config_path):
            with open(cnx_config_path) as f:
                cnx_cfg = yaml.safe_load(f)
        
        # Leer nombres de clases
        cnx_class_names = []
        if os.path.exists(cnx_class_names_path):
            with open(cnx_class_names_path) as f:
                for line in f:
                    parts = line.strip().split(": ", 1)
                    if len(parts) == 2:
                        cnx_class_names.append(parts[1])
        
        cnx_num_classes = len(cnx_class_names)
        cnx_model_name = cnx_cfg.get("MODEL", {}).get("NAME", "convnext_tiny.in12k_ft_in1k")
        cnx_img_size = cnx_cfg.get("DATA", {}).get("IMG_SIZE", 224)
        cnx_mean = cnx_cfg.get("DATA", {}).get("MEAN", [0.485, 0.456, 0.406])
        cnx_std = cnx_cfg.get("DATA", {}).get("STD", [0.229, 0.224, 0.225])
        
        cnx_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnx_model = timm.create_model(cnx_model_name, pretrained=False, num_classes=cnx_num_classes)
        cnx_model.load_state_dict(torch.load(CONVNEXT_MODEL_PATH, map_location=cnx_device, weights_only=True))
        cnx_model = cnx_model.to(cnx_device)
        cnx_model.eval()
        
        cnx_transform = transforms.Compose([
            transforms.Resize((cnx_img_size, cnx_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cnx_mean, std=cnx_std),
        ])
        
        clasificador = (cnx_model, cnx_transform, cnx_class_names, cnx_device)
        print(f"    Clases de producto ({cnx_num_classes}): {cnx_class_names}")
    else:
        print(f"[!] No se encontró ConvNeXt en {CONVNEXT_MODEL_PATH}. Se omite clasificación por especie.")

    # 3. Clases Mask R-CNN (según el script 01_fix_coco.py)
    # 0: Flores, 1: ticket, 2: Balda, 3: Planta, 4: tallo_grupo
    class_names = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]
    
    MetadataCatalog.get("temp_test_dataset").set(thing_classes=class_names)
    metadata = MetadataCatalog.get("temp_test_dataset")

    # 4. Cargar imágenes de prueba
    img_f = cv2.imread(IMAGE_FRONTAL_PATH)
    img_b = cv2.imread(IMAGE_TRASERA_PATH)

    if img_f is None or img_b is None:
        print(f"[-] No se pudieron cargar las imágenes.")
        print(f"    Frontal: {IMAGE_FRONTAL_PATH}")
        print(f"    Trasera: {IMAGE_TRASERA_PATH}")
        exit()

    # 5. Inferencia Mask R-CNN
    print("[*] Ejecutando inferencia en Frontal...")
    outputs_f = predictor(img_f)
    print("[*] Ejecutando inferencia en Trasera...")
    outputs_b = predictor(img_b)

    def extract_detections(outputs):
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        
        detections = []
        for i in range(len(boxes)):
            class_name = class_names[classes[i]]
            detections.append({
                'class': class_name,
                'bbox': boxes[i].tolist(),
            })
        return detections

    det_f = extract_detections(outputs_f)
    det_b = extract_detections(outputs_b)

    # 6. Cruce espacial
    resultado = procesar_pareja_imagenes(det_f, det_b)

    # 7. Conteo por zonas + clasificación por especie
    conteo_final, ticket_mapping = contar_articulos(
        det_f, det_b, resultado['asignacion_base'],
        img_frontal=img_f, clasificador=clasificador
    )

    print("\n" + "="*50)
    print(" RESULTADOS DEL CRUCE ESPACIAL Y CONTEO")
    print("="*50)
    print("Mapeo Lógico en Frontal (Ticket ID -> Índices Baldas dominadas [0=Top]):")
    print(resultado['asignacion_base'])
    print("\nZonas Físicas delimitadas en Trasera (Heredadas):")
    for t_idx, data in resultado['zonas_evaluacion_trasera'].items():
        print(f"  Ticket_{ticket_mapping[t_idx]} -> Domina desde Y_min: {data['rango_y_min']:.1f} hasta Y_max: {data['rango_y_max']:.1f}")

    print("\n" + "-"*50)
    print(" JSON RESULTANTE (Artículos por Balda y Ticket) ")
    print("-"*50)
    import json
    # Empaquetamos en 'Items' como pidió el usuario
    json_output = {"Items": conteo_final}
    print(json.dumps(json_output, indent=4, ensure_ascii=False))
    print("-"*50 + "\n")

    # 7. Visualización OpenCV
    print("\n[*] Generando visualizaciones...")
    v_f = Visualizer(img_f[:, :, ::-1], metadata=metadata, scale=1.0)
    out_f = v_f.draw_instance_predictions(outputs_f["instances"].to("cpu"))
    
    v_b = Visualizer(img_b[:, :, ::-1], metadata=metadata, scale=1.0)
    out_b = v_b.draw_instance_predictions(outputs_b["instances"].to("cpu"))

    # Mostrar en pantalla mediante OpenCV
    img_show_f = out_f.get_image()[:, :, ::-1].copy()
    img_show_b = out_b.get_image()[:, :, ::-1].copy()

    # --- DIBUJO DE REGLAS LÓGICAS EN FRONT ---
    baldas_frontales = [d for d in det_f if d.get('class', '').lower() == 'balda']
    baldas_frontales.sort(key=lambda x: x['bbox'][1])

    for t_idx, b_indices in resultado['asignacion_base'].items():
        t_box = det_f[t_idx]['bbox']
        color = (0, 255, 0) # Verde intenso (BGR en OpenCV puro)
        ticket_num = ticket_mapping[t_idx]
        
        # Bounding box extra gruesa para resaltar el Ticket dominador
        cv2.rectangle(img_show_f, (int(t_box[0]), int(t_box[1])), (int(t_box[2]), int(t_box[3])), color, 4)
        cv2.putText(img_show_f, f"Ticket_{ticket_num} Asist.", (int(t_box[0]), int(t_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        t_cx = int((t_box[0] + t_box[2]) / 2)
        t_cy = int((t_box[1] + t_box[3]) / 2)
        
        for b_i in b_indices:
            if b_i < len(baldas_frontales):
                b_box = baldas_frontales[b_i]['bbox']
                b_cx = int((b_box[0] + b_box[2]) / 2)
                b_cy = int((b_box[1] + b_box[3]) / 2)
                # Línea que une Ticket con sus Baldas dominadas
                cv2.line(img_show_f, (t_cx, t_cy), (b_cx, b_cy), color, 3)

    # --- DIBUJO DE ZONAS ESPACIALES EN BACK ---
    h_img, w_img = img_show_b.shape[:2]
    overlay_b = img_show_b.copy()

    for idx, (t_idx, data) in enumerate(resultado['zonas_evaluacion_trasera'].items()):
        # Colores alternos (Cian, Amarillo, Magenta)
        colores = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
        color = colores[idx % len(colores)] 
        ticket_num = ticket_mapping[t_idx]
        
        y_min = int(data['rango_y_min'])
        y_max = int(data['rango_y_max'])
        
        # Rectángulo semi-transparente abarcando el ancho
        cv2.rectangle(overlay_b, (10, y_min), (w_img-10, y_max), color, -1)
        
        # Bordes
        cv2.rectangle(img_show_b, (10, y_min), (w_img-10, y_max), color, 4)
        cv2.putText(img_show_b, f"Zona Plantas T_{ticket_num}", (20, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(img_show_b, f"Zona Plantas T_{ticket_num}", (20, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.addWeighted(overlay_b, 0.3, img_show_b, 0.7, 0, img_show_b)
    
    print("[*] Mostrando ventanas. Presiona 'q' o culquier tecla en la ventana para salir.")
    # Redimensionar para que quepan en pantalla si son muy grandes
    h, w = img_show_f.shape[:2]
    if h > 800:
        scale = 800 / h
        img_show_f = cv2.resize(img_show_f, (int(w*scale), int(h*scale)))
        img_show_b = cv2.resize(img_show_b, (int(w*scale), int(h*scale)))

    cv2.imshow("Frontal - Deteccion de Mask R-CNN", img_show_f)
    cv2.imshow("Trasera - Deteccion de Mask R-CNN", img_show_b)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
