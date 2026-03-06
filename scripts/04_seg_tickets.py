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

def contar_articulos(det_frontal, det_trasera, asignacion_base):
    """
    Cuenta los artículos por balda en frontal y trasera, agrupándolos
    bajo el ticket que domina dicha balda.
    """
    baldas_f = [d for d in det_frontal if d.get('class', '').lower() == 'balda']
    baldas_f.sort(key=lambda x: x['bbox'][1])
    
    baldas_b = [d for d in det_trasera if d.get('class', '').lower() == 'balda']
    baldas_b.sort(key=lambda x: x['bbox'][1])

    def ubicar_en_balda(bbox, lista_baldas):
        y1, y2 = bbox[1], bbox[3]
        y_center = (y1 + y2) / 2.0
        
        mejor_balda = -1
        max_overlap = -1
        
        for b_idx, b_det in enumerate(lista_baldas):
            b_y1, b_y2 = b_det['bbox'][1], b_det['bbox'][3]
            if b_y1 <= y_center <= b_y2:
                return b_idx
            
            overlap = max(0, min(y2, b_y2) - max(y1, b_y1))
            if overlap > max_overlap and overlap > 0:
                max_overlap = overlap
                mejor_balda = b_idx
                
        return mejor_balda

    def inicializar_conteo():
        return {'Flores': 0, 'Tallos': 0, 'Plantas': 0}
        
    conteo_f = {i: inicializar_conteo() for i in range(len(baldas_f))}
    conteo_b = {i: inicializar_conteo() for i in range(len(baldas_b))}

    clases_interes = {
        'flores': 'Flores',
        'tallo_grupo': 'Tallos',
        'planta': 'Plantas'
    }

    # Asignar items frontales a baldas
    for det in det_frontal:
        clase = det.get('class', '').lower()
        if clase in clases_interes:
            b_idx = ubicar_en_balda(det['bbox'], baldas_f)
            if b_idx != -1:
                conteo_f[b_idx][clases_interes[clase]] += 1

    # Asignar items traseros a baldas
    for det in det_trasera:
        clase = det.get('class', '').lower()
        if clase in clases_interes:
            b_idx = ubicar_en_balda(det['bbox'], baldas_b)
            if b_idx != -1:
                conteo_b[b_idx][clases_interes[clase]] += 1

    # Ordenar tickets de abajo a arriba (Y mayor a Y menor)
    # det_frontal[t_idx]['bbox'][1] es la coord Y superior de la caja del ticket
    t_idx_order = sorted(asignacion_base.keys(), key=lambda idx: det_frontal[idx]['bbox'][1], reverse=True)
    ticket_mapping = {t_idx: i + 1 for i, t_idx in enumerate(t_idx_order)}
    
    total_baldas = len(baldas_f)

    # Construir JSON final usando la asignación dictada por asignacion_base
    resultado_json = {}
    for t_idx, b_indices in asignacion_base.items():
        ticket_key = f"Ticket_{ticket_mapping[t_idx]}"
        resultado_json[ticket_key] = {}
        
        # Opcional: ordenar b_indices de abajo a arriba (mayor indice interno a menor, 
        # ya que 0 es Top y N-1 es Bottom en la lista asignada)
        b_indices_sorted = sorted(b_indices, reverse=True)
        
        for b_idx in b_indices_sorted:
            # Baldas: 0 es la más alta. Para numerar de abajo hacia arriba:
            # si total_baldas=3, 0->3, 1->2, 2->1
            # Así, balda_idx_visual = total_baldas - b_idx
            balda_key = f"Balda_{total_baldas - b_idx}"
            
            counts_f = conteo_f.get(b_idx, inicializar_conteo())
            counts_b = conteo_b.get(b_idx, inicializar_conteo())
            
            resultado_json[ticket_key][balda_key] = {
                "Delantera": counts_f,
                "Trasera": counts_b
            }
            
    return resultado_json, ticket_mapping

if __name__ == "__main__":
    import os
    import cv2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer

    # ======================================================================
    # CONFIGURACIÓN DEL USUARIO (Rutas de modelo e imágenes)
    # ======================================================================
    from config_manager import parse_yaml_config
    custom_cfg = parse_yaml_config()
    suj = custom_cfg.get("MODEL_INFO", {}).get("SUFIJO_VERSION", "")
    out_dir = custom_cfg.get("MODEL_INFO", {}).get("OUTPUT_DIR_BASE", "models/maskrcnn")
    
    name_file = os.path.join("data", "coco_unified", "model_name.txt")
    model_name = "default"
    if os.path.exists(name_file):
        with open(name_file, "r") as f:
            model_name = f.read().strip()
            
    run_name = f"{model_name}{suj}"
    MODEL_WEIGHTS_PATH = os.path.join(out_dir, run_name, "model_final.pth")
    SCORE_THRESH = custom_cfg.get("MODEL", {}).get("ROI_HEADS", {}).get("SCORE_THRESH_TEST", 0.10)
    
    IMAGE_FRONTAL_PATH = os.path.join("data", "dataset_final", "99F.png")
    IMAGE_TRASERA_PATH = os.path.join("data", "dataset_final", "99B.png")
    # ======================================================================

    print("[*] Iniciando prueba con Detectron2...")
    
    # 1. Configurar y Cargar el modelo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Asegúrate de usar NUM_CLASSES correcto (5)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    
    from config_manager import parse_yaml_config, apply_custom_config_to_cfg
    custom_cfg = parse_yaml_config()
    cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
    
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH 
    
    try:
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print(f"[-] Error al cargar el modelo: {e}")
        exit()

    # 2. Clases (según el script 01_fix_coco.py)
    # 0: Flores, 1: ticket, 2: Balda, 3: Planta, 4: tallo_grupo
    class_names = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]
    
    # Registrar metadata temporal para el visualizador
    MetadataCatalog.get("temp_test_dataset").set(thing_classes=class_names)
    metadata = MetadataCatalog.get("temp_test_dataset")

    # 3. Cargar imágenes de prueba real
    img_f = cv2.imread(IMAGE_FRONTAL_PATH)
    img_b = cv2.imread(IMAGE_TRASERA_PATH)

    if img_f is None or img_b is None:
        print(f"[-] No se pudieron cargar las imágenes.")
        print(f"    Frontal: {IMAGE_FRONTAL_PATH}")
        print(f"    Trasera: {IMAGE_TRASERA_PATH}")
        exit()

    # 4. Inferencia
    print("[*] Ejecutando inferencia en Frontal...")
    outputs_f = predictor(img_f)
    print("[*] Ejecutando inferencia en Trasera...")
    outputs_b = predictor(img_b)

    # Función auxiliar para extraer detecciones
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

    # 5. Lógica de cruce espacial
    resultado = procesar_pareja_imagenes(det_f, det_b)

    # 6. Conteo por zonas (nuevo requerimiento JSON)
    conteo_final, ticket_mapping = contar_articulos(det_f, det_b, resultado['asignacion_base'])

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
