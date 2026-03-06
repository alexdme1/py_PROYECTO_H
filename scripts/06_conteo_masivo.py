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
    Cuenta los artículos por balda realizando un mapeo en espejo (Flip X) 
    de las detecciones traseras sobre el sistema de coordenadas de las baldas frontales.
    """
    import math

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
            
            # Recolectar lo que hay en esta balda
            items_en_esta_balda = {}
            contador_indices = {'Flores': 1, 'Planta': 1}
            
            for masa in masas_por_balda[b_idx]:
                clase_str = masa['class'].capitalize() # "Flores" o "Planta"
                nombre_unico = f"{clase_str}_{contador_indices[clase_str]}"
                contador_indices[clase_str] += 1
                
                tallos = masa['tallos_asociados']
                
                # Regla de Negocio: 
                # Tratándose de flores, si se ven desde delante como 1 bulto, pero hay 3 tallos detrás: Unidades = 3.
                # Si una Planta se ve como 1 bulto por delante y 1 tallo por detrás: Unidades = 1.
                # En caso general, nos fiamos del máximo entre cajas frontales (1) y grupos de tallos traseros (N).
                unidades_finales = max(1, tallos)
                
                items_en_esta_balda[nombre_unico] = {
                    "Deteccion_Frontal_Volumen": 1,
                    "Tallos_Grupos_Asociados": tallos,
                    "Conteo_Final_Unidades": unidades_finales
                }
            
            resultado_json[ticket_key][balda_key] = items_en_esta_balda
            
    return resultado_json, ticket_mapping

if __name__ == "__main__":
    import os
    import cv2
    import json
    import glob
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    # ======================================================================
    # CONFIGURACIÓN DEL USUARIO
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
    
    DATASET_DIR = os.path.join("data", "dataset_final")
    JSON_OUTPUT_PATH = "final_global_conteo.json"
    # ======================================================================

    print("[*] Iniciando sistema de conteo global Masivo...")
    
    # 1. Configurar y Cargar el modelo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    
    from config_manager import apply_custom_config_to_cfg
    cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
    
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH 
    
    try:
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print(f"[-] Error al cargar el modelo: {e}")
        exit()

    class_names = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]

    # Diccionario Global para guardar todo
    db_global = {}

    # Buscar todos los archivos "F" (Frontales) para emparejarlos
    frontales = glob.glob(os.path.join(DATASET_DIR, "*F.png")) + glob.glob(os.path.join(DATASET_DIR, "*F.jpg"))
    frontales.sort()
    
    print(f"[*] Se han encontrado {len(frontales)} imágenes frontales para procesar.")

    def extract_detections(outputs):
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        detections = []
        for i in range(len(boxes)):
            class_name = class_names[classes[i]]
            detections.append({'class': class_name, 'bbox': boxes[i].tolist()})
        return detections

    for img_f_path in frontales:
        # Extraer el ID. Ej: "99F.png" -> "99"
        filename = os.path.basename(img_f_path)
        img_id = filename.replace('F.png', '').replace('F.jpg', '')
        
        # Asumir que existe su gemelo trasero
        img_b_path_png = os.path.join(DATASET_DIR, f"{img_id}B.png")
        img_b_path_jpg = os.path.join(DATASET_DIR, f"{img_id}B.jpg")
        
        if os.path.exists(img_b_path_png):
            img_b_path = img_b_path_png
        elif os.path.exists(img_b_path_jpg):
            img_b_path = img_b_path_jpg
        else:
            print(f"[-] ADVERTENCIA: No se encontró la pareja trasera para {filename}. Se omite.")
            continue
            
        print(f"\n[*] Procesando carro ID: {img_id} ...")
        
        img_f = cv2.imread(img_f_path)
        img_b = cv2.imread(img_b_path)
        
        if img_f is None or img_b is None:
            print(f"[-] Error leyendo {img_id}. Se omite.")
            continue
            
        # Inferencia
        outputs_f = predictor(img_f)
        outputs_b = predictor(img_b)
        
        det_f = extract_detections(outputs_f)
        det_b = extract_detections(outputs_b)
        
        # Mapeo y cruce cruzado
        resultado_asignacion = procesar_pareja_imagenes(det_f, det_b)
        conteo_final, _ = contar_articulos(det_f, det_b, resultado_asignacion['asignacion_base'])
        
        # Inyectar al JSON Global
        db_global[img_id] = conteo_final

    # Guardar a disco
    print("\n" + "="*50)
    print(" EXPORTANDO BASE DE DATOS GLOBAL ")
    print("="*50)
    
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(db_global, f, indent=4, ensure_ascii=False)
        
    print(f"[*] ¡Proceso Completado! Todos los datos procesados se han volcado en: {JSON_OUTPUT_PATH}")
