"""
04_build_tree_features.py
Construcción de features por detección para el árbol de conteo.
Genera data/arbol_conteo/detections_raw.csv a partir de data/dataset_final.
Ver documentation/ARBOL_CONTEO.md para especificación completa.

Uso:
    /home/servi2/Enviroments/main_venv/bin/python3 scripts/01-preprocesing/04_build_tree_features.py
"""

import os
import sys
import csv
import math
from dataclasses import dataclass, fields, asdict
from typing import List, Optional

# ---------------------------------------------------------------------------
# Asegurar que conteo_lib es importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONTEO_DIR = os.path.join(_PROJECT_ROOT, "scripts", "05-logica_conteo_tallos")
if _CONTEO_DIR not in sys.path:
    sys.path.insert(0, _CONTEO_DIR)

from conteo_lib import ubicar_en_balda, procesar_pareja_imagenes


# =============================================================================
# Dataclass DetectionFeat (sección 8 del doc)
# =============================================================================

@dataclass
class DetectionFeat:
    # Identificación
    carro_id: str = ""
    image_pair_id: str = ""
    detection_id: int = 0
    ticket_idx: int = -1
    balda_idx: int = -1
    # Atributos propios
    tipo_d: str = ""           # flor / planta / tallo
    sku_d: str = ""
    lado_d: str = ""           # F / B
    pos_rel_d: float = 0.0
    volumen_d: float = 0.0
    score_mrcnn_d: float = 0.0
    score_convnext_d: float = 0.0
    es_superpuesto_d: int = 0
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    # Vecino derecha
    vec_der_existe_d: int = 0
    vec_der_tipo: str = "none"
    vec_der_sku: str = ""
    vec_der_pos_rel: float = -1.0
    vec_der_volumen: float = 0.0
    vec_der_score_mrcnn: float = 0.0
    vec_der_score_convnext: float = 0.0
    vec_der_superpuesto: int = 0
    # Vecino izquierda
    vec_izq_existe_d: int = 0
    vec_izq_tipo: str = "none"
    vec_izq_sku: str = ""
    vec_izq_pos_rel: float = -1.0
    vec_izq_volumen: float = 0.0
    vec_izq_score_mrcnn: float = 0.0
    vec_izq_score_convnext: float = 0.0
    vec_izq_superpuesto: int = 0
    # Corresponsal otra vista
    otro_lado_existe_d: int = 0
    otro_lado_tipo: str = "none"
    otro_lado_sku: str = ""
    otro_lado_pos_rel: float = -1.0
    otro_lado_volumen: float = 0.0
    otro_lado_score_mrcnn: float = 0.0
    otro_lado_score_convnext: float = 0.0
    otro_lado_dist_pos_rel: float = 0.0
    otro_lado_superpuesto: int = 0
    otro_lado_superpuesto2: int = 0
    # Vecinos del corresponsal
    otro_lado_vec_der_existe: int = 0
    otro_lado_vec_der_tipo: str = "none"
    otro_lado_vec_der_pos_rel: float = -1.0
    otro_lado_vec_der_volumen: float = 0.0
    otro_lado_vec_der_score_mrcnn: float = 0.0
    otro_lado_vec_der_score_convnext: float = 0.0
    otro_lado_vec_izq_existe: int = 0
    otro_lado_vec_izq_tipo: str = "none"
    otro_lado_vec_izq_pos_rel: float = -1.0
    otro_lado_vec_izq_volumen: float = 0.0
    otro_lado_vec_izq_score_mrcnn: float = 0.0
    otro_lado_vec_izq_score_convnext: float = 0.0
    # Agregados de contexto
    num_mismo_tipo_misma_balda_misma_vista: int = 0
    num_mismo_tipo_misma_balda_otro_lado: int = 0
    num_tallos_misma_balda: int = 0
    # Labels
    unidades_label_d: int = -1
    unidades_pred_regla: int = 0


def detection_to_dict(d: DetectionFeat) -> dict:
    return asdict(d)

def get_csv_columns() -> List[str]:
    return [f.name for f in fields(DetectionFeat)]


# =============================================================================
# Utilidades internas
# =============================================================================

def _normalizar_tipo(class_name: str) -> str:
    cl = class_name.lower()
    if cl == 'flores': return 'flor'
    elif cl == 'planta': return 'planta'
    elif cl == 'tallo_grupo': return 'tallo'
    return cl

def _bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def _bbox_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    a1, a2 = _bbox_area(b1), _bbox_area(b2)
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0

def _pos_rel_en_balda(bbox, balda_bbox):
    cx = (bbox[0] + bbox[2]) / 2.0
    bw = balda_bbox[2] - balda_bbox[0]
    if bw <= 0: return 0.5
    return max(0.0, min(1.0, (cx - balda_bbox[0]) / bw))

def _mask_area(det):
    if 'mask_area' in det: return float(det['mask_area'])
    return _bbox_area(det['bbox'])


# =============================================================================
# Función 1: construir_detecciones_enriquecidas
# =============================================================================

def construir_detecciones_enriquecidas(
    det_front, det_back, asign_tickets, baldas_f, baldas_b,
    carro_id="poc", image_pair_id="", clasificador=None,
    img_frontal=None, img_trasera=None,
) -> List[DetectionFeat]:
    balda_to_ticket = {}
    for t_idx, b_indices in asign_tickets.items():
        for b_idx in b_indices:
            balda_to_ticket[b_idx] = t_idx

    dets = []
    det_id = 0

    def _classify_crop(det, img_src):
        if clasificador is None or img_src is None: return "", 0.0
        import cv2
        cnx_model, cnx_transform, cnx_classes, cnx_device = clasificador
        bbox = det['bbox']
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(img_src.shape[1], int(bbox[2])), min(img_src.shape[0], int(bbox[3]))
        crop = img_src[y1:y2, x1:x2]
        if crop.size == 0: return "", 0.0
        from PIL import Image as PILImage
        import torch as _torch
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pil_crop = PILImage.fromarray(crop_rgb)
        input_t = cnx_transform(pil_crop).unsqueeze(0).to(cnx_device)
        with _torch.no_grad():
            out = cnx_model(input_t)
            probs = _torch.softmax(out, dim=1)[0]
            top_prob, top_idx = probs.max(0)
        return cnx_classes[top_idx.item()], top_prob.item()

    def _process_side(det_list, baldas_list, lado, img_src):
        nonlocal det_id
        side_dets = []
        for det in det_list:
            tipo = _normalizar_tipo(det.get('class', ''))
            if tipo not in ('flor', 'planta', 'tallo'): continue
            bbox = det['bbox']
            b_idx = ubicar_en_balda(bbox, baldas_list)
            if b_idx == -1: continue
            t_idx = balda_to_ticket.get(b_idx, -1)
            balda_bbox = baldas_list[b_idx]['bbox'] if b_idx < len(baldas_list) else [0,0,1,1]
            sku, cnx_score = ("", 0.0)
            if tipo in ('flor', 'planta'):
                sku, cnx_score = _classify_crop(det, img_src)
            d = DetectionFeat(
                carro_id=carro_id, image_pair_id=image_pair_id,
                detection_id=det_id, ticket_idx=t_idx, balda_idx=b_idx,
                tipo_d=tipo, sku_d=sku, lado_d=lado,
                pos_rel_d=_pos_rel_en_balda(bbox, balda_bbox),
                volumen_d=_mask_area(det), score_mrcnn_d=det.get('score', 0.0),
                score_convnext_d=cnx_score, es_superpuesto_d=0,
                bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3],
            )
            side_dets.append(d)
            det_id += 1
        return side_dets

    dets = _process_side(det_front, baldas_f, "F", img_frontal) + \
           _process_side(det_back, baldas_b, "B", img_trasera)

    for i, d1 in enumerate(dets):
        for j, d2 in enumerate(dets):
            if i >= j: continue
            if d1.lado_d != d2.lado_d or d1.balda_idx != d2.balda_idx: continue
            if _bbox_iou([d1.bbox_x1,d1.bbox_y1,d1.bbox_x2,d1.bbox_y2],
                         [d2.bbox_x1,d2.bbox_y1,d2.bbox_x2,d2.bbox_y2]) > 0:
                d1.es_superpuesto_d = 1
                d2.es_superpuesto_d = 1
    return dets


# =============================================================================
# Función 2-4: enlazar vecinos, corresponsales, agregados
# =============================================================================

def enlazar_vecinos_misma_vista(dets):
    from collections import defaultdict
    groups = defaultdict(list)
    for d in dets: groups[(d.ticket_idx, d.balda_idx, d.lado_d)].append(d)
    for group in groups.values():
        group.sort(key=lambda x: x.pos_rel_d)
        for i, d in enumerate(group):
            if i + 1 < len(group):
                r = group[i+1]
                d.vec_der_existe_d, d.vec_der_tipo, d.vec_der_sku = 1, r.tipo_d, r.sku_d
                d.vec_der_pos_rel, d.vec_der_volumen = r.pos_rel_d, r.volumen_d
                d.vec_der_score_mrcnn, d.vec_der_score_convnext = r.score_mrcnn_d, r.score_convnext_d
                d.vec_der_superpuesto = r.es_superpuesto_d
            if i - 1 >= 0:
                l = group[i-1]
                d.vec_izq_existe_d, d.vec_izq_tipo, d.vec_izq_sku = 1, l.tipo_d, l.sku_d
                d.vec_izq_pos_rel, d.vec_izq_volumen = l.pos_rel_d, l.volumen_d
                d.vec_izq_score_mrcnn, d.vec_izq_score_convnext = l.score_mrcnn_d, l.score_convnext_d
                d.vec_izq_superpuesto = l.es_superpuesto_d

def enlazar_corresponsales_otra_vista(dets):
    from collections import defaultdict
    groups = defaultdict(list)
    for d in dets: groups[(d.ticket_idx, d.balda_idx, d.tipo_d, d.lado_d)].append(d)
    for d in dets:
        other_side = "B" if d.lado_d == "F" else "F"
        cands = groups.get((d.ticket_idx, d.balda_idx, d.tipo_d, other_side), [])
        if not cands: continue
        best = min(cands, key=lambda c: abs(c.pos_rel_d - d.pos_rel_d))
        d.otro_lado_existe_d, d.otro_lado_tipo, d.otro_lado_sku = 1, best.tipo_d, best.sku_d
        d.otro_lado_pos_rel, d.otro_lado_volumen = best.pos_rel_d, best.volumen_d
        d.otro_lado_score_mrcnn = best.score_mrcnn_d
        d.otro_lado_score_convnext = best.score_convnext_d
        d.otro_lado_dist_pos_rel = abs(d.pos_rel_d - best.pos_rel_d)
        d.otro_lado_superpuesto = best.es_superpuesto_d
        grp = groups.get((best.ticket_idx,best.balda_idx,best.tipo_d,best.lado_d), [])
        d.otro_lado_superpuesto2 = int(any(
            _bbox_iou([best.bbox_x1,best.bbox_y1,best.bbox_x2,best.bbox_y2],
                      [o.bbox_x1,o.bbox_y1,o.bbox_x2,o.bbox_y2]) > 0
            for o in grp if o.detection_id != best.detection_id))
        if best.vec_der_existe_d:
            d.otro_lado_vec_der_existe, d.otro_lado_vec_der_tipo = 1, best.vec_der_tipo
            d.otro_lado_vec_der_pos_rel, d.otro_lado_vec_der_volumen = best.vec_der_pos_rel, best.vec_der_volumen
            d.otro_lado_vec_der_score_mrcnn = best.vec_der_score_mrcnn
            d.otro_lado_vec_der_score_convnext = best.vec_der_score_convnext
        if best.vec_izq_existe_d:
            d.otro_lado_vec_izq_existe, d.otro_lado_vec_izq_tipo = 1, best.vec_izq_tipo
            d.otro_lado_vec_izq_pos_rel, d.otro_lado_vec_izq_volumen = best.vec_izq_pos_rel, best.vec_izq_volumen
            d.otro_lado_vec_izq_score_mrcnn = best.vec_izq_score_mrcnn
            d.otro_lado_vec_izq_score_convnext = best.vec_izq_score_convnext

def calcular_agregados_contexto(dets):
    from collections import defaultdict
    ct, cb = defaultdict(int), defaultdict(int)
    for d in dets:
        ct[(d.ticket_idx, d.balda_idx, d.tipo_d, d.lado_d)] += 1
        if d.tipo_d == 'tallo': cb[(d.ticket_idx, d.balda_idx)] += 1
    for d in dets:
        d.num_mismo_tipo_misma_balda_misma_vista = ct[(d.ticket_idx, d.balda_idx, d.tipo_d, d.lado_d)]
        other = "B" if d.lado_d == "F" else "F"
        d.num_mismo_tipo_misma_balda_otro_lado = ct[(d.ticket_idx, d.balda_idx, d.tipo_d, other)]
        d.num_tallos_misma_balda = cb[(d.ticket_idx, d.balda_idx)]


# =============================================================================
# Función 5: exportar_features_dataset_final
# =============================================================================

def exportar_features_dataset_final(
    dataset_dir=None, output_csv=None, mrcnn_model_path=None,
    convnext_run_dir=None, convnext_model_name="best_model.pth",
    score_thresh=0.80, carro_id="poc",
) -> str:
    """
    Recorre data/dataset_final, ejecuta Mask R-CNN + ConvNeXt,
    construye features y genera data/arbol_conteo/detections_raw.csv.
    Siempre REESCRIBE el CSV (modo 'w').
    """
    import cv2, yaml, torch, glob, re

    project_root = _PROJECT_ROOT
    if dataset_dir is None:
        dataset_dir = os.path.join(project_root, "data", "dataset_final")
    if output_csv is None:
        output_csv = os.path.join(project_root, "data", "arbol_conteo", "detections_raw.csv")
    if mrcnn_model_path is None:
        mrcnn_model_path = os.path.join(project_root, "models", "maskrcnn",
                                        "augV3_anchors_v4_hope", "model_final.pth")
    if convnext_run_dir is None:
        convnext_run_dir = os.path.join(project_root, "models", "convnext", "v2_18_clases")

    print(f"[*] Dataset: {dataset_dir}")
    print(f"[*] Output CSV: {output_csv}")

    # --- Cargar Mask R-CNN ---
    configs_dir = os.path.join(project_root, "configs")
    if configs_dir not in sys.path: sys.path.insert(0, configs_dir)
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    config1_path = os.path.join(configs_dir, "config1.yaml")
    if os.path.exists(config1_path):
        try:
            from config_manager import parse_yaml_config, apply_custom_config_to_cfg
            custom_cfg = parse_yaml_config(config1_path)
            cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
        except Exception:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = mrcnn_model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    predictor = DefaultPredictor(cfg)
    print("[+] Mask R-CNN cargado.")

    # --- Cargar ConvNeXt ---
    clasificador = None
    cnx_path = os.path.join(convnext_run_dir, convnext_model_name)
    if os.path.exists(cnx_path):
        import timm
        from torchvision import transforms
        cnx_cfg = {}
        cfg_p = os.path.join(convnext_run_dir, "config_used.yaml")
        if os.path.exists(cfg_p):
            with open(cfg_p) as f: cnx_cfg = yaml.safe_load(f)
        cnx_class_names = []
        cls_p = os.path.join(convnext_run_dir, "class_names.txt")
        if os.path.exists(cls_p):
            with open(cls_p) as f:
                for line in f:
                    parts = line.strip().split(": ", 1)
                    if len(parts) == 2: cnx_class_names.append(parts[1])
        n_cls = len(cnx_class_names)
        arch = cnx_cfg.get("MODEL", {}).get("NAME", "convnext_tiny.in12k_ft_in1k")
        img_sz = cnx_cfg.get("DATA", {}).get("IMG_SIZE", 224)
        mean = cnx_cfg.get("DATA", {}).get("MEAN", [0.485, 0.456, 0.406])
        std = cnx_cfg.get("DATA", {}).get("STD", [0.229, 0.224, 0.225])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl = timm.create_model(arch, pretrained=False, num_classes=n_cls)
        mdl.load_state_dict(torch.load(cnx_path, map_location=dev, weights_only=True))
        mdl = mdl.to(dev); mdl.eval()
        tr = transforms.Compose([transforms.Resize((img_sz, img_sz)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mean, std=std)])
        clasificador = (mdl, tr, cnx_class_names, dev)
        print(f"[+] ConvNeXt cargado ({n_cls} clases).")
    else:
        print(f"[!] ConvNeXt no encontrado en {cnx_path}.")

    class_names = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]

    # --- Encontrar pares ---
    front_files = sorted(glob.glob(os.path.join(dataset_dir, "*F.*")))
    pairs = []
    for fp in front_files:
        m = re.match(r'(\d+)F\.', os.path.basename(fp))
        if m:
            pid, ext = m.group(1), os.path.splitext(fp)[1]
            bp = os.path.join(dataset_dir, f"{pid}B{ext}")
            if os.path.exists(bp): pairs.append((pid, fp, bp))
    print(f"[*] Encontrados {len(pairs)} pares.")

    # --- Procesar ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cols = get_csv_columns()
    all_rows = []
    for idx, (pid, fp, bp) in enumerate(pairs):
        print(f"  [{idx+1}/{len(pairs)}] Par {pid}...", end=" ", flush=True)
        img_f, img_b = cv2.imread(fp), cv2.imread(bp)
        if img_f is None or img_b is None:
            print("ERROR"); continue
        out_f, out_b = predictor(img_f), predictor(img_b)
        def ext(outputs):
            inst = outputs["instances"].to("cpu")
            return [{'class': class_names[c], 'bbox': b.tolist(), 'score': float(s)}
                    for b, c, s in zip(inst.pred_boxes.tensor.numpy(),
                                       inst.pred_classes.numpy(), inst.scores.numpy())]
        det_f, det_b = ext(out_f), ext(out_b)
        res = procesar_pareja_imagenes(det_f, det_b)
        asign = res['asignacion_base']
        if not asign:
            print("SIN TICKETS/BALDAS."); continue
        bf = sorted([d for d in det_f if d['class'].lower()=='balda'], key=lambda x: x['bbox'][1])
        bb = sorted([d for d in det_b if d['class'].lower()=='balda'], key=lambda x: x['bbox'][1])
        dets = construir_detecciones_enriquecidas(det_f, det_b, asign, bf, bb,
            carro_id=carro_id, image_pair_id=pid, clasificador=clasificador,
            img_frontal=img_f, img_trasera=img_b)
        enlazar_vecinos_misma_vista(dets)
        enlazar_corresponsales_otra_vista(dets)
        calcular_agregados_contexto(dets)
        for d in dets: all_rows.append(detection_to_dict(d))
        nf = sum(1 for d in dets if d.tipo_d=='flor')
        np_ = sum(1 for d in dets if d.tipo_d=='planta')
        nt = sum(1 for d in dets if d.tipo_d=='tallo')
        print(f"{len(dets)} dets ({nf}F {np_}P {nt}T)")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(all_rows)
    print(f"\n[+] CSV generado: {output_csv} ({len(all_rows)} filas)")
    return output_csv

if __name__ == "__main__":
    exportar_features_dataset_final()
