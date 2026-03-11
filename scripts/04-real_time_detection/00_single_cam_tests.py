"""
00_single_cam_tests.py — Detección bajo demanda con Mask R-CNN (single cam).

Muestra el feed en directo de la cámara RTSP sin correr inferencia.
Al pulsar ESPACIO, captura el frame, cropea la ROI definida por el
usuario y ejecuta Mask R-CNN + asignación ticket→balda sobre ese crop.

Flujo:
  1. Al arrancar → se muestra el feed en vivo.
  2. Pulsar [r] → seleccionar ROI con el ratón (rectángulo sobre el feed).
  3. Pulsar [ESPACIO] → captura, cropea la ROI, infiere y muestra resultado.
  4. Pulsar [q] → salir.

Uso:
    python scripts/04-real_time_detection/00_single_cam_tests.py
"""

import os
import sys
import cv2
import numpy as np
import time

# ── Paths del proyecto ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "configs"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "05-logica_conteo_tallos"))

# ── Imports de Detectron2 ────────────────────────────────────────────
from detectron2.utils.logger import setup_logger
setup_logger(name="detectron2", abbrev_name="d2")
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

# ── Import del módulo de conteo ──────────────────────────────────────
from conteo_module import asignar_tickets_a_baldas
from config_manager import parse_yaml_config, apply_custom_config_to_cfg


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

RTSP_URL = "rtsp://admin:Verdnatura1.@10.1.14.15:554/Streaming/Channels/101"

MRCNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "maskrcnn",
                                 "big_aug_anchors_v4_hope", "model_final.pth")

SCORE_THRESHOLD = 0.35

CLASS_NAMES = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]

# Factor de escala para la ventana del feed en directo
LIVE_SCALE = 0.20

# Colores por clase (BGR)
CLASS_COLORS = {
    "Flores":      (0, 200, 255),
    "ticket":      (0, 255, 255),
    "Balda":       (255, 180, 0),
    "Planta":      (0, 255, 0),
    "tallo_grupo": (180, 0, 255),
}


# =====================================================================
# CARGA DEL MODELO
# =====================================================================

# Asegurar CWD = raíz del proyecto para que parse_yaml_config() encuentre configs/
os.chdir(PROJECT_ROOT)

def load_predictor(model_path, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    # Aplicar config1.yaml (anchors, RPN, ROI, etc.)
    custom_cfg = parse_yaml_config()
    cfg = apply_custom_config_to_cfg(cfg, custom_cfg)

    cfg.MODEL.WEIGHTS = model_path
    # Setear threshold DESPUÉS del YAML (el YAML sobreescribe si no)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    ds_name = "__single_cam_rt__"
    if ds_name not in MetadataCatalog.list():
        MetadataCatalog.get(ds_name).set(thing_classes=CLASS_NAMES)

    # Debug: confirmar configuración crítica
    print(f"    Anchors: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
    print(f"    Input test: MIN={cfg.INPUT.MIN_SIZE_TEST}, MAX={cfg.INPUT.MAX_SIZE_TEST}")
    print(f"    Score thresh: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    print(f"    Num classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")

    return DefaultPredictor(cfg), MetadataCatalog.get(ds_name), cfg


def extract_detections(outputs):
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    return [
        {'class': CLASS_NAMES[classes[i]], 'bbox': boxes[i].tolist(), 'score': float(scores[i])}
        for i in range(len(boxes))
    ]


# =====================================================================
# DIBUJO
# =====================================================================

def draw_results(crop, detections, asignacion):
    """Dibuja detecciones + panel de asignación sobre el crop."""
    result = crop.copy()

    # Bboxes por clase
    for det in detections:
        cls = det['class']
        x1, y1, x2, y2 = map(int, det['bbox'])
        score = det.get('score', 0)
        color = CLASS_COLORS.get(cls, (200, 200, 200))

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Panel de asignación
    baldas = sorted([d for d in detections if d['class'].lower() == 'balda'],
                    key=lambda d: d['bbox'][1])
    total_b = len(baldas)

    lines = [f"RESULTADO — {len(detections)} detecciones"]
    if asignacion:
        lines.append("")
        for t_idx, b_indices in asignacion.items():
            balda_names = [f"Balda_{total_b - bi}" for bi in b_indices]
            t_det = detections[t_idx] if t_idx < len(detections) else None
            conf_str = f" ({t_det['score']:.0%})" if t_det else ""
            lines.append(f"  Ticket{conf_str} -> {', '.join(balda_names)}")
    else:
        lines.append("  Sin asignacion (faltan baldas o tickets)")

    # Dibujar panel
    lh = 22
    pw = 380
    ph = len(lines) * lh + 12
    overlay = result.copy()
    cv2.rectangle(overlay, (5, 5), (5 + pw, 5 + ph), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, result, 0.2, 0, result)

    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(result, line, (12, 26 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "=" * 60)
    print(" MASK R-CNN — Single Cam (captura bajo demanda)")
    print("=" * 60)

    # Verificar modelo
    if not os.path.exists(MRCNN_MODEL_PATH):
        print(f"[ERROR] Modelo no encontrado: {MRCNN_MODEL_PATH}")
        sys.exit(1)

    # Cargar modelo
    print(f"[*] Cargando Mask R-CNN (umbral={SCORE_THRESHOLD})...")
    predictor, metadata, cfg = load_predictor(MRCNN_MODEL_PATH, SCORE_THRESHOLD)
    print(f"[+] Modelo cargado.\n")

    # Conectar cámara
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    print(f"[*] Conectando a cámara...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] No se pudo conectar a la cámara")
        sys.exit(1)
    print("[+] Cámara conectada.\n")

    # Carpeta para capturas
    captures_dir = os.path.join(PROJECT_ROOT, "outputs", "rt_captures")
    os.makedirs(captures_dir, exist_ok=True)

    # Estado
    roi = None           # (x, y, w, h) en coordenadas del frame rotado FULL-RES
    result_window = None  # Último resultado para mantener abierto

    WINDOW_NAME = "Feed en directo"

    print("=" * 60)
    print(" CONTROLES:")
    print("   [r]       → Seleccionar ROI (zona de detección)")
    print("   [ESPACIO] → Capturar + detectar en la ROI")
    print("   [s]       → Guardar última captura")
    print("   [c]       → Cerrar ventana de resultados")
    print("   [q]       → Salir")
    print("=" * 60 + "\n")

    while True:
        ok, frame_raw = cap.read()
        if not ok:
            continue

        # Rotar 90° CCW (igual que el test de conexión)
        frame_full = cv2.rotate(frame_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_full, w_full = frame_full.shape[:2]

        # Crear copia para display (escala reducida)
        disp_w = int(w_full * LIVE_SCALE)
        disp_h = int(h_full * LIVE_SCALE)
        frame_disp = cv2.resize(frame_full, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # Dibujar ROI en el display si está definida
        if roi is not None:
            rx, ry, rw, rh = roi
            # Escalar la ROI de full-res a display
            dx1 = int(rx * LIVE_SCALE)
            dy1 = int(ry * LIVE_SCALE)
            dx2 = int((rx + rw) * LIVE_SCALE)
            dy2 = int((ry + rh) * LIVE_SCALE)
            cv2.rectangle(frame_disp, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            cv2.putText(frame_disp, f"ROI {rw}x{rh}", (dx1, dy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Barra de estado
        status_roi = f"ROI: {roi[2]}x{roi[3]}" if roi else "ROI: no definida [r]"
        bar_text = f"{status_roi}  |  [ESPACIO] detectar  |  [r] ROI  |  [q] salir"
        cv2.rectangle(frame_disp, (0, disp_h - 22), (disp_w, disp_h), (40, 40, 40), -1)
        cv2.putText(frame_disp, bar_text, (6, disp_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, frame_disp)

        # ── Teclado ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            # Seleccionar ROI sobre una imagen estática redimensionada
            # Usamos una escala intermedia para que sea manejable
            sel_scale = 0.35
            sel_w = int(w_full * sel_scale)
            sel_h = int(h_full * sel_scale)
            frame_sel = cv2.resize(frame_full, (sel_w, sel_h), interpolation=cv2.INTER_AREA)

            print("[*] Selecciona la ROI con el ratón. Pulsa ENTER para confirmar, ESC para cancelar.")
            sel = cv2.selectROI("Seleccionar ROI", frame_sel, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Seleccionar ROI")

            sx, sy, sw, sh = sel
            if sw > 0 and sh > 0:
                # Convertir de escala de selección a full-res
                roi = (
                    int(sx / sel_scale),
                    int(sy / sel_scale),
                    int(sw / sel_scale),
                    int(sh / sel_scale),
                )
                print(f"[+] ROI definida: x={roi[0]}, y={roi[1]}, "
                      f"w={roi[2]}, h={roi[3]} (full-res)")
            else:
                print("[!] Selección cancelada, ROI no cambiada.")

        elif key == ord(' '):
            # ── CAPTURA + DETECCIÓN ──
            if roi is None:
                print("[!] Primero define una ROI con [r].")
                continue

            rx, ry, rw, rh = roi

            # Clamp para no salirse de la imagen
            rx = max(0, min(rx, w_full - 1))
            ry = max(0, min(ry, h_full - 1))
            rw = min(rw, w_full - rx)
            rh = min(rh, h_full - ry)

            crop = frame_full[ry:ry+rh, rx:rx+rw].copy()

            if crop.size == 0:
                print("[!] Crop vacío, redefine la ROI con [r].")
                continue

            # Reescalar crop a exactamente 1024x1024 (lo que espera la red)
            crop = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_AREA)

            print(f"\n[*] Captura tomada.")
            print(f"    Frame full-res: {w_full}x{h_full}")
            print(f"    ROI: x={rx}, y={ry}, w={rw}, h={rh}")
            print(f"    Crop para red: {crop.shape[1]}x{crop.shape[0]}")

            # Guardar crop para debug
            debug_path = os.path.join(captures_dir, "_last_crop_debug.png")
            cv2.imwrite(debug_path, crop)
            print(f"    Crop guardado en: {debug_path}")

            print(f"    Ejecutando Mask R-CNN...")
            t0 = time.time()

            # Inferencia
            outputs = predictor(crop)
            detections = extract_detections(outputs)
            elapsed = time.time() - t0

            print(f"[+] Inferencia: {elapsed:.2f}s — {len(detections)} detecciones")

            # Resumen por clase
            from collections import Counter
            cls_count = Counter(d['class'] for d in detections)
            for cls, n in cls_count.items():
                print(f"    {cls}: {n}")

            if not detections:
                print("    [!] 0 detecciones. Posibles causas:")
                print("        - La ROI no encuadra el carro/baldas correctamente")
                print("        - La imagen de la cámara es muy diferente al training data")
                print(f"        - Revisa el crop: {debug_path}")

            # Asignación ticket → balda
            asignacion = asignar_tickets_a_baldas(detections)

            if asignacion:
                baldas_sorted = sorted(
                    [d for d in detections if d['class'].lower() == 'balda'],
                    key=lambda d: d['bbox'][1]
                )
                total_b = len(baldas_sorted)
                for t_idx, b_indices in asignacion.items():
                    names = [f"Balda_{total_b - bi}" for bi in b_indices]
                    print(f"    Ticket idx={t_idx} → {', '.join(names)}")

            # Dibujar resultado
            result_img = draw_results(crop, detections, asignacion)
            result_window = result_img

            # Mostrar en ventana aparte (caber en pantalla: max 900px alto)
            rh_disp, rw_disp = result_img.shape[:2]
            max_h = 900
            if rh_disp > max_h:
                scale = max_h / rh_disp
                result_show = cv2.resize(result_img, (int(rw_disp * scale), int(rh_disp * scale)))
            else:
                result_show = result_img

            cv2.imshow("Resultado deteccion", result_show)
            print()

        elif key == ord('s'):
            if result_window is not None:
                fname = f"capture_{int(time.time())}.png"
                fpath = os.path.join(captures_dir, fname)
                cv2.imwrite(fpath, result_window)
                print(f"[+] Guardada: {fpath}")
            else:
                print("[!] No hay resultado para guardar. Pulsa ESPACIO primero.")

        elif key == ord('c'):
            cv2.destroyWindow("Resultado deteccion")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[*] Sesión finalizada.")


if __name__ == "__main__":
    main()
