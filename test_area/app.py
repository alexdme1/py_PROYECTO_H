"""
Test Area — Inference App (Mask R-CNN + ConvNeXt)
Run with:
    python test_area/app.py
"""

import os
import sys

# ── Auto-lanzador ──────────────────────────────────────────────────
from streamlit.runtime import exists as _st_running
if not _st_running():
    import subprocess
    sys.exit(subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.abspath(__file__),
        "--server.headless=false"
    ]).returncode)

import glob
import json
import yaml
import numpy as np
import math
import cv2
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# ── Configuración de la página ─────────────────────────────────────
st.set_page_config(
    page_title="Túnel de Flores — Test Predicciones",
    page_icon="🌸",
    layout="centered",
)

# ── Rutas base (dinámicas, funcionan en cualquier máquina) ─────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKRCNN_BASE  = os.path.join(BASE_DIR, "models", "maskrcnn")
CONVNEXT_BASE  = os.path.join(BASE_DIR, "models", "convnext")
COCO_JSON      = os.path.join(BASE_DIR, "data", "coco_unified", "annotations", "test.json")
CONFIGS_DIR    = os.path.join(BASE_DIR, "configs")


# =====================================================================
# MASK R-CNN
# =====================================================================

@st.cache_resource(show_spinner="Cargando Mask R-CNN...")
def load_maskrcnn_predictor(model_path: str, threshold: float, nms_thresh: float = 0.5):
    """Carga predictor Mask R-CNN con la config de config1.yaml (anchors, etc.)."""
    from detectron2.utils.logger import setup_logger
    setup_logger()
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Aplicar config1.yaml (anchors, RPN, ROI, NUM_CLASSES, etc.)
    config1_path = os.path.join(CONFIGS_DIR, "config1.yaml")
    if os.path.exists(config1_path):
        try:
            if CONFIGS_DIR not in sys.path:
                sys.path.insert(0, CONFIGS_DIR)
            from config_manager import parse_yaml_config, apply_custom_config_to_cfg
            custom_cfg = parse_yaml_config(config1_path)
            cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
        except Exception:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    return DefaultPredictor(cfg)


def run_maskrcnn_inference(predictor, image_bytes: bytes, class_names: list) -> np.ndarray:
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    outputs = predictor(img_bgr)

    tmp_dataset = "__tmp_test_area__"
    try:
        MetadataCatalog.get(tmp_dataset).thing_classes = class_names
    except Exception:
        pass

    v = Visualizer(img_bgr[:, :, ::-1], metadata=MetadataCatalog.get(tmp_dataset), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


# =====================================================================
# CONVNEXT
# =====================================================================

@st.cache_resource(show_spinner="Cargando ConvNeXt...")
def load_convnext_model(model_path: str, run_dir: str):
    import timm

    # Leer config usada en el entrenamiento
    config_path = os.path.join(run_dir, "config_used.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = None

    # Leer clases
    class_names_path = os.path.join(run_dir, "class_names.txt")
    class_names = []
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            for line in f:
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    class_names.append(parts[1])

    num_classes = len(class_names) if class_names else (cfg["MODEL"]["NUM_CLASSES"] if cfg else 11)
    model_name = cfg["MODEL"]["NAME"] if cfg else "convnext_tiny.in12k_ft_in1k"

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Transform
    img_size = cfg["DATA"]["IMG_SIZE"] if cfg else 224
    mean = cfg["DATA"]["MEAN"] if cfg else [0.485, 0.456, 0.406]
    std_val = cfg["DATA"]["STD"] if cfg else [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_val),
    ])

    return model, class_names, transform, device


def run_convnext_inference(model, transform, device, image_bytes: bytes, class_names: list):
    """Devuelve la clase predicha, la confianza, y un top-5."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    top5_probs, top5_indices = probs.topk(min(5, len(class_names)))

    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        results.append((class_names[idx.item()], prob.item()))

    return results


# =====================================================================
# UI
# =====================================================================

st.title("🌸 Túnel de Flores — Test Predicciones")

# Selector de módulo
model_type = st.radio(
    "🤖 Módulo:",
    [
        "Mask R-CNN (Segmentación)",
        "ConvNeXt (Clasificación)",
        "📊 Conteo (Pipeline completo)",
        "🌳 Árbol – Etiquetar",
        "🌳 Árbol – Ver anotaciones",
    ],
    horizontal=True
)

st.markdown("---")

# ─── MASK R-CNN ────────────────────────────────────────────────────
if model_type == "Mask R-CNN (Segmentación)":
    st.subheader("🎭 Mask R-CNN — Segmentación de instancias")

    run_folders = sorted(
        [d for d in glob.glob(os.path.join(MASKRCNN_BASE, "*")) if os.path.isdir(d)],
        reverse=True,
    )
    if not run_folders:
        st.error(f"No se encontraron entrenamientos en `{MASKRCNN_BASE}`")
        st.stop()

    run_labels = {os.path.basename(p): p for p in run_folders}
    selected_run = st.selectbox("📁 Run:", list(run_labels.keys()), key="mrcnn_run")
    selected_run_dir = run_labels[selected_run]

    available_models = sorted(glob.glob(os.path.join(selected_run_dir, "*.pth")))
    if not available_models:
        st.warning("No se encontraron modelos `.pth` en este run.")
        st.stop()

    model_labels = {os.path.basename(p): p for p in available_models}
    selected_model = st.selectbox("📦 Checkpoint:", list(model_labels.keys()), key="mrcnn_ckpt")
    selected_model_path = model_labels[selected_model]

    threshold = st.slider("🎯 Umbral de confianza", 0.05, 0.95, 0.50, 0.05, key="mrcnn_thresh")
    nms_thresh = st.slider("📐 NMS IoU (solapamiento)", 0.1, 0.95, 0.50, 0.05, key="mrcnn_nms",
                           help="Menor = suprime más bboxes solapadas. Mayor = permite más solapamiento.")

    # Clases
    try:
        with open(COCO_JSON) as f:
            coco = json.load(f)
        class_names = [c["name"] for c in coco["categories"]]
        num_classes = len(class_names)
        st.caption(f"Clases ({num_classes}): {', '.join(class_names)}")
    except FileNotFoundError:
        class_names = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]
        num_classes = 5

    uploaded = st.file_uploader("📂 Sube una imagen", type=["jpg", "jpeg", "png", "webp"], key="mrcnn_up")

    if uploaded is not None:
        image_bytes = uploaded.read()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image_bytes, use_container_width=True)
        with col2:
            st.subheader("Predicciones")
            with st.spinner("Ejecutando Mask R-CNN..."):
                try:
                    predictor = load_maskrcnn_predictor(selected_model_path, threshold, nms_thresh)
                    result_bgr = run_maskrcnn_inference(predictor, image_bytes, class_names)
                    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)
                except Exception as e:
                    st.error(f"Error:\n```\n{e}\n```")


# ─── CONVNEXT ──────────────────────────────────────────────────────
elif model_type == "ConvNeXt (Clasificación)":
    st.subheader("🔬 ConvNeXt — Clasificación por especie")

    run_folders = sorted(
        [d for d in glob.glob(os.path.join(CONVNEXT_BASE, "*")) if os.path.isdir(d)],
        reverse=True,
    )
    if not run_folders:
        st.error(f"No se encontraron entrenamientos en `{CONVNEXT_BASE}`")
        st.stop()

    run_labels = {os.path.basename(p): p for p in run_folders}
    selected_run = st.selectbox("📁 Run:", list(run_labels.keys()), key="cnx_run")
    selected_run_dir = run_labels[selected_run]

    available_models = sorted(glob.glob(os.path.join(selected_run_dir, "*.pth")))
    if not available_models:
        st.warning("No se encontraron modelos `.pth` en este run.")
        st.stop()

    model_labels = {os.path.basename(p): p for p in available_models}
    selected_model = st.selectbox("📦 Checkpoint:", list(model_labels.keys()), key="cnx_ckpt")
    selected_model_path = model_labels[selected_model]

    uploaded = st.file_uploader("📂 Sube una imagen (crop de flor/planta)",
                                type=["jpg", "jpeg", "png", "webp"], key="cnx_up")

    if uploaded is not None:
        image_bytes = uploaded.read()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen")
            st.image(image_bytes, use_container_width=True)

        with col2:
            st.subheader("Clasificación")
            with st.spinner("Clasificando con ConvNeXt..."):
                try:
                    model, class_names, transform, device = load_convnext_model(
                        selected_model_path, selected_run_dir
                    )
                    results = run_convnext_inference(model, transform, device, image_bytes, class_names)

                    top_class, top_prob = results[0]
                    st.metric(label="🏷️ Especie predicha", value=top_class, delta=f"{top_prob:.1%}")

                    st.markdown("**Top predicciones:**")
                    for cls_name, prob in results:
                        st.markdown(f"`{cls_name}` — **{prob:.1%}**")
                        st.progress(int(prob * 100) / 100)

                except Exception as e:
                    st.error(f"Error:\n```\n{e}\n```")


# ─── CONTEO (PIPELINE COMPLETO) ───────────────────────────────────
elif model_type == "📊 Conteo (Pipeline completo)":
    st.subheader("📊 Conteo — Pipeline completo (Mask R-CNN + ConvNeXt)")
    st.caption("Sube un par Frontal + Trasera para ejecutar detección, clasificación y conteo.")

    # --- Selectores de modelos ---
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**🎭 Mask R-CNN**")
        mrcnn_runs = sorted(
            [d for d in glob.glob(os.path.join(MASKRCNN_BASE, "*")) if os.path.isdir(d)],
            reverse=True,
        )
        if not mrcnn_runs:
            st.error("No se encontraron runs de Mask R-CNN")
            st.stop()
        mrcnn_labels = {os.path.basename(p): p for p in mrcnn_runs}
        sel_mrcnn_run = st.selectbox("Run:", list(mrcnn_labels.keys()), key="cnt_mrcnn_run")
        mrcnn_models = sorted(glob.glob(os.path.join(mrcnn_labels[sel_mrcnn_run], "*.pth")))
        mrcnn_model_labels = {os.path.basename(p): p for p in mrcnn_models}
        sel_mrcnn_ckpt = st.selectbox("Checkpoint:", list(mrcnn_model_labels.keys()), key="cnt_mrcnn_ckpt")
        mrcnn_path = mrcnn_model_labels[sel_mrcnn_ckpt]
        threshold = st.slider("Umbral confianza", 0.05, 0.95, 0.20, 0.05, key="cnt_thresh")
        nms_thresh = st.slider("NMS IoU (solapamiento)", 0.1, 0.95, 0.50, 0.05, key="cnt_nms",
                               help="Menor = suprime más bboxes solapadas.")

    with col_m2:
        st.markdown("**🔬 ConvNeXt**")
        cnx_runs = sorted(
            [d for d in glob.glob(os.path.join(CONVNEXT_BASE, "*")) if os.path.isdir(d)],
            reverse=True,
        )
        if not cnx_runs:
            st.error("No se encontraron runs de ConvNeXt")
            st.stop()
        cnx_labels = {os.path.basename(p): p for p in cnx_runs}
        sel_cnx_run = st.selectbox("Run:", list(cnx_labels.keys()), key="cnt_cnx_run")
        sel_cnx_dir = cnx_labels[sel_cnx_run]
        cnx_models = sorted(glob.glob(os.path.join(sel_cnx_dir, "*.pth")))
        cnx_model_labels = {os.path.basename(p): p for p in cnx_models}
        sel_cnx_ckpt = st.selectbox("Checkpoint:", list(cnx_model_labels.keys()), key="cnt_cnx_ckpt")
        cnx_path = cnx_model_labels[sel_cnx_ckpt]

    st.markdown("---")

    # --- Upload de imágenes ---
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_f = st.file_uploader("📷 Imagen FRONTAL", type=["jpg", "jpeg", "png", "webp"], key="cnt_front")
    with col_up2:
        uploaded_b = st.file_uploader("📷 Imagen TRASERA", type=["jpg", "jpeg", "png", "webp"], key="cnt_back")

    if uploaded_f is not None and uploaded_b is not None:
        bytes_f = uploaded_f.read()
        bytes_b = uploaded_b.read()

        # Mostrar imágenes originales
        col1, col2 = st.columns(2)
        with col1:
            st.image(bytes_f, caption="Frontal", use_container_width=True)
        with col2:
            st.image(bytes_b, caption="Trasera", use_container_width=True)

        if st.button("🚀 Ejecutar conteo", type="primary", use_container_width=True):
            with st.spinner("Procesando pipeline Mask R-CNN + ConvNeXt..."):
                try:
                    # Importar funciones de conteo
                    conteo_dir = os.path.join(BASE_DIR, "scripts", "05-logica_conteo_tallos")
                    if conteo_dir not in sys.path:
                        sys.path.insert(0, conteo_dir)
                    from conteo_module import asignar_tickets_a_baldas, procesar_pareja_imagenes, contar_articulos

                    # Decodificar imágenes
                    img_f = cv2.imdecode(np.frombuffer(bytes_f, np.uint8), cv2.IMREAD_COLOR)
                    img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

                    # Mask R-CNN
                    predictor = load_maskrcnn_predictor(mrcnn_path, threshold, nms_thresh)
                    class_names_mrcnn = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]

                    outputs_f = predictor(img_f)
                    outputs_b = predictor(img_b)

                    def extract_dets(outputs):
                        inst = outputs["instances"].to("cpu")
                        boxes = inst.pred_boxes.tensor.numpy()
                        classes = inst.pred_classes.numpy()
                        return [{'class': class_names_mrcnn[c], 'bbox': b.tolist()} for b, c in zip(boxes, classes)]

                    det_f = extract_dets(outputs_f)
                    det_b = extract_dets(outputs_b)

                    # ConvNeXt
                    cnx_model, cnx_class_names, cnx_transform, cnx_device = load_convnext_model(cnx_path, sel_cnx_dir)
                    clasificador = (cnx_model, cnx_transform, cnx_class_names, cnx_device)

                    # Pipeline de conteo
                    resultado = procesar_pareja_imagenes(det_f, det_b)
                    conteo_final, ticket_mapping, bbox_labels = contar_articulos(
                        det_f, det_b, resultado['asignacion_base'],
                        img_frontal=img_f, img_trasera=img_b, clasificador=clasificador
                    )

                    # Visualizaciones con Detectron2
                    from detectron2.utils.visualizer import Visualizer
                    from detectron2.data import MetadataCatalog
                    tmp_ds = "__conteo_app__"
                    MetadataCatalog.get(tmp_ds).set(thing_classes=class_names_mrcnn)
                    meta = MetadataCatalog.get(tmp_ds)

                    v_f = Visualizer(img_f[:, :, ::-1], metadata=meta, scale=1.0)
                    vis_f = v_f.draw_instance_predictions(outputs_f["instances"].to("cpu")).get_image()
                    v_b = Visualizer(img_b[:, :, ::-1], metadata=meta, scale=1.0)
                    vis_b = v_b.draw_instance_predictions(outputs_b["instances"].to("cpu")).get_image()

                    # Mostrar detecciones
                    st.markdown("### 🔍 Detecciones")
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.image(vis_f, caption="Frontal — Detecciones", use_container_width=True)
                    with col_v2:
                        st.image(vis_b, caption="Trasera — Detecciones", use_container_width=True)

                    # Mostrar JSON
                    st.markdown("### 📋 Resultado del conteo")
                    json_output = {"Items": conteo_final}
                    st.json(json_output, expanded=True)

                except Exception as e:
                    st.error(f"Error en el pipeline:\n```\n{e}\n```")
                    import traceback
                    st.code(traceback.format_exc())


# ─── ÁRBOL – ETIQUETAR DETECCIONES ────────────────────────────────
elif model_type == "🌳 Árbol – Etiquetar":
    import pandas as pd
    st.subheader("🌳 Árbol de Conteo — Etiquetar detecciones")
    st.caption("Etiqueta unidades por detección. Los datos provienen de `detections_raw.csv`.")

    # Importar módulo de datos
    conteo_dir = os.path.join(BASE_DIR, "scripts", "05-logica_conteo_tallos")
    if conteo_dir not in sys.path:
        sys.path.insert(0, conteo_dir)
    import decision_tree_data as dtd

    # --- Cargar CSV de features ---
    df_raw = dtd.load_raw()
    if df_raw.empty:
        st.error("❌ No existe `data/arbol_conteo/detections_raw.csv`. Ejecuta primero:\n\n"
                 "`python3 scripts/01-preprocesing/04_build_tree_features.py`")
        st.stop()

    # Solo pares que existen en el CSV
    raw_pair_ids = sorted(df_raw["image_pair_id"].astype(str).unique(),
                          key=lambda x: int(x) if x.isdigit() else x)
    dataset_path = os.path.join(BASE_DIR, "data", "dataset_final")

    # --- Inicializar navegación ---
    if "arb_pair_idx" not in st.session_state:
        st.session_state["arb_pair_idx"] = 0

    idx = st.session_state["arb_pair_idx"]
    idx = max(0, min(idx, len(raw_pair_ids) - 1))
    st.session_state["arb_pair_idx"] = idx

    # --- Helpers para callbacks ---
    def _nav_to(new_idx):
        st.session_state["arb_pair_idx"] = new_idx

    def _save_labels_for_pair(p_id, det_ids):
        new_rows = []
        for did in det_ids:
            key = f"label_{p_id}_{did}"
            val = st.session_state.get(key, 0)
            new_rows.append({"image_pair_id": p_id, "detection_id": did, "unidades_label_d": val})
        df_new = pd.DataFrame(new_rows)
        df_ex = dtd.load_labels()
        if not df_ex.empty:
            df_ex = df_ex[df_ex["image_pair_id"].astype(str) != str(p_id)]
            df_all = pd.concat([df_ex, df_new], ignore_index=True)
        else:
            df_all = df_new
        dtd.save_labels(df_all)

    def _save_and_next(p_id, det_ids, current_idx, max_idx):
        _save_labels_for_pair(p_id, det_ids)
        if current_idx < max_idx:
            _nav_to(current_idx + 1)

    # --- Progreso ---
    df_labels_all = dtd.load_labels()
    labeled_pairs = set(df_labels_all["image_pair_id"].astype(str).unique()) if not df_labels_all.empty else set()
    n_labeled = len(labeled_pairs)
    pair_id = raw_pair_ids[idx]

    st.markdown(f"### 📸 Par **{pair_id}** — {idx + 1}/{len(raw_pair_ids)} · {n_labeled} etiquetados")
    st.progress(idx / max(1, len(raw_pair_ids) - 1))

    # --- Navegación ---
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        st.button("◀ Anterior", disabled=(idx == 0), use_container_width=True,
                  key="arb_prev", on_click=_nav_to, args=(idx - 1,))
    with col_nav2:
        new_idx = st.number_input("Ir a par #", min_value=1, max_value=len(raw_pair_ids),
                                   value=idx + 1, key="arb_goto", label_visibility="collapsed")
        if new_idx - 1 != idx:
            _nav_to(new_idx - 1)
    with col_nav3:
        st.button("Siguiente ▶", disabled=(idx >= len(raw_pair_ids) - 1), use_container_width=True,
                  key="arb_next", on_click=_nav_to, args=(idx + 1,))

    st.markdown("---")

    # --- Datos del par actual desde el CSV ---
    df_pair = df_raw[df_raw["image_pair_id"].astype(str) == pair_id]

    # Cargar imágenes
    f_ext = ".png"
    f_path = os.path.join(dataset_path, f"{pair_id}F{f_ext}")
    b_path = os.path.join(dataset_path, f"{pair_id}B{f_ext}")
    # Probar con .jpg si .png no existe
    if not os.path.exists(f_path):
        f_ext = ".jpg"
        f_path = os.path.join(dataset_path, f"{pair_id}F{f_ext}")
        b_path = os.path.join(dataset_path, f"{pair_id}B{f_ext}")

    if not os.path.exists(f_path):
        st.warning(f"⚠️ No se encontraron imágenes para par {pair_id} en dataset_final.")
    else:
        img_f = cv2.imread(f_path)
        img_b = cv2.imread(b_path) if os.path.exists(b_path) else None

        # Dibujar bboxes desde el CSV (sin necesidad de inferencia)
        img_f_draw = img_f.copy() if img_f is not None else None
        img_b_draw = img_b.copy() if img_b is not None else None

        for _, row in df_pair.iterrows():
            tipo = str(row.get("tipo_d", ""))
            if tipo == "tallo":
                continue
            color = (0, 255, 0) if tipo == "flor" else (255, 165, 0)
            lado = str(row.get("lado_d", ""))
            img_d = img_f_draw if lado == "F" else img_b_draw
            if img_d is None:
                continue
            x1, y1 = int(row["bbox_x1"]), int(row["bbox_y1"])
            x2, y2 = int(row["bbox_x2"]), int(row["bbox_y2"])
            cv2.rectangle(img_d, (x1, y1), (x2, y2), color, 2)
            det_id = int(row["detection_id"])
            sku = str(row.get("sku_d", ""))
            cv2.putText(img_d, f"#{det_id} {sku or tipo}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        col1, col2 = st.columns(2)
        with col1:
            if img_f_draw is not None:
                st.image(cv2.cvtColor(img_f_draw, cv2.COLOR_BGR2RGB), caption="Frontal")
        with col2:
            if img_b_draw is not None:
                st.image(cv2.cvtColor(img_b_draw, cv2.COLOR_BGR2RGB), caption="Trasera")

    # --- Formulario de etiquetado (solo flor/planta del CSV) ---
    plant_rows = df_pair[df_pair["tipo_d"].isin(["flor", "planta"])].copy()

    # Labels existentes
    existing = {}
    if not df_labels_all.empty:
        for _, row in df_labels_all.iterrows():
            if str(row["image_pair_id"]) == str(pair_id):
                existing[int(row["detection_id"])] = int(row["unidades_label_d"])

    if plant_rows.empty:
        st.info("No hay detecciones de flor/planta en este par.")
    else:
        st.markdown(f"### ✏️ Etiquetar unidades ({len(plant_rows)} detecciones)")
        n_cols = min(4, len(plant_rows))
        cols = st.columns(n_cols)
        det_ids_for_save = []
        for i, (_, row) in enumerate(plant_rows.iterrows()):
            did = int(row["detection_id"])
            det_ids_for_save.append(did)
            with cols[i % n_cols]:
                default_val = existing.get(did, 0)
                tipo = str(row.get("tipo_d", ""))
                sku = str(row.get("sku_d", ""))
                st.number_input(
                    f"#{did} {tipo} {sku}",
                    min_value=0, max_value=50, value=default_val,
                    key=f"label_{pair_id}_{did}"
                )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.button("💾 Guardar", use_container_width=True, key="arb_save",
                      on_click=_save_labels_for_pair, args=(pair_id, det_ids_for_save))
        with col_s2:
            st.button("💾 Guardar y Siguiente ▶", type="primary", use_container_width=True,
                      key="arb_save_next",
                      on_click=_save_and_next,
                      args=(pair_id, det_ids_for_save, idx, len(raw_pair_ids) - 1))











# ─── ÁRBOL – VER ANOTACIONES ──────────────────────────────────────
elif model_type == "🌳 Árbol – Ver anotaciones":
    st.subheader("🌳 Árbol de Conteo — Visualización de anotaciones")

    conteo_dir = os.path.join(BASE_DIR, "scripts", "05-logica_conteo_tallos")
    if conteo_dir not in sys.path:
        sys.path.insert(0, conteo_dir)
    import decision_tree_data as dtd

    # Cargar datos
    import pandas as pd
    df = dtd.merge_raw_labels()
    if df.empty:
        st.warning("No hay datos. Ejecuta primero `exportar_features_dataset_final()`.")
        st.stop()

    # Estadísticas generales
    n_total = len(df)
    n_labeled = int((df["unidades_label_d"] >= 0).sum())
    n_unlabeled = n_total - n_labeled
    n_pairs = df["image_pair_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total detecciones", n_total)
    col2.metric("Etiquetadas", n_labeled)
    col3.metric("Sin etiquetar", n_unlabeled)
    col4.metric("Pares de imágenes", n_pairs)

    st.markdown("---")

    # Filtros
    pair_ids_available = sorted(df["image_pair_id"].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    show_unlabeled = st.checkbox("Mostrar solo sin etiquetar", key="arb_vis_unlabeled")

    sel_pair_id = st.selectbox("📸 Par de imágenes:", pair_ids_available, key="arb_vis_pair")

    df_pair = df[df["image_pair_id"].astype(str) == str(sel_pair_id)]
    if show_unlabeled:
        df_pair = df_pair[df_pair["unidades_label_d"] < 0]

    # Filtrar solo flor/planta
    df_display = df_pair[df_pair["tipo_d"].isin(["flor", "planta"])]

    # Tabla de detecciones
    display_cols = [
        "detection_id", "tipo_d", "sku_d", "lado_d", "balda_idx",
        "pos_rel_d", "volumen_d", "score_mrcnn_d", "score_convnext_d",
        "es_superpuesto_d", "unidades_label_d",
    ]
    cols_present = [c for c in display_cols if c in df_display.columns]

    st.markdown(f"### Detecciones del par {sel_pair_id}")
    st.dataframe(df_display[cols_present], use_container_width=True)

    # Resumen por SKU
    if not df_display.empty and "sku_d" in df_display.columns:
        st.markdown("### 📊 Resumen por SKU")
        df_labeled = df_display[df_display["unidades_label_d"] >= 0]
        if not df_labeled.empty:
            resumen = df_labeled.groupby("sku_d").agg(
                tipo=('tipo_d', 'first'),
                detecciones=('detection_id', 'count'),
                unidades_totales=('unidades_label_d', 'sum'),
            ).reset_index()
            st.dataframe(resumen, use_container_width=True)
        else:
            st.info("No hay detecciones etiquetadas para este par.")

    # Imagen con anotaciones (si existe en dataset_final)
    dataset_path = os.path.join(BASE_DIR, "data", "dataset_final")
    for ext in [".png", ".jpg", ".jpeg"]:
        f_path = os.path.join(dataset_path, f"{sel_pair_id}F{ext}")
        b_path = os.path.join(dataset_path, f"{sel_pair_id}B{ext}")
        if os.path.exists(f_path) and os.path.exists(b_path):
            img_f = cv2.imread(f_path)
            img_b = cv2.imread(b_path)

            # Dibujar anotaciones
            for _, row in df_display.iterrows():
                img_draw = img_f if row["lado_d"] == "F" else img_b
                x1, y1 = int(row["bbox_x1"]), int(row["bbox_y1"])
                x2, y2 = int(row["bbox_x2"]), int(row["bbox_y2"])
                lbl = row.get("unidades_label_d", -1)
                color = (0, 200, 0) if lbl >= 0 else (0, 0, 255)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                txt = f"#{int(row['detection_id'])} {row['sku_d']} u={lbl}"
                cv2.putText(img_draw, txt, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB), caption="Frontal", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB), caption="Trasera", use_container_width=True)
            break
