"""
Test Area — Inference App (Mask R-CNN + ConvNeXt)
Run with:
    /home/servi2/Enviroments/main_venv/bin/python test_area/app.py
"""

import os
import sys

# ── Auto-lanzador ──────────────────────────────────────────────────
from streamlit.runtime import exists as _st_running
if not _st_running():
    import subprocess
    sys.exit(subprocess.run([
        "/home/servi2/Enviroments/main_venv/bin/streamlit", "run",
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

# ── Rutas base ─────────────────────────────────────────────────────
BASE_DIR       = "/home/servi2/Escritorio/py_PROYECTO_H"
MASKRCNN_BASE  = os.path.join(BASE_DIR, "models", "maskrcnn")
CONVNEXT_BASE  = os.path.join(BASE_DIR, "models", "convnext")
COCO_JSON      = os.path.join(BASE_DIR, "data", "coco_unified", "annotations", "test.json")
CONFIGS_DIR    = os.path.join(BASE_DIR, "configs")


# =====================================================================
# MASK R-CNN
# =====================================================================

@st.cache_resource(show_spinner="Cargando Mask R-CNN...")
def load_maskrcnn_predictor(model_path: str, threshold: float):
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
    ["Mask R-CNN (Segmentación)", "ConvNeXt (Clasificación)", "📊 Conteo (Pipeline completo)"],
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

    threshold = st.slider("🎯 Umbral de confianza", 0.1, 0.95, 0.50, 0.05, key="mrcnn_thresh")

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
                    predictor = load_maskrcnn_predictor(selected_model_path, threshold)
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
else:
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
                    scripts_dir = os.path.join(BASE_DIR, "scripts")
                    if scripts_dir not in sys.path:
                        sys.path.insert(0, scripts_dir)
                    from conteo_module import asignar_tickets_a_baldas, procesar_pareja_imagenes, contar_articulos

                    # Decodificar imágenes
                    img_f = cv2.imdecode(np.frombuffer(bytes_f, np.uint8), cv2.IMREAD_COLOR)
                    img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

                    # Mask R-CNN
                    predictor = load_maskrcnn_predictor(mrcnn_path, threshold)
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
                    conteo_final, ticket_mapping = contar_articulos(
                        det_f, det_b, resultado['asignacion_base'],
                        img_frontal=img_f, clasificador=clasificador
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
