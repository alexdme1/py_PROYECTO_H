import os
import sys
import cv2

# Desde scripts/03-model_eval/ → subir 2 niveles para llegar a la raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "configs"))
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

setup_logger()

def main(model_path, min_area=5000):
    unified_images = os.path.join("data", "coco_unified", "images")
    unified_ann_dir = os.path.join("data", "coco_unified", "annotations")
    
    # Utilizaremos la anotación de "test" separada como manda detectron standard
    test_json = os.path.join(unified_ann_dir, "test.json")
    
    # Asegúrate de registrar el datset que usas (nativamente)
    if "flores_test" not in DatasetCatalog.list() and os.path.exists(test_json):
        register_coco_instances("flores_test", {}, test_json, unified_images)
        
    metadata = MetadataCatalog.get("flores_test")
    model_dir = os.path.dirname(model_path)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = model_path
    
    from config_manager import parse_yaml_config, apply_custom_config_to_cfg
    custom_cfg = parse_yaml_config()
    cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
    
    # Solo pintar lo que esté un 25% o más seguro
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
    cfg.DATASETS.TEST = ("flores_test", )

    predictor = DefaultPredictor(cfg)
    
    print("\n" + "="*50)
    print(" INICIANDO EVALUACIÓN COCO (Test Split)")
    print("="*50)
    evaluator = COCOEvaluator("flores_test", output_dir=os.path.join(model_dir, "inference_results_test"))
    val_loader = build_detection_test_loader(cfg, "flores_test")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    
    vis_output_dir = os.path.join(model_dir, "eval_vis_test")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(" GENERANDO VISUALIZACIONES FILTRADAS")
    print("="*50)
    
    dataset_dicts = DatasetCatalog.get("flores_test")
    
    for d in dataset_dicts:
        im_path = d["file_name"]
        im = cv2.imread(im_path)
        if im is None:
            continue
            
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        
        # Filtro de area (pixeles de la mascara frente a area_min)
        if instances.has("pred_masks"):
            areas = instances.pred_masks.sum(dim=(1, 2))
            keep_idx = areas >= min_area
            filtered_instances = instances[keep_idx]
        else:
            bboxes = instances.pred_boxes.tensor
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            keep_idx = areas >= min_area
            filtered_instances = instances[keep_idx]
            
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(filtered_instances)
        
        base_name = os.path.basename(im_path)
        out_path = os.path.join(vis_output_dir, base_name)
        cv2.imwrite(out_path, out.get_image()[:, :, ::-1])

    print(f"[*] Visualizaciones guardadas en: {vis_output_dir}")

if __name__ == "__main__":
    import os
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
    MODEL_PATH = os.path.join(out_dir, run_name, "model_final.pth")
    # Área mínima en píxeles para dibujar el objeto
    AREA_MIN_PIXELS = 5000
    
    main(MODEL_PATH, min_area=AREA_MIN_PIXELS)
