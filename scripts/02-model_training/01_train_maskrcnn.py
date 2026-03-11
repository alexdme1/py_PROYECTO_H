import os
import sys
import time
import requests
import subprocess
from datetime import datetime

# ==============================================================================
# IMPORTACIONES DE DETECTRON2
# ==============================================================================
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader

# Importe de configuración nativo del proyecto (nuestra pasarela YAML)
# Desde scripts/02-model_training/ → subir 2 niveles para llegar a la raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "configs"))
try:
    from config_manager import parse_yaml_config, apply_custom_config_to_cfg
except ImportError:
    raise ImportError("No se encontró config_manager.py en configs/. "
                      f"Buscado en: {os.path.join(_PROJECT_ROOT, 'configs')}")

# ==============================================================================
# AJUSTES GLOBALES
# ==============================================================================
setup_logger()
CLASES_FINAL = ["Flores", "ticket", "Balda", "Planta", "tallo_grupo"]


class CustomEvaluatorTrainer(DefaultTrainer):
    """
    Subclase personalizada del Entrenador por defecto (DefaultTrainer).
    Cumple dos funciones vitales en el proyecto H:
      1. Inyecta el COCOEvaluator para probar matemáticamente el modelo sin apagar el script.
      2. Lee si los Augmentations Fotográficos Extremos están habilitados en YAML
         y los aplica en tiempo real usando el DatasetMapper a la VRAM.
    """
    custom_cfg_data = {}

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        aug_cfg = cls.custom_cfg_data.get("INPUT", {}).get("AUGMENTATIONS_EXTRA", {})
        
        # Si el yaml maestro activa los filtros, construimos la tubería de tortura visual:
        if aug_cfg.get("ACTIVAS", False):
            print("\n[*] 📸 ¡ACTIVANDO AUGMENTATIONS FOTOGRÁFICOS EXTREMOS DE ILUMINACIÓN!")
            print(f"    -> Probabilidad de apagar las sombras: {aug_cfg.get('PROBABILIDAD', 0.50) * 100}%")
            
            # El mapper aplicará una de las alteraciones en caliente a cada foto leída
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[
                T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN, 
                                     max_size=cfg.INPUT.MAX_SIZE_TRAIN, sample_style="choice"),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomBrightness(aug_cfg.get("BRILLO_MIN", 0.6), aug_cfg.get("BRILLO_MAX", 1.1)),
                T.RandomContrast(0.6, 1.2)
            ])
            return build_detection_train_loader(cfg, mapper=mapper)
        
        # En caso contrario, usamos el Loader ordinario de Detectron2
        else:
            return build_detection_train_loader(cfg)


def main():
    print("\n" + "="*60)
    print(" INICIANDO ENTRENAMIENTO MASK R-CNN (YAML CONFIG)")
    print("="*60)

    # 1. PARSEO Y EXTRACCIÓN DEL NOMBRE BASE DE LOS DATOS CREADOS
    # ==============================================================================
    name_file = os.path.join("data", "coco_unified", "model_name.txt")
    model_name = "default"
    if os.path.exists(name_file):
        with open(name_file, "r") as f:
            model_name = f.read().strip()
            
    unified_images = os.path.join("data", "coco_unified", "images")
    unified_ann_dir = os.path.join("data", "coco_unified", "annotations")
    
    # 2. REGISTRO NATIVO DE LAS BASES DE DATOS FOTOGRÁFICAS 
    # ==============================================================================
    train_json = os.path.join(unified_ann_dir, "train.json")
    val_json = os.path.join(unified_ann_dir, "valid.json")
    test_json = os.path.join(unified_ann_dir, "test.json")
    
    if os.path.exists(train_json):
        register_coco_instances("flores_train", {}, train_json, unified_images)
    if os.path.exists(val_json):
        register_coco_instances("flores_val", {}, val_json, unified_images)
    if os.path.exists(test_json):
        register_coco_instances("flores_test", {}, test_json, unified_images)

    # 3. CONSTRUCCIÓN DE LA RED NEURONAL DESDE CERO USANDO EL YAML DE RECETAS
    # ==============================================================================
    cfg = get_cfg()
    
    # - Leemos el archivo central YAML
    custom_cfg = parse_yaml_config()
    CustomEvaluatorTrainer.custom_cfg_data = custom_cfg # Lo pasamos a la clase
    
    # - Asignación de Carpeta Dinámica
    SUFIJO_VERSION = custom_cfg.get("MODEL_INFO", {}).get("SUFIJO_VERSION", "_v_desconocida")
    output_base_dir = custom_cfg.get("MODEL_INFO", {}).get("OUTPUT_DIR_BASE", "models/maskrcnn")
    base_weights_yaml = custom_cfg.get("MODEL_INFO", {}).get("BASE_WEIGHTS_YAML", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    run_name = f"{model_name}{SUFIJO_VERSION}"
    cfg.OUTPUT_DIR = os.path.join(output_base_dir, run_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # - Inyectamos arquitectura base C++ y pesos limpios (Transfer Learning COCO puro)
    cfg.merge_from_file(model_zoo.get_config_file(base_weights_yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_weights_yaml)
    
    # - Configuramos Datasets 
    cfg.DATASETS.TRAIN = ("flores_train",)
    if os.path.exists(test_json):
        cfg.DATASETS.TEST = ("flores_test",)
    elif os.path.exists(val_json):
        cfg.DATASETS.TEST = ("flores_val",)
    else:
        cfg.DATASETS.TEST = ()

    # - [!!] EL MAGO: Inyectar TODOS los parámetros configurados en config_manager.py
    cfg = apply_custom_config_to_cfg(cfg, custom_cfg)
    
    print(f"[*] Carpeta objetivo de exportación configurada: {run_name}")
    
    # 4. LANZAMIENTO DE HERRAMIENTAS DE MONITORIZACIÓN (TENSORBOARD)
    # ==============================================================================
    tb_dir = os.path.dirname(cfg.OUTPUT_DIR)
    print(f"[*] Lanzando TensorBoard en el puerto 6006 escuchando la carpeta: {tb_dir}")
    print(f"    -> Accede en tu navegador a: http://localhost:6006")
    subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", tb_dir, "--port", "6006"],
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )

    # 5. LANZAMIENTO DEL BUCLE DE ENTRENAMIENTO (BACKPROPAGATION)
    # ==============================================================================
    print("\n[*] Arrancando procesador de gradientes (Trainer)...")
    start_time = time.time()
    
    trainer = CustomEvaluatorTrainer(cfg)
    
    # Force 'resume=False' por defecto si el nombre ha cambiado, se reescriben pesos nuevos
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    end_time = time.time()
    tiempo_total = end_time - start_time
    horas = int(tiempo_total // 3600)
    minutos = int((tiempo_total % 3600) // 60)
    
    print("\n[*] Entrenamiento completado con éxito 🥳")
    print(f"[*] Ejecutando última evaluación en el Test Split para generar métricas JSON...")
    
    # 6. EXTRACCIÓN FINAL DE MÉTRICAS MATEMÁTICAS (MAP) Y EMISIÓN DE PING
    # ==============================================================================
    metrics_dict = trainer.test(cfg, trainer.model)
    
    mensaje_push = f"✅ MODELO MASK R-CNN LISTO ({run_name})\n"
    mensaje_push += f"⏱️ Tiempo GPU total: {horas}h {minutos}m\n\n"
    
    if isinstance(metrics_dict, dict) and "bbox" in metrics_dict:
        mAP = metrics_dict["bbox"].get("AP", 0.0)
        mensaje_push += f"🎯 Precisión Pura Global (mAP): {mAP:.2f}\n"
        mensaje_push += "--- AP por Clase ---\n"
        for k, v in metrics_dict["bbox"].items():
            if str(k).startswith("AP-"):
                clase_nombre = str(k).replace("AP-", "")
                mensaje_push += f"🌸 {clase_nombre}: {v:.2f}\n"
    else:
        mensaje_push += "⚠️ No se registraron métricas Bounding-Box legibles.\n"
        
    print(f"\n[*] Enviando notificación PUSH al Teléfono (Topic: /train_mrcnn)...\n")
    print(f"--------- MENSAJE ---------\n{mensaje_push}---------------------------\n")
    
    try:
        requests.post("https://ntfy.sh/train_mrcnn",
            data=mensaje_push.encode('utf-8'),
            headers={
                "Title": "Mask R-CNN: Proyecto H",
                "Priority": "urgent",
                "Tags": "tada,robot"
            }
        )
        print("[+] Notificación PUSH inyectada a la red con éxito.")
    except Exception as e:
        print(f"[-] Ocurrió un error mandando el PING de Android: {e}")

if __name__ == "__main__":
    main()
