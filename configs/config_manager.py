import os
import yaml

def parse_yaml_config(yaml_path="configs/config1.yaml"):
    """
    Lee el archivo YAML maestro con la configuración de la red.
    """
    if not os.path.exists(yaml_path):
        # Fallback de ruta por si se ejecuta desde /test_area o similar
        parent_yaml = os.path.join("..", yaml_path)
        if os.path.exists(parent_yaml):
            yaml_path = parent_yaml
        else:
            raise FileNotFoundError(f"No se encuentra el archivo de config maestro en: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    return config_data

def apply_custom_config_to_cfg(cfg, config_data):
    """
    Inyecta el diccionario YAML leido en la estructura core estricta del
    'get_cfg()' de Detectron2, evitando problemas con nuevas claves.
    """
    if "MODEL" in config_data:
        m = config_data["MODEL"]
        if "ROI_BOX_HEAD" in m:
            box_head_name = m["ROI_BOX_HEAD"].get("NAME", cfg.MODEL.ROI_BOX_HEAD.NAME)
            cfg.MODEL.ROI_BOX_HEAD.NAME = box_head_name
            
            # Solo añadir estas si la cabeza necesita redes convolucionales adicionales
            # FastRCNNConvFCHead es el default y previene fallos del registry interno de detectron
            if box_head_name == "FastRCNNConvFCHead":
                cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = m["ROI_BOX_HEAD"].get("NUM_CONV", cfg.MODEL.ROI_BOX_HEAD.NUM_CONV)
            
            cfg.MODEL.ROI_BOX_HEAD.NUM_FC = m["ROI_BOX_HEAD"].get("NUM_FC", cfg.MODEL.ROI_BOX_HEAD.NUM_FC)
            
        if "ANCHOR_GENERATOR" in m:
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = m["ANCHOR_GENERATOR"].get("SIZES", cfg.MODEL.ANCHOR_GENERATOR.SIZES)
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = m["ANCHOR_GENERATOR"].get("ASPECT_RATIOS", cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS)
            
        if "RPN" in m:
            cfg.MODEL.RPN.NMS_THRESH = m["RPN"].get("NMS_THRESH", cfg.MODEL.RPN.NMS_THRESH)
            cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = m["RPN"].get("POST_NMS_TOPK_TRAIN", cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN)
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = m["RPN"].get("POST_NMS_TOPK_TEST", cfg.MODEL.RPN.POST_NMS_TOPK_TEST)
            
        if "ROI_HEADS" in m:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = m["ROI_HEADS"].get("NUM_CLASSES", cfg.MODEL.ROI_HEADS.NUM_CLASSES)
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = m["ROI_HEADS"].get("BATCH_SIZE_PER_IMAGE", cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = m["ROI_HEADS"].get("NMS_THRESH_TEST", cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = m["ROI_HEADS"].get("SCORE_THRESH_TEST", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)

    if "INPUT" in config_data:
        i = config_data["INPUT"]
        # Convertir listas yaml a tuples para Detectron2
        if "MIN_SIZE_TRAIN" in i: cfg.INPUT.MIN_SIZE_TRAIN = tuple(i["MIN_SIZE_TRAIN"])
        if "MAX_SIZE_TRAIN" in i: cfg.INPUT.MAX_SIZE_TRAIN = i["MAX_SIZE_TRAIN"]
        if "MIN_SIZE_TEST" in i: cfg.INPUT.MIN_SIZE_TEST = i["MIN_SIZE_TEST"]
        if "MAX_SIZE_TEST" in i: cfg.INPUT.MAX_SIZE_TEST = i["MAX_SIZE_TEST"]
        if "RANDOM_FLIP" in i: cfg.INPUT.RANDOM_FLIP = i["RANDOM_FLIP"]
        
    if "SOLVER" in config_data:
        s = config_data["SOLVER"]
        cfg.SOLVER.IMS_PER_BATCH = s.get("IMS_PER_BATCH", cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.BASE_LR = s.get("BASE_LR", cfg.SOLVER.BASE_LR)
        cfg.SOLVER.MAX_ITER = s.get("MAX_ITER", cfg.SOLVER.MAX_ITER)
        if "STEPS" in s: cfg.SOLVER.STEPS = tuple(s["STEPS"])
        cfg.SOLVER.WARMUP_ITERS = s.get("WARMUP_ITERS", cfg.SOLVER.WARMUP_ITERS)
        cfg.SOLVER.WEIGHT_DECAY = s.get("WEIGHT_DECAY", cfg.SOLVER.WEIGHT_DECAY)

    if "DATALOADER" in config_data:
        d = config_data["DATALOADER"]
        cfg.DATALOADER.NUM_WORKERS = d.get("NUM_WORKERS", cfg.DATALOADER.NUM_WORKERS)
        cfg.DATALOADER.SAMPLER_TRAIN = d.get("SAMPLER_TRAIN", cfg.DATALOADER.SAMPLER_TRAIN)
        cfg.DATALOADER.REPEAT_THRESHOLD = d.get("REPEAT_THRESHOLD", cfg.DATALOADER.REPEAT_THRESHOLD)

    if "TEST" in config_data:
        t = config_data["TEST"]
        cfg.TEST.EVAL_PERIOD = t.get("EVAL_PERIOD", cfg.TEST.EVAL_PERIOD)
        if "DETECTIONS_PER_IMAGE" in t:
            cfg.TEST.DETECTIONS_PER_IMAGE = t["DETECTIONS_PER_IMAGE"]

    return cfg
