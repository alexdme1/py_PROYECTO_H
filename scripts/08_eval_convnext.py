"""
08_eval_convnext.py — Evaluación detallada del modelo ConvNeXt Tiny.

Carga un modelo entrenado y evalúa sobre el split de test.
Genera:
  - Accuracy global y por clase
  - Matriz de confusión (imagen)
  - Reporte de clasificación (precision, recall, F1)
  - Ejemplos de predicciones erróneas guardadas como imágenes

Uso:
    /home/servi2/Enviroments/main_venv/bin/python scripts/08_eval_convnext.py
"""

import os
import yaml
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =============================================================================
# CONFIGURACIÓN (MODIFICAR AQUÍ)
# =============================================================================

# Ruta al modelo a evaluar
MODEL_PATH = os.path.join("models", "convnext", "v1_baseline", "best_model.pth")

# Ruta al YAML de config (se usa para cargar transforms y dataset)
CONFIG_PATH = os.path.join("configs", "config_convnext.yaml")

# Número máximo de errores a guardar como imagen por clase
MAX_ERRORS_PER_CLASS = 10


# =============================================================================
# UTILIDADES
# =============================================================================

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder que ignora las subcarpetas excluidas."""
    def __init__(self, root, exclude=None, **kwargs):
        self._exclude = set(exclude or [])
        super().__init__(root, **kwargs)

    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        classes = [c for c in classes if c not in self._exclude]
        class_to_idx = {k: v for k, v in class_to_idx.items() if k not in self._exclude}
        return classes, class_to_idx


def build_val_transform(cfg):
    img_size = cfg["DATA"]["IMG_SIZE"]
    mean = cfg["DATA"]["MEAN"]
    std = cfg["DATA"]["STD"]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def unnormalize(tensor, mean, std):
    """Desnormaliza un tensor para visualización."""
    img = tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


# =============================================================================
# EVALUACIÓN
# =============================================================================

def main():
    print("=" * 60)
    print(" 08_EVAL_CONVNEXT — Evaluación detallada")
    print("=" * 60)

    cfg = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Device: {device}")

    # Dataset de test
    dataset_dir = cfg["DATA"]["DATASET_DIR"]
    exclude = cfg["MODEL"].get("EXCLUDE_CLASSES", [])
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.isdir(test_dir):
        print(f"[ERROR] No se encontró la carpeta de test: {test_dir}")
        return

    val_transform = build_val_transform(cfg)
    test_dataset = FilteredImageFolder(test_dir, exclude=exclude, transform=val_transform)
    test_class_names = test_dataset.classes  # Clases presentes en test (puede faltar alguna)

    # Leer las clases del modelo entrenado (fuente de verdad)
    model_dir = os.path.dirname(MODEL_PATH)
    class_names_path = os.path.join(model_dir, "class_names.txt")
    if os.path.exists(class_names_path):
        class_names = []
        with open(class_names_path) as f:
            for line in f:
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    class_names.append(parts[1])
        print(f"[*] Clases del modelo ({len(class_names)}): {class_names}")
    else:
        class_names = test_class_names
        print(f"[*] class_names.txt no encontrado, usando clases de test ({len(class_names)})")

    num_classes = len(class_names)

    # Reasignar los targets del test_dataset para que usen los índices del modelo
    # (test puede tener menos clases, con índices distintos)
    model_class_to_idx = {name: i for i, name in enumerate(class_names)}
    test_class_to_model_idx = {}
    for test_cls, test_idx in test_dataset.class_to_idx.items():
        if test_cls in model_class_to_idx:
            test_class_to_model_idx[test_idx] = model_class_to_idx[test_cls]

    # Remapear los targets internos del dataset
    remapped_samples = []
    for path, target in test_dataset.samples:
        if target in test_class_to_model_idx:
            remapped_samples.append((path, test_class_to_model_idx[target]))
    test_dataset.samples = remapped_samples
    test_dataset.targets = [s[1] for s in remapped_samples]

    print(f"[*] Test: {len(test_dataset)} imágenes (tras remap a clases del modelo)")

    test_loader = DataLoader(
        test_dataset, batch_size=cfg["DATA"]["BATCH_SIZE"],
        shuffle=False, num_workers=cfg["DATA"]["NUM_WORKERS"], pin_memory=True
    )

    # Cargar modelo
    model = timm.create_model(
        cfg["MODEL"]["NAME"],
        pretrained=False,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"[*] Modelo cargado desde: {MODEL_PATH}")

    # Directorio de salida (junto al modelo)
    model_dir = os.path.dirname(MODEL_PATH)
    eval_dir = os.path.join(model_dir, "eval_results")
    errors_dir = os.path.join(eval_dir, "errores")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)

    # ─── Inferencia completa ───
    all_preds = []
    all_labels = []
    all_probs = []
    error_samples = defaultdict(list)  # {true_class: [(img_tensor, pred_class, prob), ...]}

    mean = cfg["DATA"]["MEAN"]
    std = cfg["DATA"]["STD"]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            # Recoger errores para visualización
            for i in range(len(labels)):
                if predicted[i].item() != labels[i].item():
                    true_cls = class_names[labels[i].item()]
                    pred_cls = class_names[predicted[i].item()]
                    prob = probs[i, predicted[i].item()].item()
                    if len(error_samples[true_cls]) < MAX_ERRORS_PER_CLASS:
                        error_samples[true_cls].append((
                            images[i].cpu(), pred_cls, prob
                        ))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ─── Accuracy global ───
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n{'='*60}")
    print(f" ACCURACY GLOBAL: {accuracy:.2f}%")
    print(f"{'='*60}")

    # ─── Reporte por clase ───
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    print(f"\n{report}")

    # Guardar reporte como txt
    report_path = os.path.join(eval_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Accuracy global: {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"[+] Reporte guardado en: {report_path}")

    # ─── Matriz de confusión ───
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Matriz de Confusión — Test Acc: {accuracy:.2f}%")
    plt.tight_layout()

    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[+] Matriz de confusión guardada en: {cm_path}")

    # ─── Guardar imágenes de errores ───
    error_count = 0
    for true_cls, samples in error_samples.items():
        cls_dir = os.path.join(errors_dir, true_cls)
        os.makedirs(cls_dir, exist_ok=True)
        for idx, (img_tensor, pred_cls, prob) in enumerate(samples):
            img_np = unnormalize(img_tensor, mean, std)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Añadir texto con la predicción errónea
            label_text = f"Pred: {pred_cls} ({prob:.1%})"
            cv2.putText(img_bgr, label_text, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            out_path = os.path.join(cls_dir, f"error_{idx}_pred_{pred_cls}.png")
            cv2.imwrite(out_path, img_bgr)
            error_count += 1

    print(f"[+] {error_count} imágenes de errores guardadas en: {errors_dir}")

    # ─── Accuracy por clase ───
    print(f"\n{'='*60}")
    print(" ACCURACY POR CLASE")
    print(f"{'='*60}")
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() == 0:
            print(f"  {name:>15}: sin muestras en test")
            continue
        cls_acc = 100.0 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        total_cls = mask.sum()
        correct_cls = (all_preds[mask] == all_labels[mask]).sum()
        print(f"  {name:>15}: {cls_acc:6.2f}%  ({correct_cls}/{total_cls})")

    print(f"\n[+] Evaluación completada. Resultados en: {eval_dir}")


if __name__ == "__main__":
    main()
