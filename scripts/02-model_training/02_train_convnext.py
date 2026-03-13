"""
07_train_convnext.py — Entrenamiento de ConvNeXt Tiny para clasificación de especies.

Lee la configuración desde configs/config_convnext.yaml.
Dataset en formato ImageFolder (subcarpetas = clases).
Lanza TensorBoard automáticamente al arrancar.

Uso:
    python scripts/07_train_convnext.py
"""

import os
import sys
import yaml
import time
import subprocess
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# CARGAR CONFIGURACIÓN YAML
# =============================================================================

CONFIG_PATH = os.path.join("configs", "config_convnext.yaml")

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


# =============================================================================
# TRANSFORMS (AUGMENTACIONES + PREPROCESSING)
# =============================================================================

def build_transforms(cfg):
    """
    Construye los pipelines de transformación para train y val/test.
    Las imágenes ya vienen a 224x224 de Roboflow, pero se fuerza el resize
    por seguridad y se aplican augmentaciones ligeras en runtime.
    """
    img_size = cfg["DATA"]["IMG_SIZE"]
    mean = cfg["DATA"]["MEAN"]
    std = cfg["DATA"]["STD"]
    aug_cfg = cfg["AUGMENTATIONS"]

    # --- TRAIN TRANSFORMS ---
    train_transforms_list = [
        transforms.Resize((img_size, img_size)),
    ]

    # Flip horizontal
    if aug_cfg.get("HORIZONTAL_FLIP", False):
        train_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # Color Jitter
    cj = aug_cfg.get("COLOR_JITTER", {})
    if cj.get("ENABLED", False):
        train_transforms_list.append(transforms.ColorJitter(
            brightness=cj.get("BRIGHTNESS", 0),
            contrast=cj.get("CONTRAST", 0),
            saturation=cj.get("SATURATION", 0),
            hue=cj.get("HUE", 0),
        ))

    train_transforms_list.append(transforms.ToTensor())
    train_transforms_list.append(transforms.Normalize(mean=mean, std=std))

    # Random Erasing (se aplica DESPUÉS de ToTensor)
    re = aug_cfg.get("RANDOM_ERASING", {})
    if re.get("ENABLED", False):
        train_transforms_list.append(transforms.RandomErasing(
            p=re.get("PROBABILITY", 0.1),
        ))

    train_transform = transforms.Compose(train_transforms_list)

    # --- VAL/TEST TRANSFORMS (sin augmentaciones, solo resize + normalizar) ---
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


# =============================================================================
# DATASETS Y DATALOADERS
# =============================================================================

class FilteredImageFolder(datasets.ImageFolder):
    """
    ImageFolder que ignora completamente las subcarpetas listadas en exclude.
    Las descarta ANTES de escanear archivos, evitando errores de carpeta vacía.
    """
    def __init__(self, root, exclude=None, **kwargs):
        self._exclude = set(exclude or [])
        super().__init__(root, **kwargs)

    def _is_excluded(self, class_name):
        """Excluye por nombre exacto o por prefijo (ej: 'borrar' descarta 'borrar' y 'borrar-xxx')."""
        for ex in self._exclude:
            if class_name == ex or class_name.startswith(ex + "-"):
                return True
        return False

    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        # Filtrar las clases excluidas (exacto + prefijo)
        classes = [c for c in classes if not self._is_excluded(c)]
        class_to_idx = {k: v for k, v in class_to_idx.items() if not self._is_excluded(k)}
        return classes, class_to_idx


def build_dataloaders(cfg, train_transform, val_transform):
    """
    Crea los DataLoaders a partir de ImageFolder.
    Excluye las clases definidas en MODEL.EXCLUDE_CLASSES del YAML.
    """
    dataset_dir = cfg["DATA"]["DATASET_DIR"]
    batch_size = cfg["DATA"]["BATCH_SIZE"]
    num_workers = cfg["DATA"]["NUM_WORKERS"]
    exclude = cfg["MODEL"].get("EXCLUDE_CLASSES", [])

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")

    train_dataset = FilteredImageFolder(train_dir, exclude=exclude, transform=train_transform)
    val_dataset = FilteredImageFolder(val_dir, exclude=exclude, transform=val_transform)

    test_dataset = None
    if os.path.isdir(test_dir):
        test_dataset = FilteredImageFolder(test_dir, exclude=exclude, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    if exclude:
        print(f"[*] Clases EXCLUIDAS del entrenamiento: {exclude}")

    return train_loader, val_loader, test_loader, train_dataset.classes


# =============================================================================
# MODELO
# =============================================================================

def build_model(cfg):
    """
    Crea el modelo ConvNeXt Tiny con timm.
    Si FREEZE_BACKBONE es True, congela todas las capas excepto la cabeza final.
    """
    model = timm.create_model(
        cfg["MODEL"]["NAME"],
        pretrained=cfg["MODEL"]["PRETRAINED"],
        num_classes=cfg["MODEL"]["NUM_CLASSES"],
    )

    if cfg["MODEL"].get("FREEZE_BACKBONE", False):
        # Congelar todo excepto la cabeza de clasificación
        for name, param in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False
        print("[*] Backbone CONGELADO. Solo se entrena la cabeza de clasificación.")
    else:
        print("[*] Fine-tuning COMPLETO de toda la red.")

    return model


# =============================================================================
# SCHEDULER
# =============================================================================

def build_scheduler(optimizer, cfg, steps_per_epoch):
    """
    Construye el scheduler de Learning Rate.
    Soporta Cosine Annealing con warmup lineal.
    """
    solver = cfg["SOLVER"]
    total_epochs = solver["EPOCHS"]
    warmup_epochs = solver.get("WARMUP_EPOCHS", 0)
    min_lr = solver.get("MIN_LR", 1e-6)

    # Cosine Annealing sobre el total de steps (sin contar warmup)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs) * steps_per_epoch,
        eta_min=min_lr,
    )

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * steps_per_epoch,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs * steps_per_epoch],
        )
    else:
        scheduler = main_scheduler

    return scheduler


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, writer, epoch, global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log cada 20 batches
        if (batch_idx + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        # TensorBoard: loss por step
        writer.add_scalar("Train/Loss_step", loss.item(), global_step)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)
        global_step += 1

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    writer.add_scalar("Train/Loss_epoch", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, writer, epoch, prefix="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    writer.add_scalar(f"{prefix}/Loss", epoch_loss, epoch)
    writer.add_scalar(f"{prefix}/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" 07_TRAIN_CONVNEXT — Clasificación de especies")
    print("=" * 60)

    # Cargar config
    cfg = load_config(CONFIG_PATH)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Transforms
    train_transform, val_transform = build_transforms(cfg)

    # Dataloaders
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg, train_transform, val_transform
    )
    print(f"\n[*] Clases detectadas ({len(class_names)}): {class_names}")
    print(f"[*] Train: {len(train_loader.dataset)} imgs | "
          f"Val: {len(val_loader.dataset)} imgs | "
          f"Test: {len(test_loader.dataset) if test_loader else 0} imgs")

    # Verificar que NUM_CLASSES coincide
    if cfg["MODEL"]["NUM_CLASSES"] != len(class_names):
        print(f"\n[WARNING] NUM_CLASSES en YAML ({cfg['MODEL']['NUM_CLASSES']}) "
              f"!= clases detectadas ({len(class_names)}). Ajustando automáticamente.")
        cfg["MODEL"]["NUM_CLASSES"] = len(class_names)

    # Modelo
    model = build_model(cfg)
    model = model.to(device)

    # Ruta de salida
    run_name = cfg["OUTPUT"]["RUN_NAME"]
    output_dir = os.path.join(cfg["OUTPUT"]["DIR_BASE"], run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Guardar copia de la config usada junto al modelo
    with open(os.path.join(output_dir, "config_used.yaml"), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Guardar mapeo de clases
    with open(os.path.join(output_dir, "class_names.txt"), 'w') as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")

    # TensorBoard
    tb_dir = cfg["OUTPUT"]["DIR_BASE"]  # Para poder comparar runs
    writer = SummaryWriter(log_dir=output_dir)

    print(f"\n[*] Lanzando TensorBoard en el puerto 6007 escuchando: {tb_dir}")
    print(f"    -> http://localhost:6007")
    subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", tb_dir, "--port", "6007"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Optimizador
    solver_cfg = cfg["SOLVER"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=solver_cfg["BASE_LR"],
        weight_decay=solver_cfg["WEIGHT_DECAY"],
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    # Loss con label smoothing
    criterion = nn.CrossEntropyLoss(
        label_smoothing=solver_cfg.get("LABEL_SMOOTHING", 0.0)
    )

    # Entrenamiento
    total_epochs = solver_cfg["EPOCHS"]
    eval_period = cfg["EVAL"]["EVAL_PERIOD"]
    save_every = cfg["OUTPUT"]["SAVE_EVERY_N_EPOCHS"]
    save_best = cfg["EVAL"]["SAVE_BEST"]
    best_val_acc = 0.0
    global_step = 0

    print(f"\n[*] Comenzando entrenamiento: {total_epochs} épocas")
    print(f"    -> Modelo: {cfg['MODEL']['NAME']}")
    print(f"    -> Output: {output_dir}")
    print("-" * 60)

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, writer, epoch, global_step
        )

        elapsed = time.time() - t0
        print(f"Epoch [{epoch}/{total_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Tiempo: {elapsed:.1f}s")

        # Evaluación en validación
        if epoch % eval_period == 0:
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, writer, epoch, prefix="Val"
            )
            print(f"  -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Guardar mejor modelo
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  -> ⭐ Nuevo mejor modelo guardado (Val Acc: {val_acc:.2f}%)")

        # Checkpoint periódico
        if epoch % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc if epoch % eval_period == 0 else None,
            }, ckpt_path)
            print(f"  -> Checkpoint guardado: {ckpt_path}")

    # Guardar modelo final
    final_path = os.path.join(output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[+] Modelo final guardado: {final_path}")

    # Evaluación final en test si existe
    if test_loader is not None:
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, writer, total_epochs, prefix="Test"
        )
        print(f"[+] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    writer.close()
    print(f"\n[+] Mejor Val Acc durante entrenamiento: {best_val_acc:.2f}%")
    print(f"[+] Entrenamiento completado. Logs en: {output_dir}")


if __name__ == "__main__":
    main()
