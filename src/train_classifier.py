"""
train_classifier.py — final version

Two-stage training:
  Stage A: Occupancy (empty vs occupied) — no label smoothing, light augmentation
  Stage B: Piece type (12-class) — EfficientNet-B0, shear augmentation, light smoothing

Place in: src/train_classifier.py
Usage:   python src/train_classifier.py
"""

import os
import sys
import time
import random
import shutil
from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# === CONFIG ===
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v6")
BALANCED_DIR = os.path.join(PROJECT_ROOT, "data", "balanced_final")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

BATCH_SIZE = 96
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

OCC_TRAIN_LIMIT = 15000
OCC_VAL_LIMIT = 1000
PIECE_TRAIN_LIMIT = 5000
PIECE_VAL_LIMIT = 500

EARLY_STOPPING_PATIENCE = 20
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Transforms
# ----------------------------
def get_occupancy_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((200, 100)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomAutocontrast(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((200, 100)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_piece_transforms(train: bool = True):
    """Moderate augmentation with critical shear for piece crops."""
    if train:
        return transforms.Compose([
            transforms.Resize((320, 160)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=(-10, 10, -5, 5),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((320, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ----------------------------
# Dataset balancing
# ----------------------------
def balance_dataset(src_dir: str, dst_dir: str, max_per_class: int) -> None:
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    random.seed(SEED)
    total = 0

    for class_name in sorted(os.listdir(src_dir)):
        class_src = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_src):
            continue

        class_dst = os.path.join(dst_dir, class_name)
        os.makedirs(class_dst, exist_ok=True)

        files = [f for f in os.listdir(class_src) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if len(files) > max_per_class:
            files = random.sample(files, max_per_class)

        for f in files:
            shutil.copy2(os.path.join(class_src, f), os.path.join(class_dst, f))

        print(f"  {class_name}: {len(files)}")
        total += len(files)

    print(f"  TOTAL: {total}\n")


# ----------------------------
# Model
# ----------------------------
def build_model(num_classes: int, model_name: str = "efficientnet_b0") -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


# ----------------------------
# Train / eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        out = model(images)
        loss = criterion(out, labels)
        loss_sum += loss.item() * images.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    true_labels, pred_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        out = model(images)
        pred_labels.extend(out.argmax(1).cpu().tolist())
        true_labels.extend(labels.tolist())
    return true_labels, pred_labels


def build_confusion_matrix(true_labels, pred_labels, num_classes):
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_labels, pred_labels):
        cm[t][p] += 1
    return cm


def save_confusion_outputs(stage_name, class_names, true_labels, pred_labels):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = stage_name.lower().replace(":", "").replace("(", "").replace(")", "")
    safe_name = safe_name.replace(" ", "_").replace("/", "_")

    cm = build_confusion_matrix(true_labels, pred_labels, len(class_names))
    csv_path = os.path.join(RESULTS_DIR, f"{safe_name}_confusion_matrix.csv")
    txt_path = os.path.join(RESULTS_DIR, f"{safe_name}_classification_report.txt")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(class_names[i] + "," + ",".join(str(x) for x in row) + "\n")

    per_class_lines = []
    worst_confusions = []

    for i, class_name in enumerate(class_names):
        tp = cm[i][i]
        row_sum = sum(cm[i])
        col_sum = sum(cm[r][i] for r in range(len(class_names)))
        recall = tp / row_sum if row_sum else 0.0
        precision = tp / col_sum if col_sum else 0.0

        mistakes = [(class_names[j], cm[i][j]) for j in range(len(class_names)) if j != i]
        mistakes.sort(key=lambda x: x[1], reverse=True)
        top_wrong = ", ".join(f"{name}:{count}" for name, count in mistakes[:3] if count > 0) or "none"
        worst_confusions.append((class_name, recall, top_wrong))
        per_class_lines.append(
            f"{class_name:15s} precision={precision*100:6.2f}% recall={recall*100:6.2f}% support={row_sum:4d} top_mistakes={top_wrong}"
        )

    worst_confusions.sort(key=lambda x: x[1])
    overall_acc = 100.0 * sum(cm[i][i] for i in range(len(class_names))) / max(1, len(true_labels))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Overall accuracy: {overall_acc:.2f}%\n\n")
        f.write("Per-class metrics:\n")
        f.write("\n".join(per_class_lines))
        f.write("\n\nWorst classes by recall:\n")
        for class_name, recall, top_wrong in worst_confusions[:5]:
            f.write(f"- {class_name}: recall={recall*100:.2f}% | most confused with {top_wrong}\n")

    print(f"\nSaved confusion matrix: {csv_path}")
    print(f"Saved report: {txt_path}")
    print("Worst classes by recall:")
    for class_name, recall, top_wrong in worst_confusions[:5]:
        print(f"  {class_name:15s} recall={recall*100:6.2f}% | top mistakes: {top_wrong}")


# ----------------------------
# Stage training
# ----------------------------
def train_stage(
    name: str,
    data_dir: str,
    save_path: str,
    num_classes: int,
    num_epochs: int,
    lr: float,
    transform_fn: Callable,
    model_name: str,
    label_smoothing: float = 0.0,
    save_confusion: bool = False,
) -> float:
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"{'='*70}")

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_fn(True))
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_fn(False))

    print(f"Classes: {train_ds.class_to_idx}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Model: {model_name}, LR: {lr}, Label smoothing: {label_smoothing}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = build_model(num_classes=num_classes, model_name=model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    class_names = [name for name, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10} | {'Time':>6}")
    print("-" * 80)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = evaluate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>5} | {tl:>10.4f} | {ta:>8.2f}% | {vl:>10.4f} | "
              f"{va:>8.2f}% | {current_lr:>10.6f} | {elapsed:>5.1f}s")

        if va > best_val_acc:
            best_val_acc = va
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": va,
                "class_to_idx": train_ds.class_to_idx,
                "num_classes": num_classes,
                "model_name": model_name,
            }, save_path)
            print(f"        ^ Saved best (val_acc={va:.2f}%)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"        ^ Early stopping after {EARLY_STOPPING_PATIENCE} stale epochs.")
            break

    print(f"\n{name} DONE — Best: {best_val_acc:.2f}% (epoch {best_epoch})")

    if save_confusion:
        checkpoint = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        true_labels, pred_labels = collect_predictions(model, val_loader)
        save_confusion_outputs(name, class_names, true_labels, pred_labels)

    return best_val_acc


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    set_seed(SEED)

    print("=" * 70)
    print("CHESS CLASSIFIER — FINAL RUN")
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Balance occupancy data
    print("\nBalancing occupancy data:")
    occ_src = os.path.join(DATA_DIR, "occupancy")
    occ_bal = os.path.join(BALANCED_DIR, "occupancy")
    for split in ["train", "val"]:
        print(f"  [{split}]")
        balance_dataset(
            os.path.join(occ_src, split),
            os.path.join(occ_bal, split),
            max_per_class=OCC_TRAIN_LIMIT if split == "train" else OCC_VAL_LIMIT,
        )

    # Balance piece data
    print("Balancing piece data:")
    piece_src = os.path.join(DATA_DIR, "pieces")
    piece_bal = os.path.join(BALANCED_DIR, "pieces")
    for split in ["train", "val"]:
        print(f"  [{split}]")
        balance_dataset(
            os.path.join(piece_src, split),
            os.path.join(piece_bal, split),
            max_per_class=PIECE_TRAIN_LIMIT if split == "train" else PIECE_VAL_LIMIT,
        )

    # Stage A: Occupancy — no label smoothing, simple augmentation
    occ_acc = train_stage(
        name="STAGE A: Occupancy (empty vs occupied)",
        data_dir=occ_bal,
        save_path=os.path.join(MODEL_DIR, "occupancy_classifier.pth"),
        num_classes=2,
        num_epochs=40,
        lr=5e-4,
        transform_fn=get_occupancy_transforms,
        model_name="resnet34",
        label_smoothing=0.1,
        save_confusion=True,
    )

    # Stage B: Pieces — EfficientNet-B0, light smoothing, long training
    piece_acc = train_stage(
        name="STAGE B: Piece Type (12-class)",
        data_dir=piece_bal,
        save_path=os.path.join(MODEL_DIR, "piece_classifier.pth"),
        num_classes=12,
        num_epochs=80,
        lr=5e-4,
        transform_fn=get_piece_transforms,
        model_name="efficientnet_b0",
        label_smoothing=0.05,
        save_confusion=True,
    )

    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"  Occupancy: {occ_acc:.2f}%")
    print(f"  Pieces:    {piece_acc:.2f}%")
    print(f"  Reports:   {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()