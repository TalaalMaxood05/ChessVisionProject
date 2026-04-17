"""
finetune_chessred.py - Fine-tune chesscog models on ChessReD real-world data

Loads the chesscog-trained occupancy and piece classifiers and fine-tunes
them on ChessReD2K crops. Uses a lower learning rate and stronger
augmentation to adapt from synthetic to real images.

Place in: src/finetune_chessred.py
Usage:   python src/finetune_chessred.py
"""

import os
import sys
import time
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# === CONFIG ===
CHESSRED_DATA = os.path.join(PROJECT_ROOT, "data", "chessred_processed")
BALANCED_DIR = os.path.join(PROJECT_ROOT, "data", "chessred_balanced")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Pre-trained chesscog model paths (these get loaded and fine-tuned)
PRETRAINED_OCC = os.path.join(MODEL_DIR, "occupancy_classifier.pth")
PRETRAINED_PIECE = os.path.join(MODEL_DIR, "piece_classifier.pth")

# Fine-tuned model output paths
FINETUNED_OCC = os.path.join(MODEL_DIR, "occupancy_chessred.pth")
FINETUNED_PIECE = os.path.join(MODEL_DIR, "piece_chessred.pth")

BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Lower limits since ChessReD2K is smaller (~1400 train images = ~90K squares)
OCC_TRAIN_LIMIT = 10000
OCC_VAL_LIMIT = 1000
PIECE_TRAIN_LIMIT = 3000
PIECE_VAL_LIMIT = 300

EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Transforms — stronger augmentation for real-world domain shift
# ----------------------------
def get_occupancy_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((200, 100)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.08),
            transforms.RandomAutocontrast(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((200, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_piece_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((320, 160)),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=(-12, 12, -6, 6),
            ),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.08),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((320, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ----------------------------
# Balancing
# ----------------------------
def balance_dataset(src_dir, dst_dir, max_per_class):
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
# Model building (matches train_classifier.py exactly)
# ----------------------------
def build_model(num_classes, model_name="efficientnet_b0"):
    model_name = model_name.lower()

    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def load_pretrained(checkpoint_path, device):
    """Load a chesscog-trained checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = ckpt.get("model_name", "resnet34")
    num_classes = ckpt.get("num_classes", 2)
    model = build_model(num_classes, model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {model_name} from {checkpoint_path} (val_acc={ckpt['val_acc']:.2f}%)")
    return model, ckpt


# ----------------------------
# Training
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


def finetune_stage(name, pretrained_path, data_dir, save_path,
                   num_epochs, lr, transform_fn, label_smoothing=0.0):
    print(f"\n{'='*70}")
    print(f"FINE-TUNING: {name}")
    print(f"{'='*70}")

    # Load pretrained model
    model, ckpt = load_pretrained(pretrained_path, DEVICE)
    num_classes = ckpt.get("num_classes", 2)
    model_name = ckpt.get("model_name", "resnet34")
    model = model.to(DEVICE)

    # Load data
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_fn(True))
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_fn(False))

    print(f"  Classes: {train_ds.class_to_idx}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Model: {model_name}, LR: {lr}, Label smoothing: {label_smoothing}")

    # Verify class mapping matches pretrained model
    pretrained_classes = ckpt.get("class_to_idx", {})
    if pretrained_classes and pretrained_classes != train_ds.class_to_idx:
        print(f"  WARNING: Class mapping mismatch!")
        print(f"    Pretrained: {pretrained_classes}")
        print(f"    ChessReD:   {train_ds.class_to_idx}")
        print(f"    Proceeding anyway — classes should be in same alphabetical order")

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

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    class_names = [n for n, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    # Evaluate pretrained model on ChessReD val BEFORE fine-tuning
    pretrained_vl, pretrained_va = evaluate(model, val_loader, criterion)
    print(f"\n  Pre-finetune val accuracy on ChessReD: {pretrained_va:.2f}%")

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
                "finetuned_from": pretrained_path,
                "pretrained_val_acc_on_chessred": pretrained_va,
            }, save_path)
            print(f"        ^ Saved best (val_acc={va:.2f}%)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"        ^ Early stopping after {EARLY_STOPPING_PATIENCE} stale epochs.")
            break

    print(f"\n{name} DONE")
    print(f"  Before fine-tuning: {pretrained_va:.2f}%")
    print(f"  After fine-tuning:  {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Improvement:        {best_val_acc - pretrained_va:+.2f}%")

    return pretrained_va, best_val_acc


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)

    print("=" * 70)
    print("CHESS CLASSIFIER — CHESSRED FINE-TUNING")
    print(f"Device: {DEVICE}")
    print(f"ChessReD data: {CHESSRED_DATA}")
    print("=" * 70)

    # Check pretrained models exist
    for path in [PRETRAINED_OCC, PRETRAINED_PIECE]:
        if not os.path.exists(path):
            print(f"ERROR: Pretrained model not found: {path}")
            print("Train on chesscog first with train_classifier.py")
            return

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Balance ChessReD data
    print("\nBalancing occupancy data:")
    occ_src = os.path.join(CHESSRED_DATA, "occupancy")
    occ_bal = os.path.join(BALANCED_DIR, "occupancy")
    for split in ["train", "val"]:
        print(f"  [{split}]")
        balance_dataset(
            os.path.join(occ_src, split),
            os.path.join(occ_bal, split),
            max_per_class=OCC_TRAIN_LIMIT if split == "train" else OCC_VAL_LIMIT,
        )

    print("Balancing piece data:")
    piece_src = os.path.join(CHESSRED_DATA, "pieces")
    piece_bal = os.path.join(BALANCED_DIR, "pieces")
    for split in ["train", "val"]:
        print(f"  [{split}]")
        balance_dataset(
            os.path.join(piece_src, split),
            os.path.join(piece_bal, split),
            max_per_class=PIECE_TRAIN_LIMIT if split == "train" else PIECE_VAL_LIMIT,
        )

    # Fine-tune Stage A: Occupancy
    occ_before, occ_after = finetune_stage(
        name="STAGE A: Occupancy (ChessReD fine-tune)",
        pretrained_path=PRETRAINED_OCC,
        data_dir=occ_bal,
        save_path=FINETUNED_OCC,
        num_epochs=30,
        lr=1e-4,           # 5x lower than original training
        transform_fn=get_occupancy_transforms,
        label_smoothing=0.05,
    )

    # Fine-tune Stage B: Pieces
    piece_before, piece_after = finetune_stage(
        name="STAGE B: Piece Type (ChessReD fine-tune)",
        pretrained_path=PRETRAINED_PIECE,
        data_dir=piece_bal,
        save_path=FINETUNED_PIECE,
        num_epochs=40,
        lr=5e-5,            # 10x lower than original training
        transform_fn=get_piece_transforms,
        label_smoothing=0.05,
    )

    print(f"\n{'='*70}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Occupancy:  {occ_before:.2f}% -> {occ_after:.2f}% ({occ_after-occ_before:+.2f}%)")
    print(f"  Pieces:     {piece_before:.2f}% -> {piece_after:.2f}% ({piece_after-piece_before:+.2f}%)")
    print(f"\n  Fine-tuned models saved to:")
    print(f"    {FINETUNED_OCC}")
    print(f"    {FINETUNED_PIECE}")
    print(f"\n  To evaluate on ChessReD test set:")
    print(f"    python src/evaluate_chesscog_test.py  (modify to use chessred models + data)")


if __name__ == "__main__":
    main()
