"""
chess_dataset.py
Place this file in: src/chess_dataset.py
"""

import os
import random
import shutil
from torchvision import datasets, transforms

CLASS_NAMES = [
    "empty",
    "white_pawn", "white_knight", "white_bishop",
    "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop",
    "black_rook", "black_queen", "black_king"
]

NUM_CLASSES = 13
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

FEN_TO_IDX = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(degrees=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_balanced_subset(src_dir, dst_dir, max_per_class):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    random.seed(42)
    total = 0

    for class_name in sorted(os.listdir(src_dir)):
        class_src = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_src):
            continue

        class_dst = os.path.join(dst_dir, class_name)
        os.makedirs(class_dst, exist_ok=True)

        files = [f for f in os.listdir(class_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(files) > max_per_class:
            files = random.sample(files, max_per_class)

        for f in files:
            shutil.copy2(os.path.join(class_src, f), os.path.join(class_dst, f))

        print(f"  {class_name}: {len(files)} images")
        total += len(files)

    print(f"  TOTAL: {total} images in {dst_dir}\n")