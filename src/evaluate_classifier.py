"""
evaluate_classifier.py

Load the trained model and evaluate on the test set.
Prints per-class accuracy and overall accuracy.

Place this file in: src/evaluate_classifier.py

Usage:
    python src/evaluate_classifier.py
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.chess_dataset import (
    ChessSquareDataset, get_val_transforms,
    CLASS_NAMES, NUM_CLASSES
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "piece_classifier.pth")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 60)
    print("CHESS PIECE CLASSIFIER EVALUATION")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Did you run: python src/train_classifier.py ?")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Model was trained to val_acc={checkpoint['val_acc']:.2f}%")

    # Load test data
    test_dataset = ChessSquareDataset(TEST_DIR, transform=get_val_transforms())
    if len(test_dataset) == 0:
        print("ERROR: No test data found!")
        return

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Run predictions
    all_preds = []
    all_labels = []

    print(f"\nRunning predictions on {len(test_dataset)} test samples...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")

    # Per-class accuracy
    print(f"\n{'Class':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = all_labels == class_idx
        if mask.sum() == 0:
            continue
        class_correct = (all_preds[mask] == class_idx).sum()
        class_total = mask.sum()
        class_acc = 100.0 * class_correct / class_total
        print(f"{class_name:<20} {class_correct:>8} {class_total:>8} {class_acc:>9.2f}%")

    # Most confused pairs
    print(f"\nMost common misclassifications:")
    from collections import Counter
    mistakes = Counter()
    for true, pred in zip(all_labels, all_preds):
        if true != pred:
            mistakes[(CLASS_NAMES[true], CLASS_NAMES[pred])] += 1

    for (true_name, pred_name), count in mistakes.most_common(10):
        print(f"  {true_name} -> {pred_name}: {count} times")


if __name__ == "__main__":
    main()
