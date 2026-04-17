"""
evaluate_chesscog_test.py - Evaluate two-stage classifier on chesscog test set

Loops through all test images, warps using ground-truth corners,
classifies all 64 squares, and compares predicted FEN to ground truth.

Reports:
  - Per-square accuracy (out of 64*N squares)
  - Per-board exact match accuracy
  - Per-board square accuracy distribution
  - Per-piece-type accuracy breakdown

Place in: src/evaluate_chesscog_test.py
Usage:   python src/evaluate_chesscog_test.py
"""

import os
import sys
import json
import time
import csv
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from stage3_classify import TwoStageClassifier, warp_chessboard_image, crop_square_piece

# === CONFIG ===
CHESSCOG_DIR = os.path.join(PROJECT_ROOT, "data", "chesscog", "render")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def ordered_json_corners(corners):
    """JSON order is [h8, h1, a1, a8]. Convert to [TL, TR, BR, BL] = [a8, h8, h1, a1]."""
    return [corners[3], corners[0], corners[1], corners[2]]


def parse_fen_to_grid(fen):
    """Parse FEN string into 8x8 grid of characters (None for empty)."""
    rows = fen.split(" ")[0].split("/")
    grid = []
    for row_str in rows:
        row = []
        for ch in row_str:
            if ch.isdigit():
                row.extend([None] * int(ch))
            else:
                row.append(ch)
        if len(row) != 8:
            raise ValueError(f"Row has {len(row)} squares")
        grid.append(row)
    return grid


def fen_char_to_class(fen_char):
    """Convert FEN character to class name."""
    mapping = {
        'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop',
        'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
        'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop',
        'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king',
    }
    if fen_char is None:
        return "empty"
    return mapping.get(fen_char, "unknown")


def evaluate_test_set(split="test"):
    split_dir = os.path.join(CHESSCOG_DIR, split)
    if not os.path.exists(split_dir):
        print(f"ERROR: {split_dir} not found")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".png")])
    print(f"Found {len(image_files)} test images in {split_dir}\n")

    # Load classifier
    classifier = TwoStageClassifier()
    print()

    # Tracking
    total_squares = 0
    correct_squares = 0
    exact_match_boards = 0
    total_boards = 0
    errors = 0

    per_board_acc = []  # square accuracy per board

    # Per-class tracking: {class_name: {"correct": N, "total": N}}
    class_stats = {}

    # Confusion tracking for piece types
    piece_confusions = []  # list of (true_class, pred_class)

    t_start = time.time()

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(split_dir, img_file)
        json_path = os.path.join(split_dir, img_file.replace(".png", ".json"))

        if not os.path.exists(json_path):
            errors += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            errors += 1
            continue

        with open(json_path, "r") as f:
            annotation = json.load(f)

        fen = annotation.get("fen")
        corners = annotation.get("corners")

        if fen is None or corners is None:
            errors += 1
            continue

        try:
            gt_grid = parse_fen_to_grid(fen)
            warped = warp_chessboard_image(image, ordered_json_corners(corners))
        except Exception as e:
            errors += 1
            continue

        # Classify
        pred_fen, pred_grid, conf_grid = classifier.classify_board(warped)

        # Compare square by square
        board_correct = 0
        for row in range(8):
            for col in range(8):
                gt_char = gt_grid[row][col]
                _, pred_char = pred_grid[row][col]

                gt_class = fen_char_to_class(gt_char)
                pred_class = fen_char_to_class(pred_char)

                # Track per-class stats
                if gt_class not in class_stats:
                    class_stats[gt_class] = {"correct": 0, "total": 0}
                class_stats[gt_class]["total"] += 1

                if gt_char == pred_char:
                    correct_squares += 1
                    board_correct += 1
                    class_stats[gt_class]["correct"] += 1
                else:
                    piece_confusions.append((gt_class, pred_class))

                total_squares += 1

        board_acc = board_correct / 64.0
        per_board_acc.append(board_acc)

        gt_fen_board = fen.split(" ")[0]
        pred_fen_board = pred_fen.split(" ")[0]
        if gt_fen_board == pred_fen_board:
            exact_match_boards += 1

        total_boards += 1

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(image_files)}] "
                  f"square_acc={100*correct_squares/total_squares:.2f}% "
                  f"exact_match={100*exact_match_boards/total_boards:.1f}% "
                  f"({elapsed:.1f}s)")

    elapsed_total = time.time() - t_start

    # === RESULTS ===
    print("\n" + "=" * 70)
    print("CHESSCOG TEST SET EVALUATION RESULTS")
    print("=" * 70)

    overall_sq_acc = 100.0 * correct_squares / total_squares if total_squares > 0 else 0
    exact_match_pct = 100.0 * exact_match_boards / total_boards if total_boards > 0 else 0

    print(f"\nBoards evaluated:    {total_boards}")
    print(f"Errors/skipped:      {errors}")
    print(f"Total squares:       {total_squares}")
    print(f"Correct squares:     {correct_squares}")
    print(f"Square accuracy:     {overall_sq_acc:.2f}%")
    print(f"Exact board match:   {exact_match_boards}/{total_boards} ({exact_match_pct:.1f}%)")
    print(f"Time:                {elapsed_total:.1f}s")

    # Per-board accuracy stats
    acc_arr = np.array(per_board_acc) * 100
    print(f"\nPer-board accuracy distribution:")
    print(f"  Mean:   {np.mean(acc_arr):.2f}%")
    print(f"  Median: {np.median(acc_arr):.2f}%")
    print(f"  Min:    {np.min(acc_arr):.2f}%")
    print(f"  Max:    {np.max(acc_arr):.2f}%")
    print(f"  Std:    {np.std(acc_arr):.2f}%")

    # Boards with 100% accuracy
    perfect = sum(1 for a in per_board_acc if a == 1.0)
    above95 = sum(1 for a in per_board_acc if a >= 0.95)
    above90 = sum(1 for a in per_board_acc if a >= 0.90)
    print(f"\n  100% squares correct: {perfect}/{total_boards} ({100*perfect/total_boards:.1f}%)")
    print(f"  >=95% squares correct: {above95}/{total_boards} ({100*above95/total_boards:.1f}%)")
    print(f"  >=90% squares correct: {above90}/{total_boards} ({100*above90/total_boards:.1f}%)")

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    print(f"  {'Class':20s} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        acc = 100.0 * s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {cls:20s} {s['correct']:>8} {s['total']:>8} {acc:>9.2f}%")

    # Top confusions
    if piece_confusions:
        print(f"\nTop confusions (true -> predicted):")
        from collections import Counter
        conf_counts = Counter(piece_confusions)
        for (true_cls, pred_cls), count in conf_counts.most_common(15):
            print(f"  {true_cls:20s} -> {pred_cls:20s}  ({count} times)")

    # Save results to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"chesscog_{split}_evaluation.txt")
    with open(report_path, "w") as f:
        f.write(f"CHESSCOG {split.upper()} SET EVALUATION\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Boards evaluated:    {total_boards}\n")
        f.write(f"Square accuracy:     {overall_sq_acc:.2f}%\n")
        f.write(f"Exact board match:   {exact_match_boards}/{total_boards} ({exact_match_pct:.1f}%)\n\n")
        f.write(f"Per-board accuracy: mean={np.mean(acc_arr):.2f}% median={np.median(acc_arr):.2f}% "
                f"min={np.min(acc_arr):.2f}% max={np.max(acc_arr):.2f}%\n\n")
        f.write(f"Per-class:\n")
        for cls in sorted(class_stats.keys()):
            s = class_stats[cls]
            acc = 100.0 * s["correct"] / s["total"] if s["total"] > 0 else 0
            f.write(f"  {cls:20s} {s['correct']:>6}/{s['total']:<6} {acc:.2f}%\n")
        if piece_confusions:
            f.write(f"\nTop confusions:\n")
            from collections import Counter
            conf_counts = Counter(piece_confusions)
            for (true_cls, pred_cls), count in conf_counts.most_common(15):
                f.write(f"  {true_cls:20s} -> {pred_cls:20s}  ({count})\n")

    print(f"\nReport saved to {report_path}")

    # Save per-board CSV for further analysis
    csv_path = os.path.join(RESULTS_DIR, f"chesscog_{split}_per_board.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "square_accuracy", "exact_match"])
        for idx, acc in enumerate(per_board_acc):
            exact = 1 if acc == 1.0 else 0
            writer.writerow([idx, f"{acc:.4f}", exact])
    print(f"Per-board CSV saved to {csv_path}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    evaluate_test_set(split)
