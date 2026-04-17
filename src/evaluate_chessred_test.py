"""
evaluate_chessred_test.py - Evaluate fine-tuned classifier on ChessReD2K test set

Loops through ChessReD2K test images (306 images with corner annotations),
warps using ground-truth corners, classifies all 64 squares using the
fine-tuned models, and compares to ground truth.

Place in: src/evaluate_chessred_test.py
Usage:   python src/evaluate_chessred_test.py
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
CHESSRED_DIR = os.path.join(PROJECT_ROOT, "data", "ChessReD")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Use fine-tuned ChessReD models
FINETUNED_OCC = os.path.join(MODEL_DIR, "occupancy_chessred.pth")
FINETUNED_PIECE = os.path.join(MODEL_DIR, "piece_chessred.pth")

# ChessReD category_id -> FEN character
CHESSRED_CAT_TO_FEN = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',   # white
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',  # black
    12: None,  # empty
}

COLS = "abcdefgh"
ROWS = "87654321"


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


def chessred_corners_to_warp_order(corners_dict):
    """
    Convert ChessReD corner format to [TL, TR, BR, BL] for our warp.
    ChessReD labels from white's perspective:
      top_left=a8, top_right=h8, bottom_right=h1, bottom_left=a1
    """
    return [
        corners_dict["top_left"],
        corners_dict["top_right"],
        corners_dict["bottom_right"],
        corners_dict["bottom_left"],
    ]


def build_gt_grid(pieces_for_image):
    """
    Build 8x8 ground truth grid from ChessReD piece annotations.
    Returns grid[row][col] where row 0 = rank 8, col 0 = a-file.
    Each cell is a FEN character or None for empty.
    """
    grid = [[None] * 8 for _ in range(8)]

    for piece in pieces_for_image:
        pos = piece['chessboard_position']
        cat_id = piece['category_id']

        col = COLS.index(pos[0])
        row = ROWS.index(pos[1])

        fen_char = CHESSRED_CAT_TO_FEN.get(cat_id, None)
        if fen_char is not None:
            grid[row][col] = fen_char

    return grid


def grid_to_fen(grid):
    """Convert 8x8 grid to FEN board string."""
    fen_rows = []
    for row in grid:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


class ChessRedClassifier(TwoStageClassifier):
    """
    Subclass that overrides classify_board to use normal mapping
    instead of flipboth (ChessReD corners already place a8 at top-left).
    """

    def classify_board(self, warped_image):
        board_grid = []
        conf_grid = []

        for fen_row in range(8):
            row_classes = []
            row_confs = []
            for fen_col in range(8):
                # Normal mapping for ChessReD (no flip needed)
                crop_row = fen_row
                crop_col = fen_col

                square_crop = crop_square_piece(warped_image, crop_row, crop_col)
                class_name, fen_char, conf = self.classify_square(square_crop)
                row_classes.append((class_name, fen_char))
                row_confs.append(conf)

            board_grid.append(row_classes)
            conf_grid.append(row_confs)

        fen = self._grid_to_fen(board_grid)
        return fen, board_grid, conf_grid


def evaluate_chessred(split="test"):
    print("=" * 70)
    print(f"CHESSRED2K {split.upper()} SET EVALUATION")
    print("=" * 70)

    # Load annotations
    ann_path = os.path.join(CHESSRED_DIR, "annotations.json")
    if not os.path.exists(ann_path):
        print(f"ERROR: {ann_path} not found")
        sys.exit(1)

    print("Loading annotations...")
    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Build lookups
    images_lookup = {img['id']: img for img in data['images']}

    pieces_by_image = {}
    for piece in data['annotations']['pieces']:
        img_id = piece['image_id']
        if img_id not in pieces_by_image:
            pieces_by_image[img_id] = []
        pieces_by_image[img_id].append(piece)

    corners_by_image = {}
    for corner in data['annotations']['corners']:
        corners_by_image[corner['image_id']] = corner['corners']

    # Get ChessReD2K test split image IDs
    split_image_ids = data['splits']['chessred2k'][split]['image_ids']
    print(f"Found {len(split_image_ids)} images in chessred2k {split} split\n")

    # Check fine-tuned models exist, fall back to chesscog models
    if os.path.exists(FINETUNED_OCC) and os.path.exists(FINETUNED_PIECE):
        print("Using fine-tuned ChessReD models")
        classifier = ChessRedClassifier(model_dir=MODEL_DIR)
        # Manually load the fine-tuned weights
        import torch
        from torchvision import models
        import torch.nn as nn

        # Reload with fine-tuned weights
        occ_ckpt = torch.load(FINETUNED_OCC, map_location=classifier.device, weights_only=True)
        classifier.occ_model.load_state_dict(occ_ckpt["model_state_dict"])
        classifier.occ_model.eval()
        print(f"  Occupancy: {FINETUNED_OCC} (val_acc={occ_ckpt['val_acc']:.2f}%)")

        piece_ckpt = torch.load(FINETUNED_PIECE, map_location=classifier.device, weights_only=True)
        classifier.piece_model.load_state_dict(piece_ckpt["model_state_dict"])
        classifier.piece_model.eval()
        print(f"  Pieces: {FINETUNED_PIECE} (val_acc={piece_ckpt['val_acc']:.2f}%)")
    else:
        print("Fine-tuned models not found, using chesscog models")
        classifier = ChessRedClassifier()

    print()

    # Tracking
    total_squares = 0
    correct_squares = 0
    exact_match_boards = 0
    total_boards = 0
    errors = 0
    skipped = 0

    per_board_acc = []
    class_stats = {}
    piece_confusions = []

    t_start = time.time()

    for i, img_id in enumerate(split_image_ids):
        # Need both corners and pieces
        if img_id not in corners_by_image:
            skipped += 1
            continue

        if img_id not in images_lookup:
            errors += 1
            continue

        img_info = images_lookup[img_id]
        img_path = os.path.join(CHESSRED_DIR, img_info['path'])

        if not os.path.exists(img_path):
            errors += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            errors += 1
            continue

        # Warp
        try:
            corners_dict = corners_by_image[img_id]
            ordered_corners = chessred_corners_to_warp_order(corners_dict)
            warped = warp_chessboard_image(image, ordered_corners)
        except Exception:
            errors += 1
            continue

        # Build ground truth grid
        pieces = pieces_by_image.get(img_id, [])
        gt_grid = build_gt_grid(pieces)

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

        gt_fen = grid_to_fen(gt_grid)
        pred_fen_board = pred_fen.split(" ")[0]
        if gt_fen == pred_fen_board:
            exact_match_boards += 1

        total_boards += 1

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(split_image_ids):
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(split_image_ids)}] "
                  f"square_acc={100*correct_squares/total_squares:.2f}% "
                  f"exact_match={100*exact_match_boards/total_boards:.1f}% "
                  f"({elapsed:.1f}s)")

    elapsed_total = time.time() - t_start

    # === RESULTS ===
    print("\n" + "=" * 70)
    print(f"CHESSRED2K {split.upper()} SET EVALUATION RESULTS")
    print("=" * 70)

    overall_sq_acc = 100.0 * correct_squares / total_squares if total_squares > 0 else 0
    exact_match_pct = 100.0 * exact_match_boards / total_boards if total_boards > 0 else 0

    print(f"\nBoards evaluated:    {total_boards}")
    print(f"Skipped (no corners):{skipped}")
    print(f"Errors:              {errors}")
    print(f"Total squares:       {total_squares}")
    print(f"Correct squares:     {correct_squares}")
    print(f"Square accuracy:     {overall_sq_acc:.2f}%")
    print(f"Exact board match:   {exact_match_boards}/{total_boards} ({exact_match_pct:.1f}%)")
    print(f"Time:                {elapsed_total:.1f}s")

    if len(per_board_acc) > 0:
        acc_arr = np.array(per_board_acc) * 100
        print(f"\nPer-board accuracy distribution:")
        print(f"  Mean:   {np.mean(acc_arr):.2f}%")
        print(f"  Median: {np.median(acc_arr):.2f}%")
        print(f"  Min:    {np.min(acc_arr):.2f}%")
        print(f"  Max:    {np.max(acc_arr):.2f}%")
        print(f"  Std:    {np.std(acc_arr):.2f}%")

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

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, f"chessred_{split}_evaluation.txt")
    with open(report_path, "w") as f:
        f.write(f"CHESSRED2K {split.upper()} SET EVALUATION\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Boards evaluated:    {total_boards}\n")
        f.write(f"Square accuracy:     {overall_sq_acc:.2f}%\n")
        f.write(f"Exact board match:   {exact_match_boards}/{total_boards} ({exact_match_pct:.1f}%)\n\n")
        if len(per_board_acc) > 0:
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

    # Save per-board CSV
    csv_path = os.path.join(RESULTS_DIR, f"chessred_{split}_per_board.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "image_id", "square_accuracy", "exact_match"])
        for idx, (acc, img_id) in enumerate(zip(per_board_acc, split_image_ids)):
            exact = 1 if acc == 1.0 else 0
            writer.writerow([idx, img_id, f"{acc:.4f}", exact])
    print(f"Per-board CSV saved to {csv_path}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    evaluate_chessred(split)
