"""
pipeline.py — Full chess recognition pipeline (Stage 2 + Stage 3)

Takes an image with known corners, warps it via Stage 2, classifies
all 64 squares via Stage 3, and outputs a FEN string.

Supports two modes:
  Default:    flipboth mapping + chesscog-trained models
  --chessred: normal mapping   + ChessReD fine-tuned models

Corner sources:
  --corners TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy   (manual, 8 floats)
  --gt IMAGE_ID                                   (ChessReD ground truth)

Usage examples:
  # ChessReD image with ground-truth corners
  python src/pipeline.py data/ChessReD/images/0/G000_IMG000.jpg --chessred --gt 0

  # ChessReD image with manual corners
  python src/pipeline.py photo.jpg --chessred --corners 100,200,900,180,920,850,110,870

  # Pre-warped 800x800 image (skip warp)
  python src/pipeline.py warped.png --warped

  # With debug output
  python src/pipeline.py photo.jpg --chessred --gt 42 --debug
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from stage2_warp import WarpConfig, rectify_board
from stage3_classify import (
    TwoStageClassifier, warp_chessboard_image, crop_square_piece,
    print_board, IMG_SIZE, MARGIN, SQUARE_SIZE
)


# ─────────────────────────────────────────────
# ChessReD classifier (normal mapping override)
# ─────────────────────────────────────────────

class ChessRedClassifier(TwoStageClassifier):
    """Uses normal mapping instead of flipboth for ChessReD images."""

    def classify_board(self, warped_image):
        board_grid = []
        conf_grid = []

        for fen_row in range(8):
            row_classes = []
            row_confs = []
            for fen_col in range(8):
                # Normal mapping — no flip
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


# ─────────────────────────────────────────────
# Corner loading helpers
# ─────────────────────────────────────────────

def parse_manual_corners(corners_str):
    """Parse 'TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy' into (4,2) array."""
    coords = list(map(float, corners_str.split(",")))
    if len(coords) != 8:
        raise ValueError(f"Need 8 corner values (TLx,TLy,...,BLx,BLy), got {len(coords)}")
    return np.array([
        [coords[0], coords[1]],  # TL
        [coords[2], coords[3]],  # TR
        [coords[4], coords[5]],  # BR
        [coords[6], coords[7]],  # BL
    ], dtype=np.float32)


def load_chessred_gt_corners(image_id, data_dir=None):
    """Load ground-truth corners for a ChessReD image by ID."""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "ChessReD")
    ann_path = os.path.join(data_dir, "annotations.json")

    with open(ann_path, "r") as f:
        ann = json.load(f)

    for corner_ann in ann["annotations"]["corners"]:
        if corner_ann["image_id"] == image_id:
            c = corner_ann["corners"]
            return np.array([
                c["top_left"],
                c["top_right"],
                c["bottom_right"],
                c["bottom_left"],
            ], dtype=np.float32)

    raise ValueError(f"No corners found for image_id={image_id}")


def load_chessred_gt_fen(image_id, data_dir=None):
    """Load ground-truth FEN for a ChessReD image (for comparison)."""
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "ChessReD")
    ann_path = os.path.join(data_dir, "annotations.json")

    with open(ann_path, "r") as f:
        ann = json.load(f)

    COLS = "abcdefgh"
    ROWS = "87654321"
    CAT_TO_FEN = {
        0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
        6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
        12: None,
    }

    grid = [[None] * 8 for _ in range(8)]
    for piece in ann["annotations"]["pieces"]:
        if piece["image_id"] == image_id:
            pos = piece["chessboard_position"]
            col = COLS.index(pos[0])
            row = ROWS.index(pos[1])
            fen_char = CAT_TO_FEN.get(piece["category_id"])
            if fen_char is not None:
                grid[row][col] = fen_char

    # Convert to FEN
    fen_rows = []
    for row in grid:
        fen_row = ""
        empty = 0
        for cell in row:
            if cell is None:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


# ─────────────────────────────────────────────
# Debug visualization
# ─────────────────────────────────────────────

def save_debug_images(image, corners, warped, output_dir):
    """Save intermediate images for debugging."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Original image with corners marked
    vis_orig = image.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip(corners.astype(int), labels):
        cv2.circle(vis_orig, (x, y), 10, (0, 255, 0), 3)
        cv2.putText(vis_orig, label, (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Draw quadrilateral
    for i in range(4):
        p1 = tuple(corners[i].astype(int))
        p2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(vis_orig, p1, p2, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "01_corners.jpg"), vis_orig)

    # 2. Warped image
    cv2.imwrite(os.path.join(output_dir, "02_warped.jpg"), warped)

    # 3. Warped image with grid overlay
    vis_grid = warped.copy()
    margin = int(MARGIN)
    sq = SQUARE_SIZE
    for i in range(9):
        x = margin + i * sq
        cv2.line(vis_grid, (x, margin), (x, margin + 8 * sq), (0, 255, 0), 2)
        y = margin + i * sq
        cv2.line(vis_grid, (margin, y), (margin + 8 * sq, y), (0, 255, 0), 2)
    # Labels
    for i in range(8):
        x = margin + i * sq + sq // 2 - 5
        cv2.putText(vis_grid, chr(ord('a') + i), (x, margin + 8 * sq + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y = margin + i * sq + sq // 2 + 5
        cv2.putText(vis_grid, str(8 - i), (margin - 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, "03_grid_overlay.jpg"), vis_grid)

    print(f"Debug images saved to {output_dir}/")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(image_path, corners, chessred_mode=False, warped_input=False,
                 debug=False, debug_dir="results/pipeline_debug", gt_image_id=None):
    """
    Run the full Stage 2 → Stage 3 pipeline.

    Args:
        image_path: Path to input image
        corners: (4,2) array [TL,TR,BR,BL] or None if warped_input=True
        chessred_mode: Use normal mapping + ChessReD models
        warped_input: If True, skip warp (image is already 800x800)
        debug: Save intermediate images
        debug_dir: Where to save debug images
        gt_image_id: ChessReD image ID for ground truth comparison

    Returns:
        fen: Predicted FEN string
        grid: 8x8 classification grid
        confs: 8x8 confidence grid
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)

    print(f"Image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"Mode: {'ChessReD (normal mapping)' if chessred_mode else 'Chesscog (flipboth mapping)'}")

    # ── STAGE 2: Warp ──
    if warped_input:
        warped = image
        print(f"Using pre-warped image ({warped.shape[1]}x{warped.shape[0]})")
    else:
        if corners is None:
            print("Error: Need corners (--corners or --gt) unless using --warped")
            sys.exit(1)

        print(f"Corners [TL,TR,BR,BL]:")
        labels = ["  TL", "  TR", "  BR", "  BL"]
        for label, (x, y) in zip(labels, corners):
            print(f"    {label}: ({x:.1f}, {y:.1f})")

        t0 = time.time()
        config = WarpConfig(output_size=800, square_size=100)
        warped = rectify_board(image, corners, config)
        t_warp = time.time() - t0
        print(f"Stage 2 warp: {warped.shape[1]}x{warped.shape[0]} ({t_warp*1000:.0f}ms)")

    # Save debug images
    if debug and not warped_input:
        save_debug_images(image, corners, warped, debug_dir)

    # ── STAGE 3: Classify ──
    t0 = time.time()

    if chessred_mode:
        # Load ChessReD fine-tuned models
        model_dir = os.path.join(PROJECT_ROOT, "models")
        classifier = ChessRedClassifier(model_dir=model_dir)

        # Swap in fine-tuned weights
        import torch
        occ_ft = os.path.join(model_dir, "occupancy_chessred.pth")
        piece_ft = os.path.join(model_dir, "piece_chessred.pth")

        if os.path.exists(occ_ft):
            occ_ckpt = torch.load(occ_ft, map_location=classifier.device, weights_only=True)
            classifier.occ_model.load_state_dict(occ_ckpt["model_state_dict"])
            classifier.occ_model.eval()
            print(f"Occupancy model: ChessReD fine-tuned (val_acc={occ_ckpt['val_acc']:.2f}%)")
        else:
            print(f"Warning: {occ_ft} not found, using chesscog occupancy model")

        if os.path.exists(piece_ft):
            piece_ckpt = torch.load(piece_ft, map_location=classifier.device, weights_only=True)
            classifier.piece_model.load_state_dict(piece_ckpt["model_state_dict"])
            classifier.piece_model.eval()
            print(f"Piece model: ChessReD fine-tuned (val_acc={piece_ckpt['val_acc']:.2f}%)")
        else:
            print(f"Warning: {piece_ft} not found, using chesscog piece model")
    else:
        classifier = TwoStageClassifier()

    fen, grid, confs = classifier.classify_board(warped)
    t_classify = time.time() - t0
    print(f"Stage 3 classify: {t_classify*1000:.0f}ms")

    # ── Output ──
    print_board(fen, grid, confs)

    # Compare to ground truth if available
    if gt_image_id is not None:
        try:
            gt_fen = load_chessred_gt_fen(gt_image_id)
            print(f"\nGround truth FEN: {gt_fen}")
            print(f"Predicted FEN:    {fen}")

            # Per-square comparison
            correct = 0
            total = 0
            gt_chars = []
            pred_chars = []
            for gt_c in gt_fen.split("/"):
                row_gt = []
                for c in gt_c:
                    if c.isdigit():
                        row_gt.extend(['.'] * int(c))
                    else:
                        row_gt.append(c)
                gt_chars.append(row_gt)
            for pred_c in fen.split("/"):
                row_pred = []
                for c in pred_c:
                    if c.isdigit():
                        row_pred.extend(['.'] * int(c))
                    else:
                        row_pred.append(c)
                pred_chars.append(row_pred)

            mismatches = []
            for r in range(8):
                for c in range(8):
                    total += 1
                    g = gt_chars[r][c] if r < len(gt_chars) and c < len(gt_chars[r]) else '.'
                    p = pred_chars[r][c] if r < len(pred_chars) and c < len(pred_chars[r]) else '.'
                    if g == p:
                        correct += 1
                    else:
                        sq_name = f"{chr(ord('a')+c)}{8-r}"
                        mismatches.append(f"  {sq_name}: GT={g} Pred={p}")

            acc = 100 * correct / total
            match = "EXACT MATCH" if correct == total else f"{correct}/{total} correct"
            print(f"\nResult: {match} ({acc:.1f}%)")
            if mismatches:
                print(f"Mismatches:")
                for m in mismatches:
                    print(m)

            print(f"\nGT view:   https://lichess.org/editor/{gt_fen}")
        except Exception as e:
            print(f"\nCould not load ground truth: {e}")

    return fen, grid, confs


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chess Vision Pipeline — Stage 2 (warp) + Stage 3 (classify)")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--corners", type=str, default=None,
                        help="Manual corners: TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy")
    parser.add_argument("--gt", type=int, default=None,
                        help="ChessReD image ID to load ground-truth corners")
    parser.add_argument("--chessred", action="store_true",
                        help="Use ChessReD mode (normal mapping + fine-tuned models)")
    parser.add_argument("--warped", action="store_true",
                        help="Image is already warped 800x800, skip Stage 2")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug images")
    parser.add_argument("--debug_dir", default="results/pipeline_debug",
                        help="Directory for debug images")
    args = parser.parse_args()

    # Resolve corners
    corners = None
    gt_image_id = None

    if args.warped:
        corners = None
    elif args.gt is not None:
        gt_image_id = args.gt
        print(f"Loading ChessReD ground-truth corners for image_id={gt_image_id}...")
        corners = load_chessred_gt_corners(gt_image_id)
    elif args.corners is not None:
        corners = parse_manual_corners(args.corners)
    else:
        print("Error: Provide corners via --corners, --gt, or use --warped")
        print("  Examples:")
        print("    python src/pipeline.py img.jpg --chessred --gt 42")
        print("    python src/pipeline.py img.jpg --corners 100,200,900,180,920,850,110,870")
        print("    python src/pipeline.py warped.png --warped")
        sys.exit(1)

    run_pipeline(
        image_path=args.image,
        corners=corners,
        chessred_mode=args.chessred,
        warped_input=args.warped,
        debug=args.debug,
        debug_dir=args.debug_dir,
        gt_image_id=gt_image_id,
    )


if __name__ == "__main__":
    main()