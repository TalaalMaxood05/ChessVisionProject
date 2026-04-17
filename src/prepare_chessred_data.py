"""
prepare_chessred_data.py - Prepare ChessReD2K for fine-tuning

Reads ChessReD annotations.json, filters to images with corner annotations
(ChessReD2K subset), warps each image using ground-truth corners, crops
all 64 squares using the same v6 crop logic as chesscog, and saves
labeled crops in ImageFolder format for fine-tuning.

Place in: src/prepare_chessred_data.py
Usage:   python src/prepare_chessred_data.py

Before running: download ChessReD dataset to data/ChessReD/
  - Should contain: annotations.json and images/ folder
"""

import os
import json
import cv2
import numpy as np
from collections import Counter

# === CHESSCOG GEOMETRY (must match prepare_chesscog_data.py v6) ===
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE        # 400
IMG_SIZE = BOARD_SIZE * 2           # 800
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2  # 200

# Tighter crop params (v6)
MIN_HEIGHT_INCREASE = 0.6
MAX_HEIGHT_INCREASE = 2.0
MIN_WIDTH_INCREASE = 0.05
MAX_WIDTH_INCREASE = 0.45

OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)    # 72
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)  # 150

# === CONFIG ===
CHESSRED_DIR = os.path.join("data", "ChessReD")
OUTPUT_DIR = os.path.join("data", "chessred_processed")

# ChessReD category_id -> our class names
# From their categories: 0=white-pawn, 1=white-rook, 2=white-knight,
# 3=white-bishop, 4=white-queen, 5=white-king, 6=black-pawn, 7=black-rook,
# 8=black-knight, 9=black-bishop, 10=black-queen, 11=black-king, 12=empty
CHESSRED_CAT_TO_CLASS = {
    0: 'white_pawn', 1: 'white_rook', 2: 'white_knight',
    3: 'white_bishop', 4: 'white_queen', 5: 'white_king',
    6: 'black_pawn', 7: 'black_rook', 8: 'black_knight',
    9: 'black_bishop', 10: 'black_queen', 11: 'black_king',
    12: 'empty',
}

# Chess coordinate helpers
COLS = "abcdefgh"
ROWS = "87654321"


def warp_chessboard_image(img, corners):
    """Warp image using corners in [TL, TR, BR, BL] order."""
    src_points = np.array(corners, dtype=np.float32)
    dst_points = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, H, (IMG_SIZE, IMG_SIZE))


def crop_square_piece(img, row, col):
    """Tighter centered crop for piece classification. Matches v6 exactly."""
    height_increase = MIN_HEIGHT_INCREASE + \
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)

    left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
    right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)

    x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
    y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=img.dtype)

    if col < 4:
        cropped = cv2.flip(cropped, 1)

    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped.dtype)
    h, w = cropped.shape[:2]
    h = min(h, OUT_HEIGHT)
    w = min(w, OUT_WIDTH)

    y_offset = max(0, (OUT_HEIGHT - h) // 2)
    x_offset = max(0, (OUT_WIDTH - w) // 2)
    result[y_offset:y_offset + h, x_offset:x_offset + w] = cropped[:h, :w]
    return result


def chessred_corners_to_warp_order(corners_dict):
    """
    Convert ChessReD corner format to [TL, TR, BR, BL] for our warp.

    ChessReD labels corners from white's perspective:
      top_left = a8 corner (TL from white's view)
      top_right = h8 corner (TR from white's view)
      bottom_right = h1 corner (BR from white's view)
      bottom_left = a1 corner (BL from white's view)

    Our warp expects [TL, TR, BR, BL] which maps to board corners
    [a8, h8, h1, a1]. This matches directly.
    """
    return [
        corners_dict["top_left"],       # TL = a8
        corners_dict["top_right"],      # TR = h8
        corners_dict["bottom_right"],   # BR = h1
        corners_dict["bottom_left"],    # BL = a1
    ]


def build_board_grid(pieces_for_image, cat_id_to_name):
    """
    Build 8x8 grid from ChessReD piece annotations for a single image.

    Returns grid[row][col] where row 0 = rank 8, col 0 = a-file.
    Each cell is a class name string ('white_pawn', 'empty', etc.)
    """
    grid = [['empty'] * 8 for _ in range(8)]

    for piece in pieces_for_image:
        pos = piece['chessboard_position']  # e.g., "a8", "e4"
        cat_id = piece['category_id']

        col = COLS.index(pos[0])        # 'a'=0, 'h'=7
        row = ROWS.index(pos[1])        # '8'=0, '1'=7

        class_name = cat_id_to_name.get(cat_id, 'empty')
        grid[row][col] = class_name

    return grid


def process_split(split_name, image_ids, images_lookup, pieces_by_image,
                  corners_by_image, cat_id_to_name):
    """Process all images in a split that have corner annotations."""

    piece_counts = Counter()
    occ_counts = Counter()
    processed = 0
    skipped_no_corners = 0
    errors = 0

    for img_id in image_ids:
        # Skip if no corners for this image
        if img_id not in corners_by_image:
            skipped_no_corners += 1
            continue

        # Get image info
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

        # Get corners and warp
        corners_dict = corners_by_image[img_id]
        try:
            ordered_corners = chessred_corners_to_warp_order(corners_dict)
            warped = warp_chessboard_image(image, ordered_corners)
        except Exception as e:
            errors += 1
            continue

        # Build board grid from piece annotations
        pieces = pieces_by_image.get(img_id, [])
        grid = build_board_grid(pieces, cat_id_to_name)

        # Crop all 64 squares
        stem = img_info['file_name'].replace('.jpg', '').replace('.png', '')

        for row in range(8):
            for col in range(8):
                class_name = grid[row][col]
                is_empty = (class_name == 'empty')

                # Normal mapping — verified correct for ChessReD
                # (ChessReD corners place a8 at top-left, matching FEN grid directly)
                crop_row = row
                crop_col = col

                crop = crop_square_piece(warped, crop_row, crop_col)

                # --- Occupancy ---
                occ_class = 'empty' if is_empty else 'occupied'
                occ_dir = os.path.join(OUTPUT_DIR, 'occupancy', split_name, occ_class)
                os.makedirs(occ_dir, exist_ok=True)
                cv2.imwrite(os.path.join(occ_dir, f"{stem}_r{row}c{col}.png"), crop)
                occ_counts[occ_class] += 1

                # --- Piece classification (only occupied) ---
                if not is_empty:
                    # Normalize class name: ChessReD uses hyphens, we use underscores
                    normalized_class = class_name.replace('-', '_')
                    piece_dir = os.path.join(OUTPUT_DIR, 'pieces', split_name, normalized_class)
                    os.makedirs(piece_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(piece_dir, f"{stem}_r{row}c{col}.png"), crop)
                    piece_counts[normalized_class] += 1

        processed += 1

    print(f"  [{split_name}] Processed: {processed}, Skipped (no corners): {skipped_no_corners}, Errors: {errors}")
    print(f"  [{split_name}] Occupancy: {dict(occ_counts)}")
    print(f"  [{split_name}] Pieces:")
    for cls in sorted(piece_counts.keys()):
        print(f"      {cls}: {piece_counts[cls]}")

    return processed


def verify_one_image(images_lookup, pieces_by_image, corners_by_image, cat_id_to_name):
    """
    Generate a diagnostic grid overlay on one warped ChessReD image,
    similar to what we did for chesscog. Saves grid_debug images to
    test all 4 possible row/col mappings.
    """
    # Find an image that has corners
    test_img_id = None
    for img_id in corners_by_image:
        if img_id in images_lookup and img_id in pieces_by_image:
            test_img_id = img_id
            break

    if test_img_id is None:
        print("WARNING: No image with both corners and pieces found for verification")
        return

    img_info = images_lookup[test_img_id]
    img_path = os.path.join(CHESSRED_DIR, img_info['path'])
    image = cv2.imread(img_path)

    corners_dict = corners_by_image[test_img_id]
    ordered_corners = chessred_corners_to_warp_order(corners_dict)
    warped = warp_chessboard_image(image, ordered_corners)

    grid = build_board_grid(pieces_by_image[test_img_id], cat_id_to_name)

    os.makedirs(os.path.join("results", "chessred_debug"), exist_ok=True)

    # Save warped image
    cv2.imwrite(os.path.join("results", "chessred_debug", "warped.png"), warped)

    # FEN-like label for each square
    CLASS_TO_CHAR = {
        'white_pawn': 'P', 'white_rook': 'R', 'white_knight': 'N',
        'white_bishop': 'B', 'white_queen': 'Q', 'white_king': 'K',
        'black_pawn': 'p', 'black_rook': 'r', 'black_knight': 'n',
        'black_bishop': 'b', 'black_queen': 'q', 'black_king': 'k',
        'empty': '.',
    }

    # Try all 4 mappings
    mappings = {
        "normal":   lambda r, c: (r, c),
        "fliprow":  lambda r, c: (7 - r, c),
        "flipcol":  lambda r, c: (r, 7 - c),
        "flipboth": lambda r, c: (7 - r, 7 - c),
    }

    for name, mapping_fn in mappings.items():
        debug_img = warped.copy()

        # Draw grid
        for i in range(9):
            x = int(MARGIN + SQUARE_SIZE * i)
            y = int(MARGIN + SQUARE_SIZE * i)
            cv2.line(debug_img, (x, int(MARGIN)), (x, int(MARGIN + BOARD_SIZE)), (0, 255, 0), 1)
            cv2.line(debug_img, (int(MARGIN), y), (int(MARGIN + BOARD_SIZE), y), (0, 255, 0), 1)

        # Label each square
        for fen_row in range(8):
            for fen_col in range(8):
                class_name = grid[fen_row][fen_col]
                label = CLASS_TO_CHAR.get(class_name, '?')

                img_row, img_col = mapping_fn(fen_row, fen_col)
                cx = int(MARGIN + SQUARE_SIZE * img_col + SQUARE_SIZE / 2)
                cy = int(MARGIN + SQUARE_SIZE * img_row + SQUARE_SIZE / 2)

                color = (0, 0, 255) if label != '.' else (128, 128, 128)
                cv2.putText(debug_img, label, (cx - 8, cy + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = os.path.join("results", "chessred_debug", f"grid_debug_{name}.png")
        cv2.imwrite(out_path, debug_img)
        print(f"  Saved {out_path}")

    print(f"\n  CHECK results/chessred_debug/grid_debug_*.png")
    print(f"  Find the one where piece labels match the actual pieces.")
    print(f"  Then update the mapping in process_split() if needed.")
    print(f"  (Image used: {img_info['file_name']}, id={test_img_id})")


def main():
    print("=" * 60)
    print("CHESSRED DATA PREPARATION (for fine-tuning)")
    print("=" * 60)

    # Load annotations
    ann_path = os.path.join(CHESSRED_DIR, "annotations.json")
    if not os.path.exists(ann_path):
        print(f"ERROR: {ann_path} not found")
        print("Download ChessReD first: https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f")
        return

    print("Loading annotations (this may take a moment)...")
    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Build lookups
    images_lookup = {img['id']: img for img in data['images']}

    # Build pieces lookup: image_id -> list of piece annotations
    pieces_by_image = {}
    for piece in data['annotations']['pieces']:
        img_id = piece['image_id']
        if img_id not in pieces_by_image:
            pieces_by_image[img_id] = []
        pieces_by_image[img_id].append(piece)

    # Build corners lookup: image_id -> corners dict
    corners_by_image = {}
    for corner in data['annotations']['corners']:
        corners_by_image[corner['image_id']] = corner['corners']

    # Category mapping
    cat_id_to_name = {c['id']: c['name'].replace('-', '_') for c in data['categories']}

    print(f"Total images: {len(images_lookup)}")
    print(f"Images with corners: {len(corners_by_image)}")
    print(f"Categories: {cat_id_to_name}")

    # Get ChessReD2K splits (only these have corners)
    chessred2k_splits = data['splits']['chessred2k']

    # --- STEP 1: Verify mapping on one image ---
    print("\n--- DIAGNOSTIC: Verifying corner/grid mapping ---")
    verify_one_image(images_lookup, pieces_by_image, corners_by_image, cat_id_to_name)

    # --- STEP 2: Process all splits ---
    print("\n--- Processing ChessReD2K splits ---")
    total_processed = 0
    for split_name in ['train', 'val', 'test']:
        split_image_ids = chessred2k_splits[split_name]['image_ids']
        print(f"\n[{split_name}] {len(split_image_ids)} images in split")
        n = process_split(
            split_name, split_image_ids, images_lookup,
            pieces_by_image, corners_by_image, cat_id_to_name
        )
        total_processed += n

    print(f"\n{'=' * 60}")
    print(f"DONE — Processed {total_processed} images")
    print(f"Data saved to {OUTPUT_DIR}")
    print(f"{'=' * 60}")
    print(f"\nNEXT STEPS:")
    print(f"1. Check results/chessred_debug/grid_debug_*.png")
    print(f"   If 'flipboth' is NOT correct, update the mapping and re-run")
    print(f"2. If mapping is correct, run fine-tuning:")
    print(f"   python src/finetune_chessred.py")


if __name__ == '__main__':
    main()