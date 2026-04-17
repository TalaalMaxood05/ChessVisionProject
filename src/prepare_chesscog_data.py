import os
import json
import cv2
import numpy as np
from collections import Counter

"""
prepare_chesscog_data_flipboth_centered_v6.py

Fixes applied:
- Uses explicit JSON corner ordering: [h8, h1, a1, a8] -> [TL, TR, BR, BL] = [a8, h8, h1, a1]
- Uses verified flip-both mapping for crop coordinates: crop_row = 7 - row, crop_col = 7 - col
- Uses tighter piece crops for classification to reduce second-piece leakage
- Centers cropped content on fixed canvas instead of bottom-aligning
- Clamps crop bounds safely
- Writes output to data/processed_v6
- Saves sample crops to results/v6_crops and warped board to results/warped_v6.png
"""

# === CHESSCOG GEOMETRY ===
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE        # 400
IMG_SIZE = BOARD_SIZE * 2           # 800
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2  # 200

# === TIGHTER CLASSIFICATION CROP PARAMS ===
MIN_HEIGHT_INCREASE = 0.6
MAX_HEIGHT_INCREASE = 2.0
MIN_WIDTH_INCREASE = 0.05
MAX_WIDTH_INCREASE = 0.45

OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)    # 72
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)  # 150

# === CONFIG ===
CHESSCOG_DIR = os.path.join("data", "chesscog", "render")
OUTPUT_DIR = os.path.join("data", "processed_v6")

FEN_MAP = {
    'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop',
    'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
    'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop',
    'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king'
}


def warp_chessboard_image(img, corners):
    """Warp image using explicit corner order TL, TR, BR, BL."""
    src_points = np.array(corners, dtype=np.float32)
    dst_points = np.array([
        [MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, MARGIN],
        [BOARD_SIZE + MARGIN, BOARD_SIZE + MARGIN],
        [MARGIN, BOARD_SIZE + MARGIN],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, H, (IMG_SIZE, IMG_SIZE))


def ordered_json_corners(corners):
    """JSON order is [h8, h1, a1, a8]. Convert to [TL, TR, BR, BL]."""
    return [corners[3], corners[0], corners[1], corners[2]]


def crop_square_piece(img, row, col):
    """Tighter centered crop for piece classification."""
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

    # Flip left-side crops so pieces face consistently relative to frame.
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


def crop_square_occupancy(img, row, col):
    """Use piece-style crop for occupancy too, so presence is visually obvious."""
    return crop_square_piece(img, row, col)


def parse_fen_to_grid(fen):
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


def process_split(split_name):
    split_dir = os.path.join(CHESSCOG_DIR, split_name)
    if not os.path.exists(split_dir):
        print(f"WARNING: {split_dir} not found")
        return

    all_files = os.listdir(split_dir)
    image_files = sorted([f for f in all_files if f.endswith('.png')])
    print(f"\n[{split_name}] Found {len(image_files)} images")

    piece_counts = Counter()
    occ_counts = Counter()
    errors = 0

    for img_file in image_files:
        img_path = os.path.join(split_dir, img_file)
        json_path = os.path.join(split_dir, img_file.replace('.png', '.json'))

        if not os.path.exists(json_path):
            errors += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            errors += 1
            continue

        with open(json_path, 'r') as f:
            annotation = json.load(f)

        fen = annotation.get('fen')
        corners = annotation.get('corners')

        if fen is None or corners is None:
            errors += 1
            continue

        try:
            fen_grid = parse_fen_to_grid(fen)
            warped = warp_chessboard_image(image, ordered_json_corners(corners))
        except Exception:
            errors += 1
            continue

        stem = img_file.replace('.png', '')

        for row in range(8):
            for col in range(8):
                piece = fen_grid[row][col]
                is_empty = piece is None

                crop_row = 7 - row
                crop_col = 7 - col

                occ_crop = crop_square_occupancy(warped, crop_row, crop_col)
                occ_class = 'empty' if is_empty else 'occupied'
                occ_dir = os.path.join(OUTPUT_DIR, 'occupancy', split_name, occ_class)
                os.makedirs(occ_dir, exist_ok=True)
                cv2.imwrite(os.path.join(occ_dir, f"{stem}_r{row}c{col}.png"), occ_crop)
                occ_counts[occ_class] += 1

                if not is_empty:
                    piece_crop = crop_square_piece(warped, crop_row, crop_col)
                    class_name = FEN_MAP[piece]
                    piece_dir = os.path.join(OUTPUT_DIR, 'pieces', split_name, class_name)
                    os.makedirs(piece_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(piece_dir, f"{stem}_r{row}c{col}.png"), piece_crop)
                    piece_counts[class_name] += 1

    print(f"[{split_name}] Errors: {errors}")
    print(f"[{split_name}] Occupancy: {dict(occ_counts)}")
    print(f"[{split_name}] Pieces:")
    for cls in sorted(piece_counts.keys()):
        print(f"    {cls}: {piece_counts[cls]}")


def verify_crops():
    json_path = os.path.join(CHESSCOG_DIR, 'train', '0000.json')
    img_path = os.path.join(CHESSCOG_DIR, 'train', '0000.png')

    if not os.path.exists(json_path):
        print('Cannot verify - 0000.json not found')
        return

    with open(json_path) as f:
        d = json.load(f)

    img = cv2.imread(img_path)
    warped = warp_chessboard_image(img, ordered_json_corners(d['corners']))
    fen_grid = parse_fen_to_grid(d['fen'])

    out_dir = os.path.join('results', 'v6_crops')
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for row in range(8):
        for col in range(8):
            piece = fen_grid[row][col]
            if piece is not None:
                crop_row = 7 - row
                crop_col = 7 - col
                crop = crop_square_piece(warped, crop_row, crop_col)
                class_name = FEN_MAP[piece]
                cv2.imwrite(os.path.join(out_dir, f"{class_name}_r{row}c{col}.png"), crop)
                count += 1

    print(f"Saved {count} sample crops to {out_dir}/")
    cv2.imwrite(os.path.join('results', 'warped_v6.png'), warped)
    print('Saved results/warped_v6.png')


def main():
    print('=' * 60)
    print('CHESSCOG DATA PREPARATION v6 (flip-both + centered + tighter)')
    print('=' * 60)

    print('\nVerifying crops on one image...')
    verify_crops()

    print('\nProcessing all splits...')
    for split in ['train', 'val', 'test']:
        process_split(split)

    print('\n' + '=' * 60)
    print('DONE — Data saved to', OUTPUT_DIR)
    print('=' * 60)


if __name__ == '__main__':
    main()
