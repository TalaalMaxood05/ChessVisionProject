"""
stage3_classify.py - Two-stage piece classification (v6)

Matches the v6 data preparation:
  - Uses chesscog warp geometry (800x800 image, 200px margin, 50px squares)
  - Tighter centered piece crops with variable height/width
  - Flip-both mapping (crop_row = 7-row, crop_col = 7-col)
  - ResNet-34 for occupancy, EfficientNet-B0 for pieces
  - Correct input resolutions (200x100 occupancy, 320x160 pieces)

Usage:
  python src/stage3_classify.py <warped_image_path>
  python src/stage3_classify.py <raw_image_path> --corners x1,y1,x2,y2,x3,y3,x4,y4
"""

import os
import sys
import torch
import cv2
import numpy as np
from torchvision import models, transforms
import torch.nn as nn


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
    """Tighter centered crop for piece classification. Matches v6 prep exactly."""
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


class TwoStageClassifier:
    """Two-stage chess piece classifier matching v6 training setup."""

    OCC_CLASSES = ["empty", "occupied"]

    PIECE_CLASSES = [
        "black_bishop", "black_king", "black_knight", "black_pawn",
        "black_queen", "black_rook",
        "white_bishop", "white_king", "white_knight", "white_pawn",
        "white_queen", "white_rook"
    ]

    PIECE_TO_FEN = {
        "black_bishop": "b", "black_king": "k", "black_knight": "n",
        "black_pawn": "p", "black_queen": "q", "black_rook": "r",
        "white_bishop": "B", "white_king": "K", "white_knight": "N",
        "white_pawn": "P", "white_queen": "Q", "white_rook": "R"
    }

    def __init__(self, model_dir=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(project_root, "models")

        # --- Load occupancy model (ResNet-34) ---
        occ_path = os.path.join(model_dir, "occupancy_classifier.pth")
        occ_ckpt = torch.load(occ_path, map_location=self.device, weights_only=True)

        occ_model_name = occ_ckpt.get("model_name", "resnet34")
        self.occ_model = self._build_model(occ_model_name, occ_ckpt.get("num_classes", 2))
        self.occ_model.load_state_dict(occ_ckpt["model_state_dict"])
        self.occ_model = self.occ_model.to(self.device)
        self.occ_model.eval()
        self.occ_class_to_idx = occ_ckpt.get("class_to_idx", {"empty": 0, "occupied": 1})
        print(f"Occupancy model loaded: {occ_model_name} (val_acc={occ_ckpt['val_acc']:.2f}%)")

        # --- Load piece model (EfficientNet-B0) ---
        piece_path = os.path.join(model_dir, "piece_classifier.pth")
        piece_ckpt = torch.load(piece_path, map_location=self.device, weights_only=True)

        piece_model_name = piece_ckpt.get("model_name", "efficientnet_b0")
        self.piece_model = self._build_model(piece_model_name, piece_ckpt.get("num_classes", 12))
        self.piece_model.load_state_dict(piece_ckpt["model_state_dict"])
        self.piece_model = self.piece_model.to(self.device)
        self.piece_model.eval()
        self.piece_class_to_idx = piece_ckpt.get("class_to_idx", {})
        print(f"Piece model loaded: {piece_model_name} (val_acc={piece_ckpt['val_acc']:.2f}%)")

        # Build reverse mappings
        self.occ_idx_to_class = {v: k for k, v in self.occ_class_to_idx.items()}
        self.piece_idx_to_class = {v: k for k, v in self.piece_class_to_idx.items()}

        # Transforms matching training
        self.occ_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((200, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.piece_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self, model_name, num_classes):
        """Build model architecture matching train_classifier.py."""
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

    def _predict(self, model, image_bgr, transform):
        """Run inference on a single crop."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
        return pred.item(), conf.item()

    def classify_square(self, square_crop):
        """Two-stage classification of a single square crop."""
        # Stage A: empty or occupied?
        occ_idx, occ_conf = self._predict(self.occ_model, square_crop, self.occ_transform)
        occ_class = self.occ_idx_to_class.get(occ_idx, "empty")

        if occ_class == "empty":
            return "empty", None, occ_conf

        # Stage B: which piece?
        piece_idx, piece_conf = self._predict(self.piece_model, square_crop, self.piece_transform)
        piece_class = self.piece_idx_to_class.get(piece_idx, "white_pawn")
        fen_char = self.PIECE_TO_FEN.get(piece_class, "P")

        combined_conf = occ_conf * piece_conf
        return piece_class, fen_char, combined_conf

    def classify_board(self, warped_image):
        """
        Classify all 64 squares of a warped 800x800 board image.

        Uses chesscog crop geometry and flip-both mapping:
          FEN row 0 / col 0 (rank 8, a-file) maps to crop_row=7, crop_col=7

        Returns: (fen_string, board_grid, confidence_grid)
        """
        board_grid = []
        conf_grid = []

        for fen_row in range(8):
            row_classes = []
            row_confs = []
            for fen_col in range(8):
                # Flip-both: FEN grid to warped image coordinates
                crop_row = 7 - fen_row
                crop_col = 7 - fen_col

                # Use the same crop function as training
                square_crop = crop_square_piece(warped_image, crop_row, crop_col)

                class_name, fen_char, conf = self.classify_square(square_crop)
                row_classes.append((class_name, fen_char))
                row_confs.append(conf)

            board_grid.append(row_classes)
            conf_grid.append(row_confs)

        fen = self._grid_to_fen(board_grid)
        return fen, board_grid, conf_grid

    def _grid_to_fen(self, board_grid):
        """Convert board grid to FEN string."""
        fen_rows = []
        for row in board_grid:
            fen_row = ""
            empty_count = 0
            for class_name, fen_char in row:
                if fen_char is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += fen_char
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        return "/".join(fen_rows)


def print_board(fen, grid, confs):
    """Pretty-print the classified board."""
    print(f"\nPredicted FEN: {fen}")
    print(f"\n    a  b  c  d  e  f  g  h")
    print(f"  +{'---+' * 8}")
    for row_idx in range(8):
        pieces = []
        for class_name, fen_char in grid[row_idx]:
            if fen_char is None:
                pieces.append(" . ")
            else:
                pieces.append(f" {fen_char} ")
        rank = 8 - row_idx
        print(f"{rank} |{'|'.join(pieces)}|")
        print(f"  +{'---+' * 8}")
    print(f"    a  b  c  d  e  f  g  h")

    confs_flat = [c for row in confs for c in row]
    print(f"\nAvg confidence: {np.mean(confs_flat):.3f}")
    print(f"Min confidence: {np.min(confs_flat):.3f}")
    print(f"\nView: https://lichess.org/editor/{fen}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/stage3_classify.py <warped_800x800_image>")
        print("  python src/stage3_classify.py <raw_image> --corners h8x,h8y,h1x,h1y,a1x,a1y,a8x,a8y")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        sys.exit(1)

    # If corners provided, warp first
    if "--corners" in sys.argv:
        corners_idx = sys.argv.index("--corners") + 1
        coords = list(map(float, sys.argv[corners_idx].split(",")))
        if len(coords) != 8:
            print("Error: Need 8 corner values: h8x,h8y,h1x,h1y,a1x,a1y,a8x,a8y")
            sys.exit(1)
        # JSON order: [h8, h1, a1, a8] -> TL,TR,BR,BL = [a8, h8, h1, a1]
        corners_json = [
            [coords[0], coords[1]],  # h8
            [coords[2], coords[3]],  # h1
            [coords[4], coords[5]],  # a1
            [coords[6], coords[7]],  # a8
        ]
        ordered = [corners_json[3], corners_json[0], corners_json[1], corners_json[2]]
        image = warp_chessboard_image(image, ordered)
        print(f"Warped image to {IMG_SIZE}x{IMG_SIZE}")

    print(f"Image: {image.shape}")
    classifier = TwoStageClassifier()
    fen, grid, confs = classifier.classify_board(image)
    print_board(fen, grid, confs)