"""
Stage 2: Homography Computation and Board Rectification

This module receives four chessboard corner coordinates from Stage 1
and produces 64 individual square crops for Stage 3 classification.

Input:  Original image (BGR) + four corner points (top-left, top-right, bottom-right, bottom-left)
Output: List of 64 cropped square images in order a1-h8, plus the full rectified board image
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class WarpConfig:
    """Configuration for the rectification pipeline."""
    output_size: int = 480          # Size of the rectified board image (square)
    square_size: int = 60           # Size of each individual square crop (480 / 8)
    border_margin: int = 0          # Optional margin around each square crop (pixels)
    interpolation: int = cv2.INTER_LINEAR  # Interpolation method for warping


# Square naming: maps (row, col) to algebraic notation
# Row 0 = rank 8 (top of board from white's perspective)
# Col 0 = file 'a' (left side from white's perspective)
SQUARE_NAMES = [
    [f"{chr(ord('a') + col)}{8 - row}" for col in range(8)]
    for row in range(8)
]


# ──────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────

def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Orders four corner points consistently as:
    [top-left, top-right, bottom-right, bottom-left]

    This ensures the homography maps correctly regardless of
    the order Stage 1 returns the corners.

    Args:
        corners: Array of shape (4, 2) with four (x, y) corner coordinates.

    Returns:
        Ordered array of shape (4, 2).
    """
    corners = np.array(corners, dtype=np.float32)
    assert corners.shape == (4, 2), f"Expected (4, 2) corners, got {corners.shape}"

    # Sort by y-coordinate to separate top pair from bottom pair
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_pair = sorted_by_y[:2]
    bottom_pair = sorted_by_y[2:]

    # Within each pair, sort by x-coordinate (left first)
    top_left, top_right = top_pair[np.argsort(top_pair[:, 0])]
    bottom_left, bottom_right = bottom_pair[np.argsort(bottom_pair[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def compute_homography(
    src_corners: np.ndarray,
    config: WarpConfig = WarpConfig(),
    padding: int = 20  # New: add some breathing room in pixels
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 3x3 homography matrix that maps the four detected
    chessboard corners to a square output image.

    The homography H satisfies: dst ~ H @ src (in homogeneous coordinates)

    For four point correspondences, this is an exact solution (no RANSAC needed)
    since 4 points give 8 equations for 8 degrees of freedom.

    Args:
        src_corners: Detected corners (4, 2) in order [TL, TR, BR, BL].
        config: Warp configuration specifying output dimensions.

    Returns:
        H: The 3x3 homography matrix.
        dst_corners: The target corners in the output image (4, 2).
    """
    src = np.array(src_corners, dtype=np.float32)
    s = config.output_size
    
    # Define destination with a small buffer inside the output image
    # This prevents the edges from being clipped at the very 0 or 480 boundary
    dst = np.array([
        [padding, padding],           # top-left
        [s - padding, padding],       # top-right
        [s - padding, s - padding],   # bottom-right
        [padding, s - padding],       # bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    return H, dst


def rectify_board(
    image: np.ndarray,
    src_corners: np.ndarray,
    config: WarpConfig = WarpConfig()
) -> np.ndarray:
    """
    Applies the homography to warp the input image into a top-down
    rectified view of the chessboard.

    Args:
        image: Original BGR image from camera.
        src_corners: Ordered corner points (4, 2) from Stage 1.
        config: Warp configuration.

    Returns:
        Rectified board image of shape (output_size, output_size, 3).
    """
    H, _ = compute_homography(src_corners, config)

    rectified = cv2.warpPerspective(
        image,
        H,
        (config.output_size, config.output_size),
        flags=config.interpolation,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rectified


def split_into_squares(
    rectified: np.ndarray,
    config: WarpConfig = WarpConfig()
) -> List[dict]:
    """
    Divides the rectified board image into 64 individual square crops.

    Each square is returned with its algebraic name (e.g., 'a8', 'b7')
    and its image crop, ordered from top-left to bottom-right:
        a8, b8, c8, ..., h8,  (rank 8, row 0)
        a7, b7, c7, ..., h7,  (rank 7, row 1)
        ...
        a1, b1, c1, ..., h1   (rank 1, row 7)

    Args:
        rectified: Rectified top-down board image.
        config: Warp configuration.

    Returns:
        List of 64 dicts, each containing:
            - 'name': Algebraic square name (e.g., 'e4')
            - 'image': Cropped square image (square_size x square_size x 3)
            - 'row': Row index (0 = top = rank 8)
            - 'col': Column index (0 = left = file a)
    """
    sq = config.square_size
    squares = []

    for row in range(8):
        for col in range(8):
            y_start = row * sq
            x_start = col * sq
            crop = rectified[y_start:y_start + sq, x_start:x_start + sq].copy()

            squares.append({
                'name': SQUARE_NAMES[row][col],
                'image': crop,
                'row': row,
                'col': col,
            })

    return squares


# ──────────────────────────────────────────────
# Pipeline Interface (called by pipeline.py)
# ──────────────────────────────────────────────

def process(
    image: np.ndarray,
    corners: np.ndarray,
    config: Optional[WarpConfig] = None
) -> dict:
    """
    Main entry point for Stage 2. Takes the original image and detected
    corners, returns everything Stage 3 and the evaluation code need.

    Args:
        image: Original BGR image.
        corners: Four corner points from Stage 1 (any order).
        config: Optional warp configuration.

    Returns:
        Dictionary containing:
            - 'rectified': Full rectified board image (480x480x3)
            - 'squares': List of 64 square dicts (see split_into_squares)
            - 'homography': The 3x3 homography matrix
            - 'ordered_corners': The corners after ordering
            - 'config': The configuration used
    """
    if corners is None:
        raise ValueError("Stage 1 failed to detect corners. Cannot proceed to Stage 2.")

    if config is None:
        config = WarpConfig()

    # Step 1: REMOVE your internal order_corners call. 
    # Stage 1 (stage1_corners.py) already performs: 
    # [top-left, top-right, bottom-right, bottom-left]
    ordered = corners.astype(np.float32) 

    # Step 2: Compute homography
    H, dst_corners = compute_homography(ordered, config)

    # Step 2: Compute homography
    H, dst_corners = compute_homography(ordered, config)

    # Step 3: Warp the image
    rectified = rectify_board(image, ordered, config)

    # Step 4: Split into 64 squares
    squares = split_into_squares(rectified, config)

    return {
        'rectified': rectified,
        'squares': squares,
        'homography': H,
        'ordered_corners': ordered,
        'config': config,
    }


# ──────────────────────────────────────────────
# Evaluation Utilities (for Fahmeed's sensitivity analysis)
# ──────────────────────────────────────────────

def compute_reprojection_error(
    H: np.ndarray,
    src_points: np.ndarray,
    dst_points_expected: np.ndarray
) -> np.ndarray:
    """
    Computes reprojection error for a set of point correspondences.

    This measures how accurately the homography maps known points.
    Useful for evaluating warp quality beyond just the four corners —
    e.g., testing all 81 grid intersections.

    Args:
        H: 3x3 homography matrix.
        src_points: Points in the original image (N, 2).
        dst_points_expected: Where those points should land (N, 2).

    Returns:
        Array of per-point Euclidean errors (N,).
    """
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    expected = np.array(dst_points_expected, dtype=np.float32)

    errors = np.linalg.norm(projected - expected, axis=1)
    return errors


def perturb_corners(
    corners: np.ndarray,
    noise_std: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Adds Gaussian noise to corner coordinates for sensitivity analysis.

    Fahmeed can use this to systematically test how corner detection
    errors propagate through the homography and affect downstream
    classification.

    Args:
        corners: Original corner coordinates (4, 2).
        noise_std: Standard deviation of Gaussian noise in pixels.
        rng: Optional random number generator for reproducibility.

    Returns:
        Perturbed corners (4, 2).
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, noise_std, size=corners.shape).astype(np.float32)
    return corners + noise


# ──────────────────────────────────────────────
# Visualization Helpers
# ──────────────────────────────────────────────

def draw_grid_overlay(
    rectified: np.ndarray,
    config: WarpConfig = WarpConfig(),
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1
) -> np.ndarray:
    """
    Draws the 8x8 grid on the rectified image for visual sanity checking.

    If the grid lines align with the actual board squares, the warp is good.
    If pieces bleed across grid lines, something is off.

    Args:
        rectified: Rectified board image.
        config: Warp configuration.
        color: BGR color for grid lines.
        thickness: Line thickness in pixels.

    Returns:
        Copy of the image with grid overlay.
    """
    vis = rectified.copy()
    sq = config.square_size

    for i in range(1, 8):
        # Vertical lines
        cv2.line(vis, (i * sq, 0), (i * sq, config.output_size), color, thickness)
        # Horizontal lines
        cv2.line(vis, (0, i * sq), (config.output_size, i * sq), color, thickness)

    return vis


def draw_corners_on_image(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 8
) -> np.ndarray:
    """
    Draws detected corners on the original image for debugging.

    Args:
        image: Original image.
        corners: Four corner points (4, 2).
        color: BGR color for corner markers.
        radius: Marker radius in pixels.

    Returns:
        Copy of the image with corner markers.
    """
    vis = image.copy()
    labels = ['TL', 'TR', 'BR', 'BL']

    for (x, y), label in zip(corners.astype(int), labels):
        cv2.circle(vis, (x, y), radius, color, -1)
        cv2.putText(vis, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis


# ──────────────────────────────────────────────
# Standalone Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Stage 2: Chessboard Rectification")
    parser.add_argument("--image", required=True, help="Path to input chessboard image")
    parser.add_argument("--corners", required=True,
                        help="Path to JSON file with corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    parser.add_argument("--output_dir", default="results/stage2",
                        help="Directory to save outputs")
    parser.add_argument("--output_size", type=int, default=480,
                        help="Size of rectified output image")
    parser.add_argument("--save_squares", action="store_true",
                        help="Save individual square crops")
    args = parser.parse_args()

    # Load inputs
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    with open(args.corners, 'r') as f:
        corners_raw = json.load(f)
    corners = np.array(corners_raw, dtype=np.float32)

    # Configure and run
    config = WarpConfig(output_size=args.output_size, square_size=args.output_size // 8)
    result = process(image, corners, config)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Save rectified board
    rectified_path = os.path.join(args.output_dir, "rectified.png")
    cv2.imwrite(rectified_path, result['rectified'])
    print(f"Saved rectified board: {rectified_path}")

    # Save grid overlay for visual verification
    overlay = draw_grid_overlay(result['rectified'], config)
    overlay_path = os.path.join(args.output_dir, "rectified_grid_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved grid overlay: {overlay_path}")

    # Save corner visualization on original image
    corners_vis = draw_corners_on_image(image, result['ordered_corners'])
    corners_path = os.path.join(args.output_dir, "detected_corners.png")
    cv2.imwrite(corners_path, corners_vis)
    print(f"Saved corner visualization: {corners_path}")

    # Optionally save individual square crops
    if args.save_squares:
        squares_dir = os.path.join(args.output_dir, "squares")
        os.makedirs(squares_dir, exist_ok=True)
        for sq_data in result['squares']:
            sq_path = os.path.join(squares_dir, f"{sq_data['name']}.png")
            cv2.imwrite(sq_path, sq_data['image'])
        print(f"Saved {len(result['squares'])} square crops to {squares_dir}/")

    # Save homography matrix
    h_path = os.path.join(args.output_dir, "homography.npy")
    np.save(h_path, result['homography'])
    print(f"Saved homography matrix: {h_path}")

    print("\nHomography matrix:")
    print(result['homography'])
    print(f"\nRectified image size: {result['rectified'].shape}")
    print(f"Number of squares: {len(result['squares'])}")
    print(f"Square crop size: {result['squares'][0]['image'].shape}")