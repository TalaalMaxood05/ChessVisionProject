# Stage 2: Homography Computation and Board Rectification

## Overview
This module serves as the **Geometric Reasoning** engine of the chess reconstruction pipeline. It transforms a distorted perspective image into a normalized, top-down $480 \times 480$ grid. By computing a **Planar Homography** from four detected corners, it ensures that every square on the board is perfectly aligned for the Stage 3 CNN piece classifier.

---

## Core Features
* **Coordinate Ordering**: Consistently sorts input points into `[Top-Left, Top-Right, Bottom-Right, Bottom-Left]` to prevent axial flips or rotations during warping.
* **Perspective Transformation**: Uses a $3 \times 3$ homography matrix to map the physical chessboard to a standardized coordinate space.
* **Square Segmentation**: Automatically slices the rectified board into 64 individual image crops, indexed by algebraic notation (a1-h8).
* **Sensitivity Analysis Support**: Built-in functions for adding Gaussian noise to corners and computing reprojection errors, facilitating rigorous testing of system robustness.

---

## Configuration (`WarpConfig`)
The pipeline behavior is controlled via a centralized configuration class:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `output_size` | 480 | The pixel width/height of the full rectified board image. |
| `square_size` | 60 | The size of each individual square crop ($480 / 8$). |
| `padding` | 20 | Internal buffer to prevent edge clipping during the warp. |
| `interpolation` | `cv2.INTER_LINEAR` | The mathematical method used for pixel resampling. |

---

## Key Functions

### `process(image, corners, config)`
The primary entry point. It orchestrates the entire Stage 2 flow:
1.  Validates input corner coordinates.
2.  Computes the homography matrix $H$.
3.  Applies `cv2.warpPerspective` to the original BGR image.
4.  Splits the resulting board into 64 square dictionaries.

### `draw_grid_overlay(rectified, config)`
A visual validation tool that draws an $8 \times 8$ green grid over the rectified image.
* **Success**: Grid lines align perfectly with the square boundaries.
* **Failure**: Grid lines bisect pieces or squares, indicating Stage 1 detection error.

### `perturb_corners(corners, noise_std)`
Specifically for **Sensitivity Analysis**, this function adds controlled noise to the detected corners to measure the "drift" in FEN accuracy as localization error increases.

---

## CLI Usage
To test rectification as a standalone component:

```bash
python stage2_wrap.py --image data/input.jpg --corners data/corners.json --save_squares --output_dir results/test
```

## Ground Truth Corners
The corners below are manually mapped for testing for chess_img.jpg
```
[
[7, 129],   # TL
[700, 62],  # TR
[1014, 505],  # BR
[100, 650]    # BL
]
```
