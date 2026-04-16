import cv2
import os
import numpy as np
from src.stage1_corners import Stage1CornerDetection


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "..", "data", "chessboard.jpg")

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image:", image_path)
        return

    detector = Stage1CornerDetection()

    # -------------------------------------------------------------------------
    # STAGE 0: Original image
    # -------------------------------------------------------------------------
    cv2.imshow("Stage 0 - Original Image", image)
    cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # STAGE 1: Preprocessing (grayscale + Gaussian blur)
    # -------------------------------------------------------------------------
    gray = detector.preprocess_image(image)

    gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(gray_display, "Grayscale + Gaussian Blur", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Stage 1 - Preprocessing", gray_display)
    cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # STAGE 2: Canny edge detection
    # -------------------------------------------------------------------------
    edges = detector.detect_edges(gray)

    edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(edges_display, "Canny Edge Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Stage 2 - Edge Detection", edges_display)
    cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # STAGE 3: Hough line detection + orientation split
    # -------------------------------------------------------------------------
    lines = detector.detect_hough_lines(gray)
    hough_output = image.copy()

    if lines is not None:
        horizontal_lines, vertical_lines = detector.split_lines_by_orientation(lines)

        for (x1, y1, x2, y2) in horizontal_lines:
            cv2.line(hough_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x1, y1, x2, y2) in vertical_lines:
            cv2.line(hough_output, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(hough_output,
                    f"Hough Lines  |  Green=Horizontal ({len(horizontal_lines)})  Blue=Vertical ({len(vertical_lines)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(hough_output, "No lines detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        horizontal_lines, vertical_lines = [], []

    cv2.imshow("Stage 3 - Hough Line Detection", hough_output)
    cv2.waitKey(0)

    print("Hough lines:")
    print(lines)

    # -------------------------------------------------------------------------
    # STAGE 4: Intersection computation
    # -------------------------------------------------------------------------
    intersections = detector.compute_intersections(horizontal_lines, vertical_lines)
    intersection_output = image.copy()

    h_img, w_img = image.shape[:2]
    for (x, y) in intersections.astype(int):
        if 0 <= x < w_img and 0 <= y < h_img:
            cv2.circle(intersection_output, (x, y), 4, (0, 255, 0), -1)

    cv2.putText(intersection_output, f"Intersections: {len(intersections)} points",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Stage 4 - Line Intersections", intersection_output)
    cv2.waitKey(0)

    # -------------------------------------------------------------------------
    # STAGE 5: Final board corner selection
    # -------------------------------------------------------------------------
    ordered_corners = detector.run_stage1(image)

    if ordered_corners is None:
        print("Could not detect board corners")
        cv2.destroyAllWindows()
        return

    final_output = image.copy()
    pts = ordered_corners.astype(int)

    corner_labels = ["0: TL", "1: TR", "2: BR", "3: BL"]
    for i, (x, y) in enumerate(pts):
        cv2.circle(final_output, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(final_output, corner_labels[i], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(final_output, p1, p2, (255, 0, 0), 3)

    cv2.putText(final_output, "Final Board Corners", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Stage 5 - Detected Board Corners", final_output)
    cv2.waitKey(0)

    print("Detected board corners:")
    print(ordered_corners)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()