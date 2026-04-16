import cv2
import os
from src.stage1_corners import Stage1CornerDetection


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "..", "data", "img_1.png")

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image:", image_path)
        return

    detector = Stage1CornerDetection()

    gray = detector.preprocess_image(image)

    # Harris corners
    harris_corners = detector.detect_harris_corners(gray)
    harris_output = image.copy()

    for (x, y) in harris_corners.astype(int):
        cv2.circle(harris_output, (x, y), 3, (0, 0, 255), -1)

    print("Harris corners:")
    print(harris_corners)

    # Hough lines
    lines = detector.detect_hough_lines(gray)
    hough_output = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("Hough lines:")
    print(lines)

    # Final board corners
    ordered_corners = detector.run_stage1(image)

    if ordered_corners is None:
        print("Could not detect board corners")
        cv2.imshow("Harris Corners", harris_output)
        cv2.imshow("Hough Lines", hough_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    final_output = image.copy()
    pts = ordered_corners.astype(int)

    for i, (x, y) in enumerate(pts):
        cv2.circle(final_output, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            final_output,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(final_output, p1, p2, (255, 0, 0), 3)

    print("Detected board corners:")
    print(ordered_corners)

    #cv2.imshow("Harris Corners", harris_output)
    #cv2.imshow("Hough Lines", hough_output)
    cv2.imshow("Detected Board", final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()