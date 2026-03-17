import cv2
import os
from src.stage1_corners import Stage1CornerDetection


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "..", "data", "chessboard.jpg")

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image:", image_path)
        return

    detector = Stage1CornerDetection()

    ordered_corners = detector.run_stage1(image)

    if ordered_corners is None:
        print("Could not detect board corners")
        return

    output = image.copy()
    pts = ordered_corners.astype(int)

    #drawing the four corners
    for i, (x, y) in enumerate(pts):
        cv2.circle(output, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    # drawing the board outline
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(output, p1, p2, (0, 0, 255), 3)

    print("Detected corners:")
    print(ordered_corners)

    cv2.imshow("Detected Board", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()