"""
Stage 1: Corner Detection Module

Detects the four outer corners of a chessboard from an input image
using Hough line detection and geometric intersection.
"""

import random
import cv2
import numpy as np


class Stage1CornerDetection:

    def __init__(
            self,
            canny_low=50,
            canny_high=150,
            hough_threshold=50,
            min_line_length=30,
            max_line_gap=20
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap


    # 1. PREPROCESSING

    def preprocess_image(self, image):
        """
        Converts input image to grayscale and smooths it.

        Parameters:
            image: BGR image loaded with cv2.imread()

        Returns:
            gray_blur: blurred grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray_blur


    # 2. EDGE DETECTION

    def detect_edges(self, gray):
        """
        Detects edges using the Canny edge detector.

        Parameters:
            gray: grayscale image

        Returns:
            edges: binary edge image
        """
        return cv2.Canny(gray, self.canny_low, self.canny_high)


    # 3. HOUGH LINE DETECTION

    def detect_hough_lines(self, gray):
        """
        Detects line segments using probabilistic Hough line detection.

        Parameters:
            gray: grayscale image

        Returns:
            lines: output as cv2.HoughLinesP
        """
        edges = self.detect_edges(gray)

        points = np.argwhere(edges > 0)

        if len(points) == 0:
            return None

        h, w = edges.shape
        max_rho = int(np.ceil(np.sqrt(h*h + w*w)))



        theta_vals = np.deg2rad(np.arange(0, 180))
        cos_vals = np.cos(theta_vals)
        sin_vals = np.sin(theta_vals)

        accumulator = np.zeros(
            (2 * max_rho + 1, len(theta_vals)),
            dtype=np.int32
        )

        edge_points = [(int(x), int(y)) for y, x in points]
        random.shuffle(edge_points)

        active_points = set(edge_points)
        detected_lines = []

        for x, y in edge_points:

            if (x, y) not in active_points:
                continue



            for theta_idx in range(len(theta_vals)):

                rho = x * cos_vals[theta_idx] + y * sin_vals[theta_idx]
                rho_idx = int(round(rho)) + max_rho

                accumulator[rho_idx, theta_idx] += 1

                if accumulator[rho_idx, theta_idx] >= self.hough_threshold:

                    rho_val = rho_idx - max_rho

                    theta = theta_vals[theta_idx]


                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)

                    line_points = []

                    for px, py in active_points:

                        dist = abs(px*cos_t + py*sin_t - rho_val)

                        if dist < 2.0:
                            line_points.append((px, py))

                    if len(line_points) < self.min_line_length:
                        continue

                    projections = []

                    dx = -sin_t

                    dy = cos_t

                    for px, py in line_points:

                        t = px*dx + py*dy
                        projections.append((t, px, py))

                    if len(projections) == 0:
                        continue

                    projections.sort()

                    best_segment = []
                    current_segment = [projections[0]]

                    for i in range(1, len(projections)):

                        if abs(projections[i][0] - projections[i-1][0]) <= self.max_line_gap:
                            current_segment.append(projections[i])

                        else:

                            if len(current_segment) > len(best_segment):
                                best_segment = current_segment
                            current_segment = [projections[i]]

                    if len(current_segment) > len(best_segment):
                        best_segment = current_segment

                    if len(best_segment) >= self.min_line_length:

                        _, x1, y1 = best_segment[0]
                        _, x2, y2 = best_segment[-1]

                        detected_lines.append(
                            np.array([[x1, y1, x2, y2]])
                        )

                        for _, px, py in best_segment:
                            if (px, py) in active_points:
                                active_points.remove((px, py))

                        accumulator[rho_idx, theta_idx] = 0

        if len(detected_lines) == 0:
            return None

        return np.array(detected_lines)


    # 4. SPLIT LINES BY ORIENTATION

    def split_lines_by_orientation(self, lines):
        """
        Splits detected Hough lines into horizontal and vertical groups.

        Parameters:
            lines: output from cv2.HoughLinesP

        Returns:
            horizontal_lines: list of horizontal line segments
            vertical_lines:   list of vertical line segments
        """
        horizontal_lines = []
        vertical_lines = []

        if lines is None:
            return horizontal_lines, vertical_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 20 or angle > 160:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 70 < angle < 110:
                vertical_lines.append((x1, y1, x2, y2))

        return horizontal_lines, vertical_lines


    # 5. INTERSECTION COMPUTATION

    def compute_intersections(self, horizontal_lines, vertical_lines):
        """
        Computes intersections between horizontal and vertical lines.

        Parameters:
            horizontal_lines: list of horizontal line segments
            vertical_lines:   list of vertical line segments

        Returns:
            intersections: Nx2 float32 array of (x, y) intersection points
        """
        intersections = []

        for x1, y1, x2, y2 in horizontal_lines:
            for x3, y3, x4, y4 in vertical_lines:
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-6:
                    continue
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                intersections.append([px, py])

        return np.array(intersections, dtype=np.float32)


    # 6. CORNER SELECTION

    def select_board_corners(self, intersections):
        """
        Selects the four outer corners of the board from intersection points.

        Parameters:
            intersections: Nx2 array of intersection points

        Returns:
            corners: 4x2 float32 array [TL, TR, BR, BL], or None
        """
        if len(intersections) < 4:
            return None

        sums = intersections[:, 0] + intersections[:, 1]
        diffs = intersections[:, 0] - intersections[:, 1]

        top_left     = intersections[np.argmin(sums)]
        bottom_right = intersections[np.argmax(sums)]
        top_right    = intersections[np.argmax(diffs)]
        bottom_left  = intersections[np.argmin(diffs)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


    def order_corners(self, corners):
        """
        Ensures corners are ordered consistently as [TL, TR, BR, BL].

        Parameters:
            corners: 4x2 array

        Returns:
            rect: 4x2 float32 array in canonical order
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        rect[0] = corners[np.argmin(s)]     # top-left
        rect[2] = corners[np.argmax(s)]     # bottom-right
        rect[1] = corners[np.argmin(diff)]  # top-right
        rect[3] = corners[np.argmax(diff)]  # bottom-left

        return rect


    # 7. FULL PIPELINE

    def run_stage1(self, image):
        """
        Runs the full corner detection pipeline on a BGR image.

        Parameters:
            image: BGR image (numpy array)

        Returns:
            ordered_corners: 4x2 float32 [TL, TR, BR, BL], or None on failure
        """
        gray = self.preprocess_image(image)
        lines = self.detect_hough_lines(gray)
        horizontal_lines, vertical_lines = self.split_lines_by_orientation(lines)
        intersections = self.compute_intersections(horizontal_lines, vertical_lines)
        board_corners = self.select_board_corners(intersections)

        if board_corners is None:
            return None

        return self.order_corners(board_corners)