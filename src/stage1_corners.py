"""
Stage 1: Corner Detection Module
Members A & B

This module handles detection of chessboard corners from input images.
"""

import cv2
import numpy as np

class Stage1CornerDetection:
    """
    Stage 1 detector for chessboard geometry.

    Handles:
        - image preprocessing
        - Harris corner detection
        - duplicate corner filtering
        - line separation by orientation
        - visualization
    """

    def __init__(
            self,
            harris_block_size=2,
            harris_ksize=3,
            harris_k=0.04,
            harris_thresh_ratio=0.01,
            min_corner_distance=10,
            canny_low=50,
            canny_high=150,
            hough_threshold=50,
            min_line_length=30,
            max_line_gap=20
    ):
        # Harris parameters
        self.harris_block_size = harris_block_size
        self.harris_ksize = harris_ksize
        self.harris_k = harris_k
        self.harris_thresh_ratio = harris_thresh_ratio
        self.min_corner_distance = min_corner_distance

        # Hough parameters
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


    # 2. HARRIS CORNER DETECTION

    def detect_harris_corners(self, gray):
        """
        Detects corners using the Harris corner detector.

        Steps:
            1. Convert grayscale image to float32
            2. Compute Harris response
            3. Dilate the response to see clearer peaks
            4. Threshold strong responses
            5. Convert detected locations into (x,y) coordinates
            6. Remove duplicate detections nearby

        Parameters:
            gray : grayscale image

        Returns:
              corners : Nx2 array of (x,y) corner points
        """
        if gray is None or gray.size == 0:
            raise ValueError("Input grayscale image is empty or None")

        gray_float = np.float32(gray)

        harris_response = cv2.cornerHarris(
            gray_float,
            self.harris_block_size,
            self.harris_ksize,
            self.harris_k
        )

        # Makes corner peaks larger
        harris_response = cv2.dilate(harris_response, None)

        threshold = self.harris_thresh_ratio * np.max(harris_response)

        # rows = y, cols = x
        ys, xs = np.where(harris_response > threshold)

        if(len(xs))==0:
            return np.empty((0,2),dtype=np.float32)

        candidates = []
        for x,y in zip(xs,ys):
            response = harris_response[y, x]
            candidates.append([x, y, response])

        # Remove cluster of repeated detections around same corner
        candidates = np.array(candidates, dtype=np.float32)
        corners = self.filter_duplicate_corners(candidates)
        return corners


    def filter_duplicate_corners(self, candidates):
        """
        Removes nearby duplicate Harris detections.

        Parameters:
            corners: Nx2 array of corner points

        Returns:
             filtered_corners : Mx2 array
        """

        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.float32)

        #Sort by response strength descending
        candidates = candidates[np.argsort(-candidates[:,2])]

        filtered = []

        for candidate in candidates:
            x,y,response = candidate
            if len(filtered) == 0:
                filtered.append([x,y])
                continue

            filteredArray = np.array(filtered, dtype=np.float32)
            distances = np.linalg.norm(filteredArray-np.array([x,y]) , axis=1)

            if np.all(distances > self.min_corner_distance):
                filtered.append([x,y])

        return np.array(filtered, dtype=np.float32)

    #3. EDGE DETECTION

    def detect_edges(self, gray):

        """
        Detects edges using the Canny edge detector

        Parameters:
            gray: grayscale image

        Returns:
            edges: binary edge image
        """

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges


    #4. HOUGH LINE DETECTION

    def detect_hough_lines(self,gray):
        """
        Detects line segments using Hough line detection

         Parameters:
            gray: grayscale image

        Returns:
            lines: output from cv2.HoughLinesp
        """

        edges = self.detect_edges(gray)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines

    #5. SPLIT LINES BY ORIENTATION

    def split_lines_by_orientation(self, lines):

        """
        Split detected Hough lines into horizontal and vertical groups

        Parameters:
            lines: output from cv2.HoughLines

        Returns:
            horizontal_lines = list of horizontal line segments
            vertical_liens = list of vertical line segments
        """

        horizontal_lines = []
        vertical_lines = []

        if lines is None:
            return horizontal_lines, vertical_lines

        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1,x2-x1)))

            #Horizontal lines (near 0 and 180 degrees)
            if angle<20 or angle>160:
                horizontal_lines.append((x1,y1,x2,y2))

            #vertical lines (near 90 degrees)
            elif 70<angle<110:
                vertical_lines.append((x1,y1,x2,y2))

        return horizontal_lines, vertical_lines

    #6. CORNER DETECTION

    def compute_intersections(self, horizontal_lines, vertical_lines):

        """
        Computes intersection between horizontal and vertical lines
        Parameters:
            horizontal lines: List of horizontal lines
            vertical lines: List of vertical lines

        Returns:
            intersections: Array of intersection points numpy.ndarray (Nx2)
        """

        intersections = []

        for h in horizontal_lines:
            x1,y1,x2,y2 = h

            for v in vertical_lines:
                x3,y3,x4,y4 = v

                #If the denominator from the line intersection formula is 0, they intersect
                denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

                if abs(denom) < 1e-6:
                    continue
                px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

                intersections.append([px, py])

        return np.array(intersections, dtype=np.float32)

    def select_board_corners(self, intersections):

        """
        Selects the four outer corners of the board

        Parameters:
            intersections: List of intersections

        Returns:
            corners: returns four corners of the board
        """

        if len(intersections)<4:
            return None

        #smallest x+y is top-left, largest is bottom-right
        sum = intersections[:,0] + intersections[:,1]

        #smallest x-y is bottom-left, largest is top-right
        diff = intersections[:,0] - intersections[:,1]

        top_left = intersections[np.argmin(sum)]
        bottom_right = intersections[np.argmax(sum)]

        top_right = intersections[np.argmax(diff)]
        bottom_left = intersections[np.argmin(diff)]

        corners = np.array([ top_left, top_right, bottom_right, bottom_left ], dtype=np.float32)
        return corners

    def order_corners(self, corners):
        """
        Ensures corners are ordered consistently

        Parameters:
            corners: List of corners

        Returns:
            Returns chess board corners
        """

        rect = np.zeros((4,2), dtype=np.float32)

        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        rect[0] = corners[np.argmin(s)]     # top-left
        rect[2] = corners[np.argmax(s)]     # bottom-right
        rect[1] = corners[np.argmin(diff)]  # top-right
        rect[3] = corners[np.argmax(diff)]  # bottom-left

        return rect

    def run_stage1(self, image):
        gray = self.preprocess_image(image)
        corners = self.detect_harris_corners(gray)
        lines = self.detect_hough_lines(gray)
        horizontal_lines, vertical_lines = self.split_lines_by_orientation(lines)
        intersections = self.compute_intersections(horizontal_lines, vertical_lines)
        board_corners = self.select_board_corners(intersections)
        ordered_corners = self.order_corners(board_corners)

        return ordered_corners

