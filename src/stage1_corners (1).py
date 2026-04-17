"""
Stage 1: Corner Detection Module
Updated: Contour Aggregation Method

This module handles the detection of chessboard corners from input images
by finding valid inner squares and aggregating them to infer the outer boundary.
"""

import cv2
import numpy as np

class Stage1CornerDetection:
    """
    Advanced Stage 1 detector using Bottom-Up Contour Aggregation.
    
    Handles:
        - Canny Edge Detection & Thresholding
        - Contour Extraction (cv2.findContours)
        - Geometric filtering (4-sided, equal length checks)
        - Mask dilation to form a single board blob
        - Extreme point extraction to bypass corner occlusion
    """

    def __init__(
            self,
            canny_low=50,
            canny_high=150,
            min_square_area=1000,   # Adjusted for standard image sizes
            max_square_area=50000,
            length_tolerance=0.3,   # 30% tolerance for square side lengths
            dilation_kernel_size=7,
            dilation_iterations=2
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_square_area = min_square_area
        self.max_square_area = max_square_area
        self.length_tolerance = length_tolerance
        self.dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        self.dilation_iterations = dilation_iterations


    # 1. PREPROCESSING
    def preprocess_image(self, image):
        """Converts to grayscale and prepares for contour detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        
        # Dilate edges slightly to ensure contours are closed
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        return edges


    # 2. VALID SQUARE FILTERING
    def is_valid_square(self, approx_pts):
        """
        Translates C++ is_valid_square logic: Checks if 4 sides are roughly equal.
       
        """
        if len(approx_pts) != 4:
            return False
            
        pts = approx_pts.reshape(4, 2)
        lengths = []
        
        # Calculate distance between adjacent points
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            dist = np.linalg.norm(p1 - p2)
            lengths.append(dist)
            
        max_l = max(lengths)
        min_l = min(lengths)
        
        # Avoid division by zero
        if max_l == 0: 
            return False
            
        # Check if difference between longest and shortest side is within tolerance
        return (max_l - min_l) < (self.length_tolerance * max_l)

    def find_valid_squares(self, edges):
        """
        Finds all contours and filters them for valid chess squares.
       
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid_squares = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_square_area < area < self.max_square_area:
                # Approximate polygon to smooth out rough edges
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if self.is_valid_square(approx):
                    valid_squares.append(approx)
                    
        return valid_squares


    # 3. BLOB AGGREGATION & EXTREME POINTS
    def find_board_corners(self, valid_squares, image_shape):
        """
        Draws valid squares onto a blank mask, merges them, and finds extreme points.
       
        """
        if not valid_squares:
            return None
            
        # Create blank black image
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Fill the valid squares with white
        cv2.drawContours(mask, valid_squares, -1, 255, -1)
        
        # Dilate to merge individual squares into one massive blob
        mask = cv2.dilate(mask, self.dilation_kernel, iterations=self.dilation_iterations)
        
        # Find the contour of the new massive blob
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Get the largest blob
        biggest_contour = max(contours, key=cv2.contourArea)
        pts = biggest_contour.reshape(-1, 2)
        
        # Mathematical extreme points (ignores pieces blocking actual corners)
        s = pts.sum(axis=1)           # x + y
        diff = np.diff(pts, axis=1)   # y - x
        
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


    # 4. UTILITY
    def order_corners(self, corners):
        """Ensures consistent order: [TL, TR, BR, BL]"""
        if corners is None:
            return None
            
        rect = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        
        rect[0] = corners[np.argmin(s)]       # top-left
        rect[2] = corners[np.argmax(s)]       # bottom-right
        rect[1] = corners[np.argmin(diff)]    # top-right
        rect[3] = corners[np.argmax(diff)]    # bottom-left
        return rect


    # 5. MAIN EXECUTION
    def run_stage1(self, image):
        """
        Executes the full bottom-up contour aggregation pipeline.
        """
        edges = self.preprocess_image(image)
        valid_squares = self.find_valid_squares(edges)
        board_corners = self.find_board_corners(valid_squares, image.shape)
        ordered_corners = self.order_corners(board_corners)
        
        return ordered_corners