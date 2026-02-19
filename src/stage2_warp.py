"""
Stage 2: Homography and Rectification Module
Members A & B

This module handles homography transformation and board rectification.
"""

import cv2
import numpy as np


def compute_homography(corners):
    """
    Compute homography matrix from detected corners.
    
    Args:
        corners (numpy.ndarray): Corner coordinates from stage 1
        
    Returns:
        H (numpy.ndarray): Homography matrix
    """
    # TODO: Implement homography computation
    pass


def warp_image(image, H):
    """
    Apply homography transformation to rectify the chessboard.
    
    Args:
        image (numpy.ndarray): Input image
        H (numpy.ndarray): Homography matrix
        
    Returns:
        warped_image (numpy.ndarray): Rectified chessboard image
    """
    # TODO: Implement image warping
    pass


def main():
    """Main function for testing homography and warping."""
    # TODO: Implement testing logic
    pass


if __name__ == "__main__":
    main()
