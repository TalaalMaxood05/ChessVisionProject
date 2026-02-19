"""
Utility Functions
Shared helper functions used across multiple stages.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load an image from disk.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        image (numpy.ndarray): Loaded image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image


def save_image(image, output_path):
    """
    Save an image to disk.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path to save the image
    """
    cv2.imwrite(output_path, image)


def visualize_corners(image, corners, output_path=None):
    """
    Visualize detected corners on the image.
    
    Args:
        image (numpy.ndarray): Input image
        corners (numpy.ndarray): Corner coordinates
        output_path (str, optional): Path to save visualization
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if corners is not None:
        plt.plot(corners[:, 0], corners[:, 1], 'ro', markersize=10)
    plt.title('Detected Corners')
    
    if output_path:
        plt.savefig(output_path)
    plt.show()


def split_board_into_squares(warped_image):
    """
    Split a warped chessboard image into 64 individual squares.
    
    Args:
        warped_image (numpy.ndarray): Rectified chessboard image
        
    Returns:
        squares (list): List of 64 square images
    """
    # TODO: Implement square extraction
    pass


def chess_notation_to_index(notation):
    """
    Convert chess notation (e.g., 'e4') to array index.
    
    Args:
        notation (str): Chess notation (e.g., 'e4')
        
    Returns:
        row, col (tuple): Array indices
    """
    col = ord(notation[0].lower()) - ord('a')
    row = 8 - int(notation[1])
    return row, col


def index_to_chess_notation(row, col):
    """
    Convert array indices to chess notation.
    
    Args:
        row, col (int): Array indices
        
    Returns:
        notation (str): Chess notation (e.g., 'e4')
    """
    file = chr(ord('a') + col)
    rank = str(8 - row)
    return file + rank
