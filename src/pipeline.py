"""
Chess Vision Pipeline
Member A

This module connects all three stages into a complete pipeline.
"""

import argparse
import cv2
import numpy as np

from stage1_corners import detect_corners
from stage2_warp import compute_homography, warp_image
from stage3_classify import load_model, classify_piece


def process_image(image_path, model_path):
    """
    Process a chess image through the complete pipeline.
    
    Args:
        image_path (str): Path to input chess image
        model_path (str): Path to trained classification model
        
    Returns:
        board_state (dict): Dictionary representing the board state
    """
    # Stage 1: Detect corners
    print("Stage 1: Detecting corners...")
    corners = detect_corners(image_path)
    
    # Stage 2: Apply homography and warp
    print("Stage 2: Applying homography transformation...")
    H = compute_homography(corners)
    image = cv2.imread(image_path)
    warped = warp_image(image, H)
    
    # Stage 3: Classify pieces
    print("Stage 3: Classifying pieces...")
    model = load_model(model_path)
    # TODO: Split warped image into 64 squares and classify each
    
    print("Pipeline complete!")
    return {}


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Chess Vision Pipeline')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input chess image')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Path to trained classification model')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output results')
    
    args = parser.parse_args()
    
    board_state = process_image(args.image, args.model)
    
    if args.output:
        # TODO: Save results to file
        print(f"Results saved to {args.output}")
    
    return board_state


if __name__ == "__main__":
    main()
