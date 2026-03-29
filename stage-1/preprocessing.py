"""
Preprocessing module for CAD drawing images.
Performs grayscale conversion, denoising, thresholding, deskewing, contrast normalization, border removal, and optional tiling.
"""
import cv2
import numpy as np
from typing import Optional


def preprocess_image(path: str, tile: bool = False, tile_size: int = 2048) -> np.ndarray:
    """
    Preprocess a CAD drawing image for text extraction.
    Args:
        path (str): Path to the input image (PNG).
        tile (bool): Whether to tile large images into crops.
        tile_size (int): Size of each tile if tiling is enabled.
    Returns:
        np.ndarray: Preprocessed image (single tile or full image).
    """
    # Load image
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Thresholding (adaptive for robustness)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    # Deskew
    coords = np.column_stack(np.where(thresh < 255))
    angle = 0
    if coords.shape[0] > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = thresh

    # Normalize contrast
    norm = cv2.normalize(deskewed, None, 0, 255, cv2.NORM_MINMAX)

    # Remove borders
    contours, _ = cv2.findContours(norm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = norm[y:y+h, x:x+w]
    else:
        cropped = norm

    # Optional: tile large images
    if tile and (cropped.shape[0] > tile_size or cropped.shape[1] > tile_size):
        # For simplicity, return the first tile (extend as needed)
        cropped = cropped[0:tile_size, 0:tile_size]

    return cropped
