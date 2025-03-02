import cv2
import numpy as np

def apply_threshold(image):
    """
    Améliore le seuillage pour renforcer encore plus les lignes des marches.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    _, binary_otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
    )

    binary = cv2.bitwise_or(binary_otsu, adaptive_thresh)

    sobel_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.bitwise_or(cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y))
    normalized_sobel = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

    kernel_dilate = np.ones((2, 2), np.uint8)
    thick_edges = cv2.dilate(normalized_sobel, kernel_dilate, iterations=2)

    kernel_close = np.ones((3, 3), np.uint8)
    final_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return final_edges
