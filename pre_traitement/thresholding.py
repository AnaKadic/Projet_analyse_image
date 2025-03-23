import cv2
import numpy as np

def is_overexposed(image, threshold=200, percentage=0.25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > threshold)
    total_pixels = gray.size
    return (bright_pixels / total_pixels) > percentage

def filter_connected_components(image, min_width=40, min_height=2, min_aspect_ratio=3.0):
    """
    Supprime les composantes non significatives.
    On garde uniquement les objets allongés en largeur (lignes de marches).
    """
    output = np.zeros_like(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    for i in range(1, num_labels):  # Ignorer le fond (label 0)
        x, y, w, h, area = stats[i]

        # Calculer le ratio largeur/hauteur
        aspect_ratio = w / h if h > 0 else 0

        # Garder uniquement les objets larges et plats (lignes)
        if w >= min_width and h >= min_height and aspect_ratio >= min_aspect_ratio:
            output[labels == i] = 255

    return output

def apply_threshold(image):
    if is_overexposed(image):
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=-20)
        gamma = 2.0
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        image = cv2.LUT(image, table)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    _, otsu = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6
    )
    combined_thresh = cv2.bitwise_or(otsu, adaptive)

    # Sobel
    sobel_x = cv2.Sobel(combined_thresh, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(combined_thresh, cv2.CV_64F, 0, 1, ksize=3)
    combined_sobel = cv2.bitwise_or(cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y))
    sobel_norm = cv2.normalize(combined_sobel, None, 0, 255, cv2.NORM_MINMAX)

    # Morphologie
    kernel_dilate = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(sobel_norm, kernel_dilate, iterations=3)

    kernel_close = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # ✅ Renforcement du filtrage des composantes
    final_edges = filter_connected_components(
        morph,
        min_width=10,          
        min_height=1.5,          
        min_aspect_ratio=3.0   
    )

    return final_edges
