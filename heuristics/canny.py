import cv2
import numpy as np

def apply_canny(image):
    """
    Applique l'algorithme de Canny après le seuillage.
    - Se base sur l’image déjà seuillée pour extraire les contours.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 30, 120)
    kernel = np.ones((3,3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)

    enhanced_edges = cv2.bitwise_or(thick_edges, gray)


    return  enhanced_edges 
