import cv2
import numpy as np

def apply_canny(image):
    """
    Applique l'algorithme de Canny après le seuillage.
    - Se base sur l’image déjà seuillée pour extraire les contours.
    """
    # Vérifier si l'image est déjà en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Appliquer la détection des contours de Canny
    edges = cv2.Canny(gray, 30, 120)
     # 2️⃣ Dilatation légère pour rendre les contours plus visibles
    kernel = np.ones((2, 2), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)

    # 3️⃣ Fusionner les contours de Canny avec l’image seuillée (optionnel)
    enhanced_edges = cv2.bitwise_or(thick_edges, gray)


    return  enhanced_edges 
