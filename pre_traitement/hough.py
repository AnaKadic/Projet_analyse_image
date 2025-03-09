import cv2
import numpy as np

def detect_all_lines(image, min_length=150, max_gap=30):
    """
    Détecte toutes les lignes dans une image.
    Retourne :
    - L'image avec les lignes dessinées
    - La liste brute des lignes détectées
    """

    # Convertir l'image en niveaux de gris si ce n'est pas déjà fait
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Détecter les contours avec Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Dilater les contours pour améliorer la détection des lignes
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Détecter les lignes avec HoughLinesP
    lines = cv2.HoughLinesP(
        edges_dilated,
        rho=1,  # Résolution de la distance en pixels
        theta=np.pi / 180,  # Résolution de l'angle en radians
        threshold=50,  # Seuil de détection
        minLineLength=min_length,  # Longueur minimale d'une ligne
        maxLineGap=max_gap  # Écart maximal entre les segments de ligne
    )

    # Créer une image en couleur pour dessiner les lignes
    image_with_lines = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Liste pour stocker les lignes détectées
    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Ajouter la ligne à la liste
            detected_lines.append([x1, y1, x2, y2])
            # Dessiner la ligne sur l'image
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Ligne verte, épaisseur 2

    return image_with_lines, detected_lines