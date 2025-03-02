import cv2
import numpy as np

def detect_horizontal_lines(image, min_length=150, angle_tolerance=25, max_gap=30):
    """
    Détecte les grandes lignes quasi-horizontales en tenant compte de la perspective.
    - min_length : longueur minimale des lignes détectées.
    - angle_tolerance : tolérance en degrés pour capturer des lignes inclinées.
    - max_gap : permet de combler les interruptions dans les lignes détectées.
    """

    # 🔹 Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 🔹 Détection des contours avec Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 🔹 Dilatation pour renforcer les lignes continues
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 🔹 Détection des lignes avec Hough
    lines = cv2.HoughLinesP(edges_dilated, 1, np.pi / 180, threshold=50, minLineLength=min_length, maxLineGap=max_gap)

    # 🔹 Convertir en image couleur pour affichage
    image_with_lines = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 🔹 Filtrer et dessiner uniquement les grandes lignes quasi-horizontales
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calcul de l'angle de la ligne
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Vérifier si l'angle est proche de l'horizontale avec une tolérance large
            if abs(angle) <= angle_tolerance:
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert

    return image_with_lines


