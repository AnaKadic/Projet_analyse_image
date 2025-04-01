import cv2
import numpy as np

def detect_all_lines(edges, min_length=150, max_gap=30, threshold=50, dilate=True):
    """
    Détecte toutes les lignes dans une image avec HoughLinesP.
    
    Params :
        - edges : image binaire
        - min_length : longueur minimale d'une ligne
        - max_gap : distance entre segments
        - threshold : nombre de votes pour qu'une ligne soit retenue
        - dilate : booléen, appliquer ou non une dilatation

    Retour :
        - image annotée
        - liste des lignes brutes [x1, y1, x2, y2]
    """

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    image_with_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append([x1, y1, x2, y2])
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_with_lines, detected_lines
