import cv2
import numpy as np

import cv2
import numpy as np

def detect_stair_rectangles(lines, image):
    """Détecte les paires de lignes pour former des rectangles correspondant aux marches."""
    if lines is None:
        return 0, image

    filtered_lines = []
    y_positions = []

    # Filtrer les lignes horizontales
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10 and abs(x2 - x1) > 50:  # Lignes horizontales assez longues
            filtered_lines.append((x1, y1, x2, y2))
            y_positions.append(y1)

    # Trier les lignes par position verticale (haut vers bas)
    filtered_lines.sort(key=lambda x: x[1])

    # Associer les lignes pour former des rectangles
    rectangles = []
    used = set()

    for i, (x1, y1, x2, y2) in enumerate(filtered_lines):
        if i in used:
            continue
        for j, (x1_bis, y1_bis, x2_bis, y2_bis) in enumerate(filtered_lines):
            if i != j and j not in used:
                # Vérifier la proximité verticale
                if 10 <= abs(y1 - y1_bis) <= 70:  # Hauteur raisonnable pour une marche
                    rectangles.append(((x1, y1), (x2, y2), (x1_bis, y1_bis), (x2_bis, y2_bis)))
                    used.add(i)
                    used.add(j)
                    break

    # Dessiner les rectangles sur l'image
    output_image = image.copy()
    for rect in rectangles:
        (x1, y1), (x2, y2), (x1_bis, y1_bis), (x2_bis, y2_bis) = rect
        cv2.rectangle(output_image, (x1, y1), (x2_bis, y1_bis), (0, 255, 0), 2)

    stair_count = len(rectangles)

    return stair_count, output_image
