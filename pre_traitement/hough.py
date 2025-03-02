import cv2
import numpy as np

def detect_horizontal_lines(image, min_length=150, angle_tolerance=35, max_gap=30, min_y_gap=10):
    """
    Détecte les grandes lignes quasi-horizontales en tenant compte de la perspective.
    Retourne :
    - L'image avec les lignes dessinées
    - La liste brute des lignes détectées après filtrage
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges_dilated, 1, np.pi / 180, threshold=50, minLineLength=min_length, maxLineGap=max_gap)

    image_with_lines = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    filtered_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if abs(angle) <= angle_tolerance:
                filtered_lines.append([x1, y1, x2, y2])
        filtered_lines.sort(key=lambda l: (l[1] + l[3]) // 2, reverse=True)

        final_lines = []
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            y_avg = (y1 + y2) // 2

            if not final_lines or abs(y_avg - (final_lines[-1][1] + final_lines[-1][3]) // 2) > min_y_gap:
                final_lines.append(line)
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  #

    return image_with_lines, final_lines  
