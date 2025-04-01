import cv2
import numpy as np


def count_stairs(image, detected_lines, y_threshold=30, min_length=50, min_y_gap=25):
    """
    Compte automatiquement les marches d'un escalier en analysant les lignes horizontales détectées.

    Entrées :
        - image 
        - detected_lines : liste des lignes détectées (coordonnées [x1,y1,x2,y2]).
        - y_threshold : seuil pour fusionner les lignes proches verticalement.
        - min_length : longueur minimale.
        - min_y_gap : distance verticale minimale entre deux marches.

    Sortie :
        - Nombre entier correspondant au nombre de marches détectées.
    """
    if detected_lines is None or len(detected_lines) == 0:
        print("Aucune ligne détectée !")
        return 0

    angles = []
    for line in detected_lines:
        if isinstance(line, (list, np.ndarray)) and len(line) == 4:
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)


    from scipy.stats import mode
    most_common_angle = mode(angles, keepdims=True).mode[0]
    print(f"Angle horizontal détecté : {most_common_angle:.2f} degrés")

    angle_tolerance = 10
    y_coordinates = []
    for line, angle in zip(detected_lines, angles):
        if abs(angle - most_common_angle) <= angle_tolerance:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                y_coordinates.append((y1 + y2) // 2)

    if len(y_coordinates) == 0:
        print(" Aucune ligne horizontale valide trouvée .")
        return 0

    y_coordinates = sorted(y_coordinates, reverse=True)
    filtered_y = []
    cluster = []
    for y in y_coordinates:
        if not cluster or abs(y - cluster[-1]) <= y_threshold:
            cluster.append(y)
        else:
            filtered_y.append(int(np.median(cluster)))
            cluster = [y]
    if cluster:
        filtered_y.append(int(np.median(cluster)))

    final_filtered_y = []
    for i, y in enumerate(filtered_y):
        if i == 0 or abs(y - final_filtered_y[-1]) > min_y_gap:
            final_filtered_y.append(y)

    stair_count = len(final_filtered_y)
    print(f"Nombre de marches détectées : {stair_count}")
    return stair_count
