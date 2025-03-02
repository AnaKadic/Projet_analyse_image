import cv2
import numpy as np

def count_stairs(image, detected_lines, y_threshold=30, min_length= 50, min_y_gap=25):
    """
    Compte le nombre de marches en fusionnant plus fortement les lignes détectées.
    - detected_lines : liste des lignes détectées.
    - y_threshold : distance max entre deux lignes pour être fusionnées.
    - min_length : longueur minimale des lignes détectées.
    - min_y_gap : distance minimale entre deux lignes après regroupement.
    """

    if detected_lines is None or len(detected_lines) == 0:
        print("⚠️ Aucune ligne détectée !")
        return 0

    y_coordinates = []

    # 🔹 1️⃣ Filtrer les lignes trop courtes et extraire `y`
    for line in detected_lines:
        if isinstance(line, (list, np.ndarray)) and len(line) == 4:
            x1, y1, x2, y2 = line


            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= min_length:  # Garder seulement les grandes lignes
                y_coordinates.append((y1 + y2) // 2)  # Moyenne pour stabiliser

    if len(y_coordinates) == 0:
        print("⚠️ Aucune ligne horizontale valide trouvée !")
        return 0

    y_coordinates = sorted(y_coordinates, reverse=True)

    # 🔹 3️⃣ Fusionner les lignes proches de manière plus agressive (clustering fort)
    filtered_y = []
    cluster = []  # Temporaire pour regrouper les lignes proches

    for y in y_coordinates:
        if not cluster or abs(y - cluster[-1]) <= y_threshold:
            cluster.append(y)  # Ajouter au cluster actuel
        else:
            # Ajouter la médiane du cluster dans la liste finale pour éviter les décalages
            filtered_y.append(int(np.median(cluster)))
            cluster = [y]  # Créer un nouveau cluster

    # Ajouter le dernier cluster trouvé
    if cluster:
        filtered_y.append(int(np.median(cluster)))

    # 🔹 4️⃣ Supprimer les lignes trop denses après fusion (éviter encore les doublons)
    final_filtered_y = []
    for i, y in enumerate(filtered_y):
        if i == 0 or abs(y - final_filtered_y[-1]) > min_y_gap:
            final_filtered_y.append(y)

    stair_count = len(final_filtered_y)

    print(f"🔢 Nombre de marches détectées : {stair_count}")
    
    return stair_count
