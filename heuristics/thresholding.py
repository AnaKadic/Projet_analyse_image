import cv2
import numpy as np
from scipy.signal import find_peaks


def analyse_image(image):
    """
    Analyse les propriétés d'une image pour adapter le prétraitement.

    Entrée :
        - image : Image couleur chargée avec OpenCV 

    Sortie :
        - dict : luminosité moyenne, contraste,  netteté, pourcentage de lignes verticales détectées
    """
    results = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)
    results["Trop lumineuse"] = mean_brightness > 180
    results["Trop sombre"] = mean_brightness < 50

    results["Faible contraste"] = gray.std() < 30

    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    results["Floue"] = laplacian < 50

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        angles = [np.rad2deg(theta[0][1]) for theta in lines]
        verticals = [a for a in angles if 75 <= a <= 105]
        results["% lignes verticales"] = len(verticals) / len(lines) * 100
    else:
        results["% lignes verticales"] = 0

    return results

def filter_connected_components(image, min_width=40, min_height=2, min_aspect_ratio=3.0):
    """
    Filtre les composantes connexes sur une image binaire.

    Entrée :
        - image : Image binaire
        - min_width : largeur minimale acceptée
        - min_height : hauteur minimale acceptée
        - min_aspect_ratio : rapport L/l minimal pour considérer un composant valide

    Sortie :
        - image binaire contenant uniquement les composantes filtrées
    """
    output = np.zeros_like(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    for i in range(1, num_labels):  # Ignorer le fond
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h if h > 0 else 0
        if w >= min_width and h >= min_height and aspect_ratio >= min_aspect_ratio:
            output[labels == i] = 255
    return output

def detect_marches_par_projection(binary_img, prominence=50, distance=15):
    """
    Détecte les lignes horizontales significatives via projection verticale.
    - prominence : force des pics détectés (proportion de pixels blancs).
    - distance : espacement minimal entre pics (marches).

    Retourne : image filtrée contenant uniquement les lignes détectées.
    """
    projection = np.sum(binary_img == 255, axis=1)  # somme des pixels blancs par ligne
    peaks, _ = find_peaks(projection, prominence=prominence, distance=distance)

    result = np.zeros_like(binary_img)
    for y in peaks:
        result[max(0, y - 1):y + 2, :] = 255 

    return result


def apply_threshold(image):
    """
    Prétraitement et adapte les traitements selon ses caractéristiques.

    Entrée :
        - image 

    Sortie :
        - Image binaire filtrée avec les lignes.
    """
    infos = analyse_image(image)

    if infos["Trop lumineuse"]:
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=-40)
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if infos["Faible contraste"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if not infos["Floue"]:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 6)
    combined_thresh = cv2.bitwise_or(otsu, adaptive)

    sobel_x = cv2.Sobel(combined_thresh, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(combined_thresh, cv2.CV_64F, 0, 1, ksize=3)
    vertical_edges = cv2.convertScaleAbs(sobel_x)
    horizontal_edges = cv2.convertScaleAbs(sobel_y)

    is_top_down_view = infos["% lignes verticales"] > 70 and gray.std() < 50

    if is_top_down_view:
        print("Vue du dessus détectée : traitement spécial vertical.")
        filtered = vertical_edges.copy()
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8), iterations=1)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

        final_edges = filter_connected_components(
            filtered,
            min_width=2,
            min_height=20,
            min_aspect_ratio=0.2
        )
        return final_edges

    is_complex = infos["% lignes verticales"] < 10 and gray.std() > 40
    sobel_combined = cv2.bitwise_or(horizontal_edges, vertical_edges)
    sobel_norm = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

    if is_complex:
        morph = cv2.morphologyEx(horizontal_edges, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    else:
        morph = sobel_norm.copy()

    closed = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    if is_complex:
        final_edges = filter_connected_components(closed, min_width=20, min_height=2, min_aspect_ratio=4.0)
    elif infos["% lignes verticales"] > 30:
        final_edges = closed
    else:
        final_edges = filter_connected_components(closed, min_width=10, min_height=1.5, min_aspect_ratio=3.0)

    if not is_top_down_view:
        final_edges = detect_marches_par_projection(final_edges, prominence=50, distance=15)

    return final_edges
