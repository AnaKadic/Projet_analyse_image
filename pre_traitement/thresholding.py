import cv2
import numpy as np

def analyse_image(image):
    """
    Analyse les propriétés globales d'une image pour adapter le prétraitement.
    """
    results = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Luminosité moyenne
    mean_brightness = np.mean(gray)
    results["Trop lumineuse"] = mean_brightness > 180
    results["Trop sombre"] = mean_brightness < 50

    # Contraste
    results["Faible contraste"] = gray.std() < 30

    # Netteté
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    results["Floue"] = laplacian < 50

    # Orientation (via HoughLines)
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
    output = np.zeros_like(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    for i in range(1, num_labels):  # Ignorer le fond
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h if h > 0 else 0
        if w >= min_width and h >= min_height and aspect_ratio >= min_aspect_ratio:
            output[labels == i] = 255
    return output

def apply_threshold(image):
    """
    Prétraitement dynamique basé sur les caractéristiques globales de l’image.
    """
    infos = analyse_image(image)  # 🔍 Analyse intelligente

    # 🔧 Luminosité (si trop claire)
    if infos["Trop lumineuse"]:
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=-40)
        # Gamma correction
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)

    # Conversion en gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contraste CLAHE (si faible)
    if infos["Faible contraste"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Flou → on évite d’accentuer les bords (filtrage léger)
    if not infos["Floue"]:
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Seuillage
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 6
    )
    combined_thresh = cv2.bitwise_or(otsu, adaptive)

    # Sobel
    sobel_x = cv2.Sobel(combined_thresh, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(combined_thresh, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.bitwise_or(cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y))
    sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)

    # Morphologie
    dilated = cv2.dilate(sobel_norm, np.ones((2, 2), np.uint8), iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Filtrage composantes : attention si beaucoup de verticales (rampe ?)
    if infos["% lignes verticales"] > 30:
        final_edges = closed  # on évite un filtrage trop agressif
    else:
        final_edges = filter_connected_components(
            closed,
            min_width=10,
            min_height=1.5,
            min_aspect_ratio=3.0
        )

    return final_edges
