import cv2
import numpy as np

def extract_features_image(image):
    results = {}

    # 1. Luminosité moyenne
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    results["Luminosité moyenne"] = round(mean_brightness, 2)
    results["Trop lumineuse"] = mean_brightness > 180
    results["Trop sombre"] = mean_brightness < 50

    # 2. Contraste global
    contrast = gray.std()
    results["Contraste global (std)"] = round(contrast, 2)
    results["Faible contraste"] = contrast < 30

    # 3. Flou (détection avec Laplacien)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    results["Niveau de netteté (Laplacian)"] = round(laplacian_var, 2)
    results["Floue"] = laplacian_var < 50

    # 4. Orientation dominante (Hough Transform)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        angles = [np.rad2deg(theta[0][1]) for theta in lines]
        vertical_lines = [angle for angle in angles if 75 <= angle <= 105]
        horizontal_lines = [angle for angle in angles if angle <= 15 or angle >= 165]

        results["Nb lignes détectées"] = len(lines)
        results["% lignes horizontales"] = round(len(horizontal_lines) / len(lines) * 100, 2)
        results["% lignes verticales"] = round(len(vertical_lines) / len(lines) * 100, 2)
    else:
        results["Nb lignes détectées"] = 0
        results["% lignes horizontales"] = 0
        results["% lignes verticales"] = 0

    return results
