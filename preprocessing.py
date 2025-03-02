import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1.0):
    """Applique une correction gamma pour ajuster la luminosité."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def estimate_noise(image):
    """Estime le niveau de bruit de l'image en utilisant l'écart-type des pixels."""
    return np.std(image)

def get_kernel_size(noise_level):
    """Détermine la taille du noyau en fonction du niveau de bruit."""
    if noise_level < 10:
        return (3, 3)  # Faible bruit
    elif noise_level < 20:
        return (5, 5)  # Bruit modéré
    elif noise_level < 30:
        return (9, 9)  # Bruit élevé
    else:
        return (11, 11)  # Bruit très élevé

def adaptive_gaussian_blur(image):
    """Applique un flou gaussien avec une taille de noyau adaptative."""
    noise_level = estimate_noise(image)
    kernel_size = get_kernel_size(noise_level)
    print(f"Niveau de bruit : {noise_level:.2f}, Noyau choisi : {kernel_size}")
    return cv2.GaussianBlur(image, kernel_size, 0)

def preprocess_image(image_path):
    """Prétraitement de l'image pour la détection des marches."""
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou gaussien adaptatif
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Binarisation par Otsu après le flou gaussien
    _, otsu_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Post-traitement : Morphologie pour nettoyer le bruit (Opening) et combler les trous (Closing)
    kernel = np.ones((5, 5), np.uint8)

    # 1. Opening : Supprimer le bruit externe
    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)

    # 2. Closing : Combler les trous internes
    cleaned_image = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Détection des bords horizontaux avec Sobel Y
    sobel_y = cv2.Sobel(cleaned_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Affichage des résultats : Seuillage Otsu, Opening, Closing, Sobel Y
    titles = ['Image Originale', 'Flou Gaussien', 'Seuillage Otsu', 'Opening + Closing', 'Sobel Y (Bords horizontaux)']
    images = [gray_image, blurred_image, otsu_thresh, cleaned_image, sobel_y]

    plt.figure(figsize=(14, 12))
    for i in range(5):
        plt.subplot(3, 2, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

    return image, gray_image, blurred_image, otsu_thresh, cleaned_image, sobel_y

def detect_stairs(image_path):
    """Détecte et compte les marches d'escalier dans une image."""
    _, _, _, _, processed_image = preprocess_image(image_path)

    # Détection des contours avec Canny
    edges = cv2.Canny(processed_image, 50, 150)

    # Détection des lignes horizontales avec la transformée de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    stair_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Vérification de l'inclinaison pour garder les lignes horizontales
            if abs(y1 - y2) < 10:
                stair_count += 1

    print(f"Nombre de marches détectées : {stair_count}")
    return stair_count

def process_images(input_folder, output_folder):
    """Traite un ensemble d'images dans un dossier."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            print(f"\nTraitement de l'image : {filename}")
            stair_count = detect_stairs(image_path)
            print(f"{filename} → Marches détectées : {stair_count}")
