import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_image
from detection import detect_stair_rectangles
from evaluation import evaluate_performance
import numpy as np
from pre_traitement.thresholding import apply_threshold  # Seuillage
from pre_traitement.canny import apply_canny  # Canny
from pre_traitement.hough import detect_horizontal_lines  # Hough
from pre_traitement.count_stairs import count_stairs  # ✅ Correction de l'import

"""
# 🔧 Chemins des répertoires
base_dir = os.path.expanduser("~/Documents/M1/s2/analyse d'image/Projet_analyse_image")
images_dir = os.path.join(base_dir, "images")
csv_path = os.path.join(base_dir, "annotations", "data annotations - Feuille 1.csv")

# ✅ Charger les données
ground_truth = pd.read_csv(csv_path, sep=',', encoding='utf-8')
ground_truth.columns = ground_truth.columns.str.strip()

# ✅ Filtrer les images des Groupes 2 et 3 (7 images seulement)
filtered_data = ground_truth[ground_truth['Identifiant équipe'].isin(['Groupe2', 'Groupe3'])].head(7)

# ✅ Stocker les résultats
detected_results = []

# 🚀 Traitement des images
for _, row in filtered_data.iterrows():
    image_name = row['Nom image'].strip()
    true_stairs = int(row['Nombre de marches'])
    group = row['Identifiant équipe']
    group_folder = os.path.join(images_dir, group.replace("Groupe", "Groupe "))

    image_path = os.path.join(group_folder, image_name)

    if os.path.exists(image_path):
        print(f"🔍 Traitement de l'image : {image_path}")

        try:
            # Prétraitement
            original_image, gray_image, blurred_image, otsu_thresh, cleaned_image, sobel_y = preprocess_image(image_path)

            # Détection des contours avec Canny
            edges = cv2.Canny(cleaned_image, 100, 200)

            # Détection des lignes avec HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=10)

            # Appel correct de detect_stair_rectangles avec les lignes et l'image
            detected_count, output_image = detect_stair_rectangles(lines, cleaned_image)

            # Stockage des résultats
            detected_results.append({
                "Image": image_name,
                "Groupe": group,
                "Marches détectées": detected_count,
                "Vérité Terrain": true_stairs,
                "Écart": detected_count - true_stairs
            })

            # Affichage des résultats
            plt.figure(figsize=(18, 12))
            plt.subplot(2, 4, 1)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Image originale : {image_name}")
            plt.axis("off")

            plt.subplot(2, 4, 2)
            plt.imshow(gray_image, cmap='gray')
            plt.title("Image en niveaux de gris")
            plt.axis("off")

            plt.subplot(2, 4, 3)
            plt.imshow(blurred_image, cmap='gray')
            plt.title("Flou Gaussien")
            plt.axis("off")

            plt.subplot(2, 4, 4)
            plt.imshow(otsu_thresh, cmap='gray')
            plt.title("Seuillage Otsu")
            plt.axis("off")

            plt.subplot(2, 4, 5)
            plt.imshow(sobel_y, cmap='gray')
            plt.title("Sobel Y (Bords horizontaux)")
            plt.axis("off")

            plt.subplot(2, 4, 6)
            plt.imshow(edges, cmap='gray')
            plt.title("Contours Canny")
            plt.axis("off")

            plt.subplot(2, 4, 7)
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Détection des marches : {detected_count} détectées")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ Erreur lors du traitement de l'image {image_name} : {e}")

    else:
        print(f"⚠️ Image non trouvée : {image_path}")

# 📈 Évaluation des performances
results_df = pd.DataFrame(detected_results)
print("\n📊 **Résumé des performances :**")
print(results_df[['Image', 'Groupe', 'Vérité Terrain', 'Marches détectées', 'Écart']])

# ✅ Export des résultats
output_path = os.path.join(base_dir, "resultats_comparatif_groupes2_3.csv")
results_df.to_csv(output_path, index=False)
print(f"\n✅ Fichier de résultats enregistré : {output_path}")
"""

image_path = "/home/user/Documents/M1/s2/analyse d'image/Projet_analyse_image/images/Groupe 2/Groupe2_Image3.jpeg"

image = cv2.imread(image_path)

if image is None:
    print(f"⚠️ Impossible de charger l’image : {image_path}")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 🔹 Étape 1 :
thresholded_image = apply_threshold(image)

# 🔹 Étape 2 : 
edges_image = apply_canny(thresholded_image)

# 🔹 Étape 3 :
hough_image, detected_lines = detect_horizontal_lines(edges_image, min_length=80, min_y_gap=10)


stair_count = count_stairs(image, detected_lines, y_threshold= 42, min_length=120, min_y_gap=15)

fig, axes = plt.subplots(1, 5, figsize=(15, 5))

axes[0].imshow(gray_image, cmap="gray")
axes[0].set_title("Image en Niveaux de Gris")
axes[0].axis("off")

axes[1].imshow(thresholded_image, cmap="gray")
axes[1].set_title("Image Seuillée (Corrigée)")
axes[1].axis("off")

axes[2].imshow(edges_image, cmap="gray")
axes[2].set_title("Contours après Canny")
axes[2].axis("off")

axes[3].imshow(hough_image)
axes[3].set_title("Lignes Hough (Horizontales)")
axes[3].axis("off")

axes[4].text(0.5, 0.5, f"Nombre de marches détectées : {stair_count}", 
             fontsize=15, ha='center', va='center')
axes[4].set_title("Comptage des Marches")
axes[4].axis("off")

plt.tight_layout()
plt.show()