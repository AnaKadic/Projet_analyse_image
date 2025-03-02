import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chemins des répertoires
base_dir = os.path.expanduser("~/Documents/M1/s2/analyse d'image/Projet_analyse_image")
images_dir = os.path.join(base_dir, "images")
csv_path = os.path.join(base_dir, "annotations", "data annotations - Feuille 1.csv")

# Charger les annotations
annotations = pd.read_csv(csv_path, sep=',', encoding='utf-8')
annotations.columns = annotations.columns.str.strip()

# Filtrer les images des Groupes 2 et 3 (7 images seulement)
filtered_data = annotations[annotations['Identifiant équipe'].isin(['Groupe2', 'Groupe3'])].head(7)

# Fonction de prétraitement et de détection
def preprocess_and_detect(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {image_path}")

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flou Gaussien
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des contours avec Canny
    canny_edges = cv2.Canny(blurred, 80, 240)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=5)

    # Filtrer les lignes horizontales
    detected_steps = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 20:  # Lignes horizontales seulement
                if all(abs(y1 - step) > 20 for step in detected_steps):
                    detected_steps.append(y1)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    stair_count = len(detected_steps)

    return image, canny_edges, stair_count

# Stockage des résultats
results = []

# Traitement des images
for _, row in filtered_data.iterrows():
    image_name = row['Nom image'].strip()
    true_count = int(row['Nombre de marches'])
    group = row['Identifiant équipe']
    group_folder = os.path.join(images_dir, group.replace("Groupe", "Groupe "))
    image_path = os.path.join(group_folder, image_name)

    if os.path.exists(image_path):
        print(f"🔍 Traitement de l'image : {image_name}")
        try:
            processed_img, edges_img, detected_count = preprocess_and_detect(image_path)

            # Stocker les résultats
            results.append({
                "Image": image_name,
                "Groupe": group,
                "Vérité Terrain": true_count,
                "Marches Détectées": detected_count,
                "Écart": detected_count - true_count
            })

            # Affichage des résultats
            plt.figure(figsize=(18, 10))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Image originale avec lignes détectées ({detected_count} marches)")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(edges_img, cmap="gray")
            plt.title("Contours Canny")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.text(0.1, 0.5, f"Vérité Terrain: {true_count}\nMarches Détectées: {detected_count}\nÉcart: {detected_count - true_count}", fontsize=12)
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ Erreur lors du traitement de {image_name} : {e}")

# Résultats finaux
results_df = pd.DataFrame(results)
print("\n📊 Résumé des performances :")
print(results_df)

# Exporter les résultats en CSV
output_csv = os.path.join(base_dir, "resultats_detection_marches.csv")
results_df.to_csv(output_csv, index=False)
print(f"\n✅ Fichier de résultats enregistré : {output_csv}")
