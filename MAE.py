import os
import cv2
import pandas as pd
import numpy as np
from pre_traitement.thresholding import apply_threshold
from pre_traitement.canny import apply_canny
from pre_traitement.hough import detect_all_lines
from pre_traitement.count_stairs import count_stairs

# Chemins des répertoires (chemins relatifs)
images_dir = "images"
csv_path = os.path.join("annotations", "data annotations - Feuille 1.csv")

# Charger les données
ground_truth = pd.read_csv(csv_path, sep=',', encoding='utf-8')
ground_truth.columns = ground_truth.columns.str.strip()

# Nettoyer et normaliser les noms dans le CSV
ground_truth['Nom image'] = ground_truth['Nom image'].str.strip().str.lower()

# Stocker les résultats
detected_results = []

# Fonction pour traiter une image et compter les marches
def process_image(image_path, true_stairs):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l’image : {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = apply_threshold(image)
    edges_image = apply_canny(thresholded_image)
    hough_image, detected_lines = detect_all_lines(edges_image, min_length=80)
    stair_count = count_stairs(image, detected_lines, y_threshold=42, min_length=120, min_y_gap=15)

    return stair_count

# Traitement des images
for _, row in ground_truth.iterrows():  # Parcourir toutes les lignes du CSV
    image_name = row['Nom image'].strip()
    true_stairs = int(row['Nombre de marches'])
    group = row['Identifiant équipe']
    group_folder = os.path.join(images_dir, group.replace("Groupe", "Groupe "))

    image_path = os.path.join(group_folder, image_name)

    if os.path.exists(image_path):
        print(f"Traitement de l'image : {image_path}")

        # Traiter l'image
        detected_count = process_image(image_path, true_stairs)
        if detected_count is not None:
            detected_results.append({
                "Image": image_name,
                "Groupe": group,
                "Marches détectées": detected_count,
                "Vérité Terrain": true_stairs,
                "Écart": abs(detected_count - true_stairs)
            })
    else:
        print(f"Erreur : Image non trouvée : {image_path}")

# Calcul du MAE global
if detected_results:
    mae_global = np.mean([result['Écart'] for result in detected_results])
    print(f"\nErreur Absolue Moyenne (MAE) globale : {mae_global}")
else:
    print("Aucun résultat à évaluer.")

# Calcul de la moyenne des marches en vérité terrain
if detected_results:
    moyenne_verite_terrain = np.mean([result['Vérité Terrain'] for result in detected_results])
    print(f"Moyenne des marches en vérité terrain : {moyenne_verite_terrain}")

    # Comparaison du MAE global avec la moyenne des marches en vérité terrain
    print(f"\nComparaison :")
    print(f"MAE global : {mae_global}")
    print(f"Moyenne des marches en vérité terrain : {moyenne_verite_terrain}")
    print(f"MAE global représente {mae_global / moyenne_verite_terrain * 100:.2f}% de la moyenne des marches en vérité terrain.")
else:
    print("Aucun résultat à évaluer.")

# Affichage des résultats
if detected_results:
    results_df = pd.DataFrame(detected_results)
    print("\nRésumé des performances :")
    print(results_df[['Image', 'Groupe', 'Vérité Terrain', 'Marches détectées', 'Écart']])

    # Export des résultats
    output_path = os.path.join("resultats_comparatif_tous_groupes.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Fichier de résultats enregistré : {output_path}")
else:
    print("Aucun résultat à exporter.")