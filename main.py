import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_image
from detection import detect_stair_rectangles
from evaluation import evaluate_performance

from sklearn.metrics import mean_absolute_error
import numpy as np
from pre_traitement.thresholding import apply_threshold  # Seuillage
from pre_traitement.canny import apply_canny  # Canny
from pre_traitement.hough import detect_all_lines  # Hough
from pre_traitement.count_stairs import count_stairs  # ✅ Correction de l'import
from pre_traitement.analyse_image import analyse_image  # Assurez-vous que cette fonction est bien importée


# 📁 Dossiers
base_folder = "/home/user/Documents/M1/s2/analyse d'image/Projet_analyse_image/images"
annotations_path = "/home/user/Documents/M1/s2/analyse d'image/Projet_analyse_image/annotations/data annotations - Feuille 1.csv"

# 📄 Charger la vérité terrain
annotations_df = pd.read_csv(annotations_path, delimiter=",", encoding="utf-8")
annotations_df.columns = ["Nom image", "Nombre de marches", "Identifiant équipe"]

# 🔍 Extensions autorisées
extensions_autorisees = (".jpg", ".jpeg", ".png")

# 🔁 Données pour MAE
true_counts = []
predicted_counts = []
image_names = []
differences = []

# 🔄 Parcours des images
for root, _, files in os.walk(base_folder):
    for file_name in files:
        if not file_name.lower().endswith(extensions_autorisees):
            continue

        image_path = os.path.join(root, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Impossible de charger l’image : {image_path}")
            continue

        print(f"📸 Traitement de : {file_name}")

        # 🔍 Trouver la vérité terrain correspondante
        match = annotations_df[annotations_df["Nom image"].str.lower() == file_name.lower()]
        if match.empty:
            print(f"⚠️ Pas d'annotation pour l'image : {file_name}")
            continue
        true = int(match["Nombre de marches"].values[0])

        # 🧪 Traitement
        thresholded = apply_threshold(image)
        edges = apply_canny(thresholded)
        _, lines = detect_all_lines(edges, min_length=80)
        predicted = count_stairs(image, lines, y_threshold=42, min_length=120, min_y_gap=15)

        # 📊 Stocker les résultats
        image_names.append(file_name)
        true_counts.append(true)
        predicted_counts.append(predicted)
        differences.append(abs(true - predicted))

# 📏 Calcul du MAE
mae = mean_absolute_error(true_counts, predicted_counts)
print(f"\n📏 Erreur Absolue Moyenne (MAE) sur {len(predicted_counts)} images : {mae:.2f} marches")

# 💾 Sauvegarde des résultats
df = pd.DataFrame({
    "Image": image_names,
    "Marches détectées": predicted_counts,
    "Vérité Terrain": true_counts,
    "Écart": differences
})

df.to_csv("resultats_marche_detectees.csv", index=False, encoding="utf-8")
print("✅ Résultats enregistrés dans resultats_marche_detectees.csv")
"""

# 📁 Dossier à analyser
base_folder = "/home/user/Documents/M1/s2/analyse d'image/Projet_analyse_image/images"
extensions_autorisees = (".jpg", ".jpeg", ".png")

# 📊 Résultats à stocker
analysis_results = []

# 🔁 Analyse de toutes les images
for root, _, files in os.walk(base_folder):
    for file_name in files:
        if not file_name.lower().endswith(extensions_autorisees):
            continue

        image_path = os.path.join(root, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Impossible de charger l’image : {image_path}")
            continue

        print(f"🔍 Analyse de : {file_name}")
        infos = analyse_image(image)
        infos["Image"] = file_name
        analysis_results.append(infos)

# 💾 Sauvegarde dans un fichier CSV
df_analysis = pd.DataFrame(analysis_results)
df_analysis.to_csv("analyse_images_globales.csv", index=False, encoding="utf-8")
print("✅ Analyse complète enregistrée dans analyse_images_globales.csv")

"""