import os
import cv2
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error

# 🔌 Modules perso
from heuristics.thresholding import apply_threshold
from heuristics.canny import apply_canny
from heuristics.hough import detect_all_lines
from heuristics.count_stairs import count_stairs
from features.analyse_image import analyse_image

# 📁 Dossiers (chemins relatifs)
base_folder = os.path.join("data", "images")
annotations_path = os.path.join("data", "annotations", "data annotations - Feuille 1.csv")
model_path = os.path.join("models", "modele_correction_erreur.pkl")
results_path = os.path.join("results", "resultats_fusion_heuristique_ml.csv")

# 📄 Charger annotations
annotations_df = pd.read_csv(annotations_path, delimiter=",", encoding="utf-8")
annotations_df.columns = ["Nom image", "Nombre de marches", "Identifiant équipe"]

# 🔍 Extensions autorisées
extensions_autorisees = (".jpg", ".jpeg", ".png")

# 🔄 Charger le modèle de correction
correction_model = joblib.load(model_path)
print("✅ Modèle de correction chargé")

# 📊 Résultats
image_names = []
true_counts = []
predicted_counts = []
differences = []

# 🔁 Parcours
for root, _, files in os.walk(base_folder):
    for file_name in files:
        if not file_name.lower().endswith(extensions_autorisees):
            continue

        image_path = os.path.join(root, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Impossible de lire : {file_name}")
            continue

        print(f"\n📸 Traitement de : {file_name}")

        # 🔍 Vérité terrain
        match = annotations_df[annotations_df["Nom image"].str.lower() == file_name.lower()]
        if match.empty:
            print(f"⚠️ Pas d'annotation pour l'image : {file_name}")
            continue
        true = int(match["Nombre de marches"].values[0])

        # 🔬 Étape 1 – Pipeline heuristique
        thresholded = apply_threshold(image)
        edges = apply_canny(thresholded)
        _, lines = detect_all_lines(edges, min_length=80)
        heuristique_pred = count_stairs(image, lines, y_threshold=42, min_length=120, min_y_gap=15)

        # 🔬 Étape 2 – Extraire features
        features = analyse_image(image)
        feature_vector = np.array([
            features.get("Luminosité moyenne", 0),
            features.get("Contraste global (std)", 0),
            features.get("Niveau de netteté (Laplacian)", 0),
            features.get("% lignes verticales", 0),
            features.get("% lignes horizontales", 0),
            features.get("Nb lignes détectées", 0),
            heuristique_pred  # Ajout de la prédiction heuristique comme feature
        ]).reshape(1, -1)

        # 🔮 Étape 3 – Correction par ML
        correction = correction_model.predict(feature_vector)[0]
        final_prediction = int(round(heuristique_pred + correction))

        print(f"🔢 Heuristique : {heuristique_pred} | Correction ML : {correction:.2f} ➡️ Final : {final_prediction}")

        # 🧾 Stockage
        image_names.append(file_name)
        true_counts.append(true)
        predicted_counts.append(final_prediction)
        differences.append(abs(true - final_prediction))

# 📏 MAE
mae = mean_absolute_error(true_counts, predicted_counts)
print(f"\n📏 MAE (fusion heuristique + ML) sur {len(predicted_counts)} images : {mae:.2f} marches")

# 💾 Sauvegarde CSV
df = pd.DataFrame({
    "Image": image_names,
    "Prédiction finale": predicted_counts,
    "Vérité terrain": true_counts,
    "Écart absolu": differences
})

# ✅ Créer le dossier 'results' s'il n'existe pas
os.makedirs("results", exist_ok=True)

# 🔽 Chemin du fichier à sauvegarder
output_path = os.path.join("results", "resultats_fusion_heuristique_ml.csv")
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Résultats enregistrés dans {output_path}")

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