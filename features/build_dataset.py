import os
import cv2
import pandas as pd
import numpy as np
import sys
from features.extract_features_image import extract_features_image
from heuristics.thresholding import apply_threshold
from heuristics.canny import apply_canny
from heuristics.hough import detect_all_lines
from heuristics.count_stairs import count_stairs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

base_folder = "/home/user/Documents/M1/s2/analyse d'image/Projet_analyse_image/data/images"
annotations_path = "/home/user/Documents/M1/s2/analyse d'image/Projet_analyse_image/data/annotations/data annotations - Feuille 1.csv"
annotations_df = pd.read_csv(annotations_path, delimiter=",", encoding="utf-8")
annotations_df.columns = ["Nom image", "Nombre de marches", "Identifiant équipe"]

extensions_autorisees = (".jpg", ".jpeg", ".png")

features_data = []

for root, _, files in os.walk(base_folder):
    for file_name in files:
        if not file_name.lower().endswith(extensions_autorisees):
            continue

        image_path = os.path.join(root, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Impossible de lire : {file_name}")
            continue

        print(f" Analyse : {file_name}")

        # Analyse des features visuelles
        infos = extract_features_image(image)

        match = annotations_df[annotations_df["Nom image"].str.lower() == file_name.lower()]
        if match.empty:
            print(f" Pas d'annotation pour {file_name}")
            continue

        nb_marches = int(match["Nombre de marches"].values[0])

        thresholded = apply_threshold(image)
        edges = apply_canny(thresholded)
        _, lines = detect_all_lines(edges, min_length=80)
        heuristique_pred = count_stairs(image, lines, y_threshold=42, min_length=120, min_y_gap=15)

        erreur = nb_marches - heuristique_pred

        row = {
            "Nom image": file_name,
            "Luminosité": infos.get("Luminosité moyenne", 0),
            "Contraste": infos.get("Contraste global (std)", 0),
            "Netteté": infos.get("Niveau de netteté (Laplacian)", 0),
            "% lignes verticales": infos.get("% lignes verticales", 0),
            "% lignes horizontales": infos.get("% lignes horizontales", 0),
            "Nb lignes détectées": infos.get("Nb lignes détectées", 0),
            "Prédiction heuristique": heuristique_pred,
            "Erreur": erreur
        }
        features_data.append(row)


df = pd.DataFrame(features_data)
df.to_csv("features_dataset.csv", index=False, encoding="utf-8")
print(" Dataset enregistré sous features_dataset.csv")