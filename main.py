import cv2
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from heuristics.thresholding import apply_threshold
from heuristics.canny import apply_canny
from heuristics.hough import detect_all_lines
from heuristics.count_stairs import count_stairs
from features.extract_features_image import extract_features_image

# === CHEMINS ===
base_folder = "data/images"
test_csv_path = "results/splits/test.csv"
annotations_path = "data/annotations/data annotations - Feuille 1.csv"
model_path = "models/modele_correction.pkl"
output_dir = "results"
visu_dir = os.path.join(output_dir, "analyses_visuelles")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(visu_dir, exist_ok=True)

# === CHARGEMENT DES DONNÉES ===
print("==== Chargement du modèle et des données ====")
test_df = pd.read_csv(test_csv_path)
annotations_df = pd.read_csv(annotations_path)
annotations_df.columns = ["Nom image", "Nombre de marches", "Identifiant équipe"]
merged_df = test_df.merge(annotations_df[["Nom image", "Nombre de marches"]], on="Nom image", how="left")
print(f"----|| Test set chargé : {test_csv_path}")
correction_model = joblib.load(model_path)
print(f"----|| Modèle chargé depuis : {model_path}")

# === VARIABLES RÉSULTAT ===
image_names = []
true_counts = []
predicted_counts = []
differences = []
ignored_images = []

# === OUTILS ===
def find_image_path(filename, root_dir):
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def save_visualization(original, thresholded, edges, lines, file_name, true, prediction, save_path):
    hough_image = np.zeros((*edges.shape, 3), dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Analyse complète", fontsize=16)

    axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Originale")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(thresholded, cmap="gray")
    axs[0, 1].set_title("Prétraitement")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(edges, cmap="gray")
    axs[1, 0].set_title("Canny")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(hough_image)
    axs[1, 1].set_title(f"Lignes Hough ({len(lines) if lines is not None else 0})")
    axs[1, 1].axis("off")

    plt.figtext(0.5, 0.02, f"{file_name} — Vérité : {true} | Prédiction finale : {prediction}",
                ha="center", fontsize=12, color="red")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# === TRAITEMENT DES IMAGES ===
print("\n==== Début du traitement des images ====")

for idx, row in merged_df.iterrows():
    file_name = row["Nom image"]
    true = row["Nombre de marches"]

    if pd.isna(true):
        ignored_images.append(file_name)
        continue

    image_path = find_image_path(file_name, base_folder)
    if image_path is None or not os.path.exists(image_path):
        ignored_images.append(file_name)
        continue

    image = cv2.imread(image_path)
    if image is None:
        ignored_images.append(file_name)
        continue

    print("\n------------------------------------------------------------")
    print(f"|| Image : {file_name}")
    print("------------------------------------------------------------")

    thresholded = apply_threshold(image)
    edges = apply_canny(thresholded)
    _, lines = detect_all_lines(edges, min_length=80)
    heuristique_pred = count_stairs(image, lines, y_threshold=42, min_length=120, min_y_gap=15)

    # Lecture depuis fichier CSV fusionné
    feature_vector = [
        row["Luminosité"],
        row["Contraste"],
        row["Netteté"],
        row["% lignes verticales"],
        row["% lignes horizontales"],
        row["Nb lignes détectées"],
        row["Prédiction heuristique"] if "Prédiction heuristique" in row else heuristique_pred
    ]

    features_names = [
        "Luminosité", "Contraste", "Netteté",
        "% lignes verticales", "% lignes horizontales",
        "Nb lignes détectées", "Prédiction heuristique"
    ]

    X = pd.DataFrame([feature_vector], columns=features_names)
    correction = correction_model.predict(X)[0]
    final_prediction = int(round(heuristique_pred + correction))

    print(f"----|| Prédiction heuristique : {heuristique_pred}")
    print(f"----|| Correction ML : {correction:.2f}")
    print(f"----|| ➤ Prédiction finale : {final_prediction}")
    print(f"----|| Vérité terrain : {int(true)}")

    if heuristique_pred == int(true):
        print("----|| Résultat exact (heuristique)")
    elif abs(heuristique_pred - int(true)) < abs(final_prediction - int(true)):
        print("----|| ML a dégradé le résultat")
    elif abs(heuristique_pred - int(true)) > abs(final_prediction - int(true)):
        print("----|| ML a amélioré la prédiction")
    else:
        print("----|| Même écart heuristique / ML")

    image_names.append(file_name)
    true_counts.append(int(true))
    predicted_counts.append(final_prediction)
    differences.append(abs(int(true) - final_prediction))

    if len(image_names) <= 15:
        save_path = os.path.join(visu_dir, f"{os.path.splitext(file_name)[0]}_analyse.png")
        save_visualization(image, thresholded, edges, lines, file_name, int(true), final_prediction, save_path)

# === RÉSUMÉ GLOBAL ===
print("\n==== Résultats globaux ====")
if true_counts:
    mae = mean_absolute_error(true_counts, predicted_counts)
    print(f"----|| MAE global (test) : {mae:.2f} marches")

    df = pd.DataFrame({
        "Image": image_names,
        "Prédiction finale": predicted_counts,
        "Vérité terrain": true_counts,
        "Écart absolu": differences
    })
    output_csv = os.path.join(output_dir, "resultats_test_seulement.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"----|| Résultats enregistrés dans : {output_csv}")
else:
    print("----|| Aucune image traitée correctement.")

if ignored_images:
    ignored_path = os.path.join(output_dir, "images_non_lues.txt")
    with open(ignored_path, "w") as f:
        for img in ignored_images:
            f.write(img + "\n")
    print(f"----|| {len(ignored_images)} image(s) ignorée(s) enregistrée(s) dans : {ignored_path}")
