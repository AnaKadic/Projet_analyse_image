import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# 📁 Calculer le chemin absolu du dossier racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 📥 Chemin du CSV de features
features_path = os.path.join(project_root, "features", "features_dataset_correction.csv")

# 📄 Charger les données
df = pd.read_csv(features_path)

# 🎯 X = features + prédiction heuristique / y = erreur
X = df.drop(columns=["Nom image", "Erreur"])  # Les features
y = df["Erreur"]  # La cible ici est l'erreur

# 🔀 Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🌳 Entraînement du modèle de correction
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔍 Prédictions de correction
y_pred = model.predict(X_test)

# 📏 Évaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 MAE sur la correction : {mae:.2f}")
print(f"🔁 R² score : {r2:.2f}")

# Création du dossier models/ s'il n'existe pas
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Sauvegarde du modèle
model_path = os.path.join(models_dir, "modele_correction_erreur.pkl")
joblib.dump(model, model_path)
print(f"✅ Modèle de correction sauvegardé dans '{model_path}'")
