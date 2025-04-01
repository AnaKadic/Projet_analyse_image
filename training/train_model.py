import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Nom image", "Erreur"])
    y = df["Erreur"]
    return X, y, df  # On retourne aussi le DataFrame complet pour retrouver les noms


def evaluate_model(model, X, y, label=""):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{label} MAE : {mae:.2f}  |  R² : {r2:.2f}")
    return mae, r2


def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mae_scores = -scores
    return mae_scores.mean(), mae_scores.std()


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModèle sauvegardé dans : {path}")


def plot_model_mae_summary(csv_path, save_path):
    df = pd.read_csv(csv_path)
    summary = df.groupby("model")["mean_mae"].mean().sort_values()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(summary.index, summary.values, color="skyblue")
    plt.title("MAE moyenne par modèle (validation croisée)")
    plt.ylabel("MAE")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height:.2f}",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f" Barplot sauvegardé dans : {save_path}")
    plt.close()


def grid_search(X, y, save_path):
    print(" Lancement du Grid Search...\n")
    results = []
    best_mae = float("inf")
    best_model = None
    best_params = {}

    for model_name, model_class in [("RandomForest", RandomForestRegressor), ("GradientBoosting", GradientBoostingRegressor)]:
        for max_depth in [8, 10, 12]:
            for min_samples_leaf in [1, 2, 4]:
                for max_features in ["sqrt", "log2"] if model_name == "RandomForest" else [None]:
                    kwargs = {
                        "n_estimators": 100,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "random_state": 42
                    }
                    if model_name == "RandomForest":
                        kwargs["max_features"] = max_features

                    model = model_class(**kwargs)
                    mean_mae, std_mae = cross_validate_model(model, X, y)
                    config = {
                        "model": model_name,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "max_features": max_features,
                        "mean_mae": round(mean_mae, 4),
                        "std_mae": round(std_mae, 4)
                    }
                    results.append(config)

                    print(f" {model_name} | max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, max_features={max_features}")
                    print(f"   ➤ MAE moy. : {mean_mae:.2f} | ± {std_mae:.2f}\n")

                    if mean_mae < best_mae:
                        best_mae = mean_mae
                        best_model = model
                        best_params = config

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    print(f" Résultats complets sauvegardés dans : {save_path}\n")

    print(" Meilleure configuration :")
    print(best_params)
    print(f"   ➤ MAE moy. : {best_mae:.2f}\n")

    best_model.fit(X, y)
    return best_model


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "features", "features_dataset.csv")
    model_path = os.path.join(root, "models", "modele_correction.pkl")
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, "grid_search_results.csv")
    barplot_path = os.path.join(results_dir, "comparaison_mae_models.png")

    X, y, full_df = load_data(data_path)

    # Triple split : 70% train / 15% val / 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=42)  # ≈ 15%

    #  Sauvegarder les splits dans /results/splits/
    split_dir = os.path.join(results_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    def save_split(X_part, y_part, original_df, filename):
        df_split = X_part.copy()
        df_split["Erreur"] = y_part
        df_split["Nom image"] = original_df.loc[X_part.index, "Nom image"].values
        df_split.to_csv(os.path.join(split_dir, filename), index=False)

    save_split(X_train, y_train, full_df, "train.csv")
    save_split(X_val, y_val, full_df, "val.csv")
    save_split(X_test, y_test, full_df, "test.csv")

    print(f" Fichiers de split sauvegardés dans : {split_dir}\n")

    #  Grid search + entraînement
    best_model = grid_search(X_train, y_train, results_csv)

    print("\n Évaluation sur validation :")
    evaluate_model(best_model, X_val, y_val, label="Validation")

    print("\n Évaluation finale sur test :")
    evaluate_model(best_model, X_test, y_test, label="Test")

    save_model(best_model, model_path)
    plot_model_mae_summary(results_csv, barplot_path)


if __name__ == "__main__":
    main()
