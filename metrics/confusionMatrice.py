import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_path: str) -> pd.DataFrame:
    """Charge les résultats du fichier CSV."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Fichier non trouvé : {results_path}")
    return pd.read_csv(results_path)

def compute_absolute_error(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne Écart absolu au DataFrame."""
    df["Écart absolu"] = abs(df["Vérité terrain"] - df["Prédiction finale"])
    return df

def plot_error_distribution(df: pd.DataFrame, save_path: str):
    """Trace l’histogramme de la distribution des erreurs absolues."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Écart absolu"], bins=range(0, df["Écart absolu"].max() + 2), kde=False, color='salmon')
    plt.title("Distribution des erreurs absolues (|prédiction - vérité|)")
    plt.xlabel("Erreur absolue (en marches)")
    plt.ylabel("Nombre d'images")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("📊 Distribution des erreurs enregistrée.")

def plot_tolerance_rates(df: pd.DataFrame, save_path: str, tolerances=(0, 1, 2, 3)):
    """Calcule et trace le taux de prédiction correcte à ±k marches."""
    taux_data = []
    print("\n📊 Taux de prédiction correcte selon tolérance :")
    for tol in tolerances:
        rate = (df["Écart absolu"] <= tol).mean()
        taux_data.append((f"±{tol}", rate))
        print(f"  ➤ ±{tol} marches : {rate:.2%}")

    labels, values = zip(*taux_data)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color='mediumseagreen')
    plt.ylim(0, 1)
    plt.ylabel("Taux de bonne prédiction")
    plt.title("Prédictions correctes dans une tolérance de ± marches")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f"{height:.2%}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("📈 Taux de prédiction par tolérance enregistrés.")

def show_top_errors(df: pd.DataFrame, n: int = 10):
    """Affiche les erreurs les plus fréquentes."""
    print("\n📉 Top erreurs les plus fréquentes (Vérité ➞ Prédiction) :")
    top_errors = df[df["Écart absolu"] > 0].groupby(
        ["Vérité terrain", "Prédiction finale"]
    ).size().sort_values(ascending=False).head(n)
    print(top_errors)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_path = os.path.join(root_dir, "results", "resultats_test_seulement.csv")
    metrics_dir = os.path.join(root_dir, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    df = load_results(results_path)
    df = compute_absolute_error(df)

    plot_error_distribution(df, os.path.join(metrics_dir, "distribution_erreurs_absolues.png"))
    plot_tolerance_rates(df, os.path.join(metrics_dir, "barplot_taux_tolerance.png"))
    show_top_errors(df)

if __name__ == "__main__":
    main()
