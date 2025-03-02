import pandas as pd

def evaluate_performance(csv_path, detected_results):
    """Compare les résultats de détection des marches avec la vérité terrain."""
    ground_truth = pd.read_csv(csv_path)
    results = []

    for result in detected_results:
        image_name = result['Image']
        detected_stairs = result['Marches détectées']

        true_row = ground_truth[ground_truth['Nom image'] == image_name]

        if not true_row.empty:
            true_stairs = int(true_row['Nombre de marches'].values[0])
            ecart = detected_stairs - true_stairs

            results.append({
                "Image": image_name,
                "Vérité Terrain": true_stairs,
                "Marches détectées": detected_stairs,
                "Écart": ecart
            })

    results_df = pd.DataFrame(results)

    # 🔍 Résultats et exportation
    print("\n📊 Résultats comparatifs :")
    print(results_df)

    correct_detections = sum(results_df["Écart"] == 0)
    total_images = len(results_df)
    precision = correct_detections / total_images if total_images > 0 else 0

    print(f"\n📈 Performance globale : {precision * 100:.2f}%")
    results_df.to_csv("resultats_comparatif.csv", index=False)
    print("✅ Résultats enregistrés dans 'resultats_comparatif.csv'.")

    return results_df
