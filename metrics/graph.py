import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('results/resultats_fusion_heuristique_ml.csv')

# Extraire les valeurs
y_true = df['Vérité terrain']
y_pred = df['Prédiction finale']

# Calculer la moyenne des marches détectées pour chaque nombre de marches réelles
mean_detected_steps = df.groupby('Vérité terrain')['Prédiction finale'].mean()

# Tracer le graphique
plt.figure(figsize=(10, 8))
sns.lineplot(x=mean_detected_steps.index, y=mean_detected_steps.values, marker='o', linestyle='-', color='b', label='Moyenne des marches détectées')

# Tracer la ligne de référence (ce qui devrait être correct)
plt.plot(mean_detected_steps.index, mean_detected_steps.index, linestyle='--', color='r', label='Prédiction parfaite')

plt.xlabel('Marches réelles')
plt.ylabel('Moyenne des marches détectées')
plt.title('Moyenne des marches détectées en fonction des marches réelles')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Sauvegarder et afficher le graphique
plt.savefig('metrics/results/moyenne_marches_detectees_avec_reference.png', dpi=300)
plt.show()
