import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Charger les données
df = pd.read_csv('results/resultats_fusion_heuristique_ml.csv')

# Extraire les valeurs
y_true = df['Vérité terrain']
y_pred = df['Prédiction finale']

# Déterminer les bornes de la matrice de confusion
min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))
labels = np.arange(min_val, max_val + 1)

# Calculer la matrice de confusion
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Afficher la matrice de confusion avec les axes inversés
plt.figure(figsize=(10, 8))
sns.heatmap(cm.T, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Nombre d\'images'})
plt.xlabel('Vérité terrain')
plt.ylabel('Prédiction finale')
plt.title('Matrice de confusion: Marches prédites vs Réelles')
plt.tight_layout()

# Sauvegarder et afficher
plt.savefig('metrics/results/matrice_confusion_marches.png', dpi=300)
plt.show()

# Calculer et afficher les métriques
accuracy = np.trace(cm) / np.sum(cm)
print(f"\nPrécision globale: {accuracy:.2%}")

# Afficher les erreurs fréquentes
errors = df[df['Écart absolu'] > 0]
if not errors.empty:
    print("\nErreurs fréquentes:")
    print(errors.groupby(['Vérité terrain', 'Prédiction finale']).size().sort_values(ascending=False).head(10))
