# Projet_analyse_image

## Auteurs

Ce projet a été réalisé par :

- Anaïs Kadic  
- Titouan Brierre  
- Wided Dhfallah

Objectif -> Créer un système capable de détecter et compter le nombre de marches sur une photo d’escalier, à partir d’un smartphone, en combinant heuristique et Machine Learning.


##  Sommaire

1. [Structure du code](#structure-du-projet)
2. [Dépendances](#dépendances)
3. [Exécution](#installation--exécution)
4. [Fonctionnement du système](#fonctionnement-du-système)
    - [Prétraitement](#prétraitement)
    - [Analyse des lignes](#analyse-des-lignes)
    - [Extraction de features](#extraction-de-features)
    - [Machine Learning](#modèle-machine-learning)
5. [Résultats](#fonctionnement-du-système)
---

### 1. Structure du code
Projet_analyse_image/ <br> 
├── main.py      <br>                  
├── README.md     <br>               
├── data/<br>
│   ├── images/                   
│   └── annotations/       <br>        
│       └── data annotations.csv  <br>
├── features/<br>
│   ├── extract_features_image.py  # Code pour extraire les descripteurs visuels<br>
│   ├── build_dataset.py <br>
│   └── features_dataset.csv      <br>
├── heuristics/<br>
│   ├── thresholding.py            <br>
│   ├── canny.py                <br>  
│   ├── hough.py                <br>   
│   └── count_stairs.py            
├── metrics/<br>
│   ├── confusionMatrice.py 
│   └── graph.py
├── models/           
│   └── modele_correction.pkl  # Modèle entraîné sauvegardé <br>
├── training/ <br>
│   └── train_model.py    <br>
├── results/<br>
│   ├── resultats_test_seulement.csv <br>
│   ├── grid_search_results.csv      <br>
│   ├── analyses_visuelles/       <br>   
│   └── splits/               <br>     
│       ├── train.csv<br> 
│       ├── val.csv<br>
│       └── test.csv<br>

---
## 2. Dépendances

Ce projet utilise Python 3.8+ avec les bibliothèques suivantes :

- `opencv-python`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `joblib`


---
## 3. Exécution
1. Génération du dataset (optionnel)

Uniquement si vous souhaitez reconstruire les descripteurs visuels à partir des images :

python features/build_dataset.py

2. Entraînement d’un nouveau modèle (optionnel)

Si vous souhaitez réentraîner un modèle de correction depuis zéro :

python training/train_model.py

3. Exécution :

Tout est déjà prêt (dataset + modèle), exécutez simplement :

python main.py

---
## 4. Fonctionnement du système
Il s'agit d'une approche hybride combinant traitement d'image et correction via machine learning .


### 🔹 4.1 Prétraitement

pour améliorer la lisibilité on mes en place:

- **Analyse adaptative** : détection automatique des images trop sombres, floues ou peu contrastées.
- **Correction visuelle** : ajustement de la **luminosité**, du **contraste**, ou application de **filtres** (Gaussian, Bilateral).
- **Seuillage combiné** : méthode d’Otsu + seuillage adaptatif.
- **Détection de contours** : via **Canny** et **Sobel** pour faire ressortir les structures linéaires.

---

### 🔹 4.2 Analyse des lignes

Une fois l'image prétraitée, on applique :

- **Transformée de Hough** pour détecter les **lignes** horizontales et verticales.
- **Filtrage géométrique** : suppression des lignes trop courtes ou inclinées.
- **Fusion de lignes proches** : regroupement des lignes similaires pour éviter les doublons.
- **Comptage heuristique** des marches par regroupement des coordonnées verticales.

---

### 🔹 4.3 Extraction de features

Pour chaque image, des **descripteurs visuels** sont extraits afin d'entraîner un modèle de correction :

- **Luminosité moyenne**
- **Contraste global**
- **Netteté (Laplacien)**
- **% de lignes verticales et horizontales**
- **Nombre total de lignes détectées**
- **Estimation heuristique du nombre de marches**

Ces valeurs sont ensuite stockées dans un fichier `.csv`.

---

### 🔹 4.4 Modèle Machine Learning

Un modèle de **Random Forest Regressor** est entraîné pour **prédire l'erreur** commise par l'approche heuristique.

- **Entraînement** via `scikit-learn`
- **Validation croisée (5-fold)** pour évaluer la robustesse
- **Recherche de paramètres (Grid Search)** sur :
  - `max_depth`
  - `min_samples_leaf`
  - `max_features`
- **Comparaison de plusieurs modèles** : Random Forest vs. Gradient Boosting
- **Évaluation finale** via :
  - **MAE** (Mean Absolute Error)
  - **R²** (coefficient de détermination)

Le modèle choisi est ensuite utilisé pour **corriger** les prédictions heuristiques.

---
### 5. Resultats


Le système a été évalué sur un ensemble d’images de test représentant différentes conditions (lumière, angle, flou...).

#### Résumé :

- **MAE** : ~ **4 marches**
- **R² (Score de détermination)** : valeur comprise entre **0.75 et 0.85** selon les splits
- **Testé sur l'ensemble de test**
- **Gain moyen par rapport à l’heuristique seule** : entre **15 % et 25 %** d’erreur corrigée

#### Prédiction :

| Image               | Vérité terrain | Heuristique | ML Correction | Final |
|---------------------|----------------|-------------|---------------|--------|
| Groupe1_Image9.jpg  | 9              | 6           | +4.28         | **10** |
| Groupe5_image10.jpg | 14              | 9          | +1.20        | **10**  |

#### Visualisation :

On peut visulise dans dans `results/analyses_visuelles/` pour chaque image :
- Image originale
- Prétraitement (binarisation)
- Détection de contours (Canny)
- Lignes détectées (Hough)



---
