# Projet_analyse_image
Projet d'analyse d'image pour la détection et compte le nombre de marches d'escalier
## ETAPE Prétraitement:
### 1. **Conversion en niveaux de gris**
il faut convertir les images couleur (RGB) en niveaux de gris avec OpenCV
### 2. **Réduction du bruit** 
filtre comme le Gaussian Blur pour atténuer le bruit tout en conservant les contours importants.
### 3. **Amélioration du contraste**
égalisation d'histogramme pour améliorer le contraste global de l'image.
### 4.  **Binarisation** 
Transformer l'image en noir et blanc (pixels 0 ou 255) pour isoler les régions d'intérêt : seuil global
### 5.  **Sauvegarder**
pour Vérifier visuellement la qualité du prétraitement
