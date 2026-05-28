# OCR-powered-by-Neural-Networks

[Version française](#version-française) | [English version](#english-version)

---

## English version

OCR system built from scratch in NumPy, without TensorFlow or PyTorch, with a Flask web interface. The model was trained on the EMNIST byclass dataset at the LIRIS laboratory.

**Academic context:** BUT Informatique Special Year Program — SAE S2 2026

**Repository:** [github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks](https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks)

### Architecture

The neural network is a CNN implemented from scratch:

```
Input (32x32 grayscale)
  → Conv (32 filters 3x3) → ReLU → MaxPool 2x2
  → Conv (64 filters 3x3) → ReLU → MaxPool 2x2
  → Flatten (2304)
  → Dense (256) → ReLU
  → Dense (128) → ReLU
  → Dense (62) → Softmax
```

62 output classes: digits 0-9, uppercase A-Z, lowercase a-z. Trained with SGD and cross-entropy loss over 5 epochs.

### How it works

**Isolated character mode:**
1. RGB → grayscale → 32x32 resize (nearest neighbor) → normalization [0, 1]
2. Forward pass through the CNN → predicted character

**Full text mode:**
1. RGB → grayscale → Otsu binarization
2. Character segmentation via horizontal and vertical projections
3. Each detected region is cropped, resized to 32x32, normalized, then classified
4. Spaces are detected automatically based on gaps between characters

### Quick start

```bash
pip install flask pillow numpy scipy
python backend/app.py
```

Open `http://localhost:5000`.

### Limitations

- Touching characters are not separated
- Low contrast, noise, or visual effects degrade accuracy
- Limited to Latin alphanumeric characters (0-9, A-Z, a-z)

### Contributors

- **Mehdi KHATTAB** — Image preprocessing, segmentation, Flask integration, web interface
- **Jounaïd MAZNI** — CNN architecture, training, EMNIST dataset management

---

## Version française

Système OCR développé from scratch en NumPy, sans TensorFlow ni PyTorch, avec une interface web Flask. Le modèle a été entraîné sur le dataset EMNIST byclass au laboratoire LIRIS.

**Contexte académique :** BUT Informatique Année Spéciale — SAE S2 2026

**Dépôt :** [github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks](https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks)

### Architecture

Le réseau de neurones est un CNN implémenté from scratch :

```
Entrée (32x32 niveaux de gris)
  → Conv (32 filtres 3x3) → ReLU → MaxPool 2x2
  → Conv (64 filtres 3x3) → ReLU → MaxPool 2x2
  → Flatten (2304)
  → Dense (256) → ReLU
  → Dense (128) → ReLU
  → Dense (62) → Softmax
```

62 classes de sortie : chiffres 0-9, majuscules A-Z, minuscules a-z. Entraîné avec SGD et cross-entropy sur 5 époques.

### Fonctionnement

**Mode caractère isolé :**
1. RGB → niveaux de gris → redimensionnement 32x32 (nearest neighbor) → normalisation [0, 1]
2. Passage dans le CNN → caractère prédit

**Mode texte complet :**
1. RGB → niveaux de gris → binarisation par méthode d'Otsu
2. Segmentation des caractères par projections horizontales et verticales
3. Chaque zone détectée est découpée, redimensionnée en 32x32, normalisée puis classifiée
4. Les espaces sont détectés automatiquement selon les écarts entre caractères

### Démarrage rapide

```bash
pip install flask pillow numpy scipy
python backend/app.py
```

Ouvrir `http://localhost:5000`.

### Limitations

- Les caractères collés ne sont pas séparés
- Faible contraste, bruit ou effets visuels dégradent la précision
- Limité aux caractères alphanumériques latins (0-9, A-Z, a-z)

### Contributeurs

- **Mehdi KHATTAB** — Prétraitement d'images, segmentation, intégration Flask, interface web
- **Jounaïd MAZNI** — Architecture CNN, entraînement, gestion du dataset EMNIST
