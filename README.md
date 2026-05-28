# OCR-powered-by-Neural-Networks

[Version française](#version-française) | [English version](#english-version)

---

## English version

OCR system built from scratch with a neural network and a Flask web interface, without TensorFlow or PyTorch. The model was trained on the EMNIST dataset at the LIRIS laboratory.

**Academic context:** BUT Informatique Special Year Program — SAE S2 2026

**Repository:** [github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks](https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks)

### How it works

1. The uploaded image is preprocessed: grayscale conversion, resizing to 32x32, normalization, Otsu binarization.
2. Characters are segmented using horizontal and vertical projections.
3. Each character is fed into a MLP (1024 → 128 → 64 → 62 classes) trained with backpropagation and mini-batch SGD.

Supports two modes: single isolated character, or full text with automatic segmentation. Accuracy above 85% on the test set (62 alphanumeric classes: 0-9, A-Z, a-z).

### Quick start

```bash
pip install flask pillow numpy
python backend/app.py
```

Open `http://localhost:5000`.

### Limitations

- Touching characters are not separated
- Low contrast or noisy images degrade accuracy
- Limited to Latin alphanumeric characters

### Contributors

- **Mehdi KHATTAB** — Image preprocessing, Flask integration, web interface
- **Jounaïd MAZNI** — Neural network architecture, training, EMNIST dataset

---

## Version française

Système OCR développé from scratch avec un réseau de neurones et une interface web Flask, sans TensorFlow ni PyTorch. Le modèle a été entraîné sur le dataset EMNIST au laboratoire LIRIS.

**Contexte académique :** BUT Informatique Année Spéciale — SAE S2 2026

**Dépôt :** [github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks](https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks)

### Fonctionnement

1. L'image est prétraitée : conversion en niveaux de gris, redimensionnement en 32x32, normalisation, binarisation par méthode d'Otsu.
2. Les caractères sont segmentés par projections horizontales et verticales.
3. Chaque caractère est passé dans un MLP (1024 → 128 → 64 → 62 classes) entraîné par rétropropagation et SGD par mini-batchs.

Deux modes disponibles : caractère isolé, ou texte complet avec segmentation automatique. Précision supérieure à 85% sur le jeu de test (62 classes alphanumériques : 0-9, A-Z, a-z).

### Démarrage rapide

```bash
pip install flask pillow numpy
python backend/app.py
```

Ouvrir `http://localhost:5000`.

### Limitations

- Les caractères collés ne sont pas séparés
- Faible contraste ou bruit dégradent la précision
- Limité aux caractères alphanumériques latins

### Contributeurs

- **Mehdi KHATTAB** — Prétraitement d'images, intégration Flask, interface web
- **Jounaïd MAZNI** — Architecture réseau de neurones, entraînement, dataset EMNIST
