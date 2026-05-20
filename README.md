# OCR-powered-by-Neural-Networks

[🇫🇷 Version française](#version-française) | [🇬🇧 English version](#english-version)

---

## English Version

### 🔍 Overview

Complete OCR (Optical Character Recognition) system built entirely from scratch using neural networks and a Flask web interface. This project demonstrates a pedagogical approach to machine learning by implementing core algorithms without relying on high-level frameworks like TensorFlow or PyTorch.

**Academic Context:** BUT Informatique Special Year Program (SAÉ S2 - 2026)  
**Repository:** [GitHub Project](https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks)

---

### ✨ Features

#### 🎯 Core Functionality

- **Dual Recognition Modes:**
  - Single isolated character recognition
  - Multi-character text recognition with automatic segmentation
- **From-Scratch Implementation:**
  - Image preprocessing pipeline (grayscale conversion, resizing, normalization, binarization)
  - Character segmentation using projection-based algorithms
  - Neural network architecture (MLP) implemented in pure NumPy
- **Web Interface:**
  - Intuitive Flask-based interface
  - Real-time image upload and processing
  - Clear result visualization with mode indication

#### 🔧 Technical Components

**Image Preprocessing:**

- RGB → Grayscale conversion (manual weighted averaging)
- Pillow Image → NumPy matrix conversion
- Nearest Neighbor resizing to 32×32
- Pixel normalization (0-255 → 0-1)
- Otsu's method for adaptive thresholding
- Horizontal/vertical projection for character detection

**Neural Network:**

- Multi-layer Perceptron (MLP) architecture
- Input: 1024 neurons (32×32 normalized pixels)
- Hidden layers: 128 and 64 neurons with ReLU activation
- Output: 62 classes (0-9, A-Z, a-z) with Softmax
- Training: Backpropagation with mini-batch SGD on EMNIST byclass dataset
- Accuracy: >85% on test set

**Web Application:**

- Secure file upload with validation
- Mode selection (isolated character vs. full text)
- Responsive design
- Error handling and user feedback

---

### 🚀 Quick Start

#### Prerequisites

```bash
Python 3.8+
pip install flask pillow numpy
```

#### Installation

```bash
git clone https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks.git
cd OCR-powered-by-Neural-Networks
```

#### Running the Application

```bash
python backend/app.py
```

Open your browser at `http://localhost:5000`

---

### 📁 Project Structure

OCR-powered-by-Neural-Networks/
├── backend/ # Flask web application
│ ├── static/ # CSS, JS, uploaded images
│ ├── templates/ # HTML templates
│ └── app.py # Flask entry point
├── ml_model/ # Neural network implementation
│ ├── convolution.py # Convolution operations
│ ├── forward.py # Forward pass
│ ├── predict.py # Prediction interface
│ └── params_epoch_30.pkl # Trained model weights
├── preprocessing/ # Image processing pipeline
│ ├── images/ # Test images
│ └── image_processing.py # Processing functions
└── docs/ # Documentation

---

### 🎓 Educational Approach

This project follows a **"from scratch"** philosophy to:

- **Understand fundamentals:** Grasp what happens at each pixel-level operation
- **Master data structures:** Work with matrices, arrays, and tensors manually
- **Learn ML internals:** Implement backpropagation and gradient descent without black-box libraries
- **Debug deeply:** Confront and solve coordinate convention issues (NumPy vs. Pillow)

**Result:** Deep understanding of OCR and neural network mechanics beyond library usage.

---

### ⚠️ Known Limitations

- **Segmentation:** Works best with well-spaced characters; touching characters are not separated
- **Image quality:** Low contrast, noise, or visual effects (shadows, colors, relief) degrade accuracy
- **Character set:** Limited to alphanumeric Latin characters (0-9, A-Z, a-z)

---

### 👥 Contributors

- **Mehdi KHATTAB** - Image preprocessing pipeline, Flask integration, web interface, deployment
- **Jounaïd MAZNI** - Neural network architecture, training, EMNIST dataset management

---

### 📄 License

Academic project - BUT Informatique (2026)

---

## Version Française

### 🔍 Aperçu

Système complet de reconnaissance optique de caractères (OCR) développé entièrement from scratch avec réseau de neurones et interface web Flask. Ce projet illustre une approche pédagogique du machine learning en implémentant les algorithmes fondamentaux sans recourir aux frameworks haut niveau comme TensorFlow ou PyTorch.

**Contexte académique :** BUT Informatique Année Spéciale (SAÉ S2 - 2026)  
**Dépôt :** [Projet GitHub](https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks)

---

### ✨ Fonctionnalités

#### 🎯 Fonctionnalités principales

- **Double mode de reconnaissance :**
  - Reconnaissance de caractère isolé
  - Reconnaissance de texte complet avec segmentation automatique
- **Implémentation from scratch :**
  - Pipeline de prétraitement d'images (conversion grayscale, redimensionnement, normalisation, binarisation)
  - Segmentation de caractères par algorithmes de projection
  - Architecture de réseau de neurones (MLP) implémentée en NumPy pur
- **Interface web :**
  - Interface Flask intuitive
  - Upload et traitement d'images en temps réel
  - Visualisation claire des résultats avec indication du mode

#### 🔧 Composants techniques

**Prétraitement d'images :**

- Conversion RGB → Grayscale (moyenne pondérée manuelle)
- Conversion Image Pillow → matrice NumPy
- Redimensionnement Nearest Neighbor vers 32×32
- Normalisation des pixels (0-255 → 0-1)
- Binarisation par méthode d'Otsu (seuillage adaptatif)
- Projection horizontale/verticale pour détection de caractères

**Réseau de neurones :**

- Architecture Perceptron multicouche (MLP)
- Entrée : 1024 neurones (pixels 32×32 normalisés)
- Couches cachées : 128 et 64 neurones avec activation ReLU
- Sortie : 62 classes (0-9, A-Z, a-z) avec Softmax
- Entraînement : Rétropropagation avec SGD par mini-batchs sur EMNIST byclass
- Précision : >85% sur jeu de test

**Application web :**

- Upload sécurisé avec validation de fichiers
- Sélection du mode (caractère isolé vs. texte complet)
- Design responsive
- Gestion d'erreurs et feedback utilisateur

---

### 🚀 Démarrage rapide

#### Prérequis

```bash
Python 3.8+
pip install flask pillow numpy
```

#### Installation

```bash
git clone https://github.com/Project-OCR-dev/OCR-powered-by-Neural-Networks.git
cd OCR-powered-by-Neural-Networks
```

#### Lancement de l'application

```bash
python backend/app.py
```

Ouvrir le navigateur à `http://localhost:5000`

---

### 📁 Structure du projet

OCR-powered-by-Neural-Networks/
├── backend/ # Application web Flask
│ ├── static/ # CSS, JS, images uploadées
│ ├── templates/ # Templates HTML
│ └── app.py # Point d'entrée Flask
├── ml_model/ # Implémentation réseau de neurones
│ ├── convolution.py # Opérations de convolution
│ ├── forward.py # Passe avant
│ ├── predict.py # Interface de prédiction
│ └── params_epoch_30.pkl # Poids du modèle entraîné
├── preprocessing/ # Pipeline de traitement d'image
│ ├── images/ # Images de test
│ └── image_processing.py # Fonctions de traitement
└── docs/ # Documentation

---

### 🎓 Approche pédagogique

Ce projet suit une philosophie **"from scratch"** pour :

- **Comprendre les fondamentaux :** Saisir ce qui se passe à chaque opération au niveau pixel
- **Maîtriser les structures de données :** Manipuler matrices, tableaux et tenseurs manuellement
- **Apprendre l'interne du ML :** Implémenter rétropropagation et descente de gradient sans boîtes noires
- **Débugger en profondeur :** Affronter et résoudre les problèmes de conventions de coordonnées (NumPy vs. Pillow)

**Résultat :** Compréhension approfondie de l'OCR et des mécanismes des réseaux de neurones au-delà de l'usage de bibliothèques.

---

### ⚠️ Limitations connues

- **Segmentation :** Fonctionne mieux avec caractères bien espacés ; les caractères collés ne sont pas séparés
- **Qualité d'image :** Faible contraste, bruit ou effets visuels (ombres, couleurs, relief) dégradent la précision
- **Jeu de caractères :** Limité aux caractères alphanumériques latins (0-9, A-Z, a-z)

---

### 👥 Contributeurs

- **Mehdi KHATTAB** - Pipeline de prétraitement d'images, intégration Flask, interface web, déploiement
- **Jounaïd MAZNI** - Architecture du réseau de neurones, entraînement, gestion du dataset EMNIST

---

### 📄 Licence

Projet académique - BUT Informatique (2026)
