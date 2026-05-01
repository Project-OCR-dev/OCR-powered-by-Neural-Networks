<<<<<<< HEAD

# OCR-powered-by-Neural-Networks

[🇫🇷 Version française](#version-française) | [🇬🇧 English version](#english-version)

---

## English Version

Character recognition system (OCR) developed from scratch with neural networks and Flask web application.

### Description

Complete OCR system development project combining:

- **Frontend**: Flask web application with responsive interface
- **Backend**: Image preprocessing pipeline from scratch
- **Machine Learning**: Neural network for character recognition (EMNIST dataset)

Developed as part of the BUT Informatique Special Year program (SAÉ S2 - 2026).

### Features

#### Currently Implemented

- Secure image upload (file type validation)
- RGB → Grayscale conversion from scratch
- Image → NumPy Matrix conversion from scratch
- Data normalization (0-255 → 0-1)
- Responsive web interface with user feedback (Flash messages)
- 3-page workflow: Upload → Processing → Results
- Error handling and validation

#### Under Development

- Image resizing (28×28)
- Region of interest cropping
- Neural network training
- ML model integration in Flask

#### Next Steps

- Complete preprocessing pipeline
- Real-time prediction
- Results visualization
- Deployment

### Technologies Used

**Backend:**

- Python 3.x
- Flask (web server)
- NumPy (matrix computations)
- Pillow (image manipulation)

**Frontend:**

- HTML5 / CSS3
- JavaScript
- Responsive design

**Machine Learning:**

- EMNIST dataset (handwritten characters)
- Neural network from scratch

### User Workflow

1. **Upload**: Upload an image (PNG, JPG, JPEG, GIF)
2. **Processing**: Image is automatically processed
3. **Results**: Prediction result display

### Educational Approach

This project adopts a **"from scratch"** approach to:

- Understand image processing algorithms
- Master data structures (matrices, arrays)
- Learn ML preprocessing fundamentals

**Goal**: Develop deep understanding rather than solely using high-level libraries.

### Contributors

- **Mehdi KHATTAB** - Web development (Flask, Frontend) and preprocessing
- **Jounaïd MAZNI** - Machine Learning (Neural network, Training)

**Status:** Under active development

---

## Version Française

Système de reconnaissance de caractères (OCR) développé from scratch avec réseau de neurones et application web Flask.

### Description

Projet de développement d'un système OCR complet combinant :

- **Frontend** : Application web Flask avec interface responsive
- **Backend** : Pipeline de preprocessing d'images from scratch
- **Machine Learning** : Réseau de neurones pour la reconnaissance de caractères (EMNIST dataset)

Développé dans le cadre du BUT Informatique Année Spéciale (SAÉ S2 - 2026).

### Fonctionnalités

#### Actuellement implémenté

- Upload d'images sécurisé (validation des types)
- Conversion RGB → Grayscale from scratch
- Conversion Image → Matrice NumPy from scratch
- Normalisation des données (0-255 → 0-1)
- Interface web responsive avec feedback utilisateur (Flash messages)
- Workflow 3 pages : Upload → Processing → Results
- Gestion d'erreurs et validation

#### En cours de développement

- Redimensionnement d'images (28×28)
- Découpage de zones d'intérêt
- Entraînement du réseau de neurones
- Intégration modèle ML dans Flask

#### Prochaines étapes

- Pipeline complet de preprocessing
- Prédiction en temps réel
- Visualisation des résultats
- Déploiement

### Technologies utilisées

**Backend :**

- Python 3.x
- Flask (serveur web)
- NumPy (calculs matriciels)
- Pillow (manipulation d'images)

**Frontend :**

- HTML5 / CSS3
- JavaScript
- Design responsive

**Machine Learning :**

- Dataset EMNIST (caractères manuscrits)
- Réseau de neurones from scratch

### Workflow utilisateur

1. **Upload** : Télécharger une image (PNG, JPG, JPEG, GIF)
2. **Processing** : L'image est automatiquement traitée
3. **Results** : Affichage du résultat de la prédiction

### Approche pédagogique

Ce projet adopte une approche **"from scratch"** pour :

- Comprendre les algorithmes de traitement d'images
- Maîtriser les structures de données (matrices, arrays)
- Apprendre les fondamentaux du preprocessing ML

**Objectif** : Développer une compréhension approfondie plutôt que d'utiliser uniquement des bibliothèques haut niveau.

### Contributeurs

- **Mehdi KHATTAB** - Développement web (Flask, Frontend) et preprocessing
- **Jounaïd MAZNI** - Machine Learning (Réseau de neurones, Entraînement)

**Statut :** En développement actif

> > > > > > > 2d26eec2590a3031c0b78cf779126bbec0a697d2
