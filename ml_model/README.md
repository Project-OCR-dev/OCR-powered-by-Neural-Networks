# CNN OCR

## Prérequis

- Python 3.9+ — télécharger sur [python.org](https://www.python.org/downloads/)

---

## Installation

### Windows

```powershell
# Depuis le dossier ml_model
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
# Depuis le dossier ml_model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Lancer l'entraînement

```bash
python main.py        # Windows
python3 main.py       # Linux / macOS
```

---

## Structure des fichiers

```
ml_model/
├── main.py          # Boucle d'entraînement
├── forward.py       # Passe avant
├── backward.py      # Rétropropagation + mise à jour des poids
├── convolution.py   # Convolution, MaxPooling, initialisation des poids
├── utils.py         # relu, softmax, cross_entropy, one_hot, chargement dataset
├── test.py          # Inférence sur une image
└── requirements.txt
```
