from preprocessing.image_processing import *
from PIL import Image

# Test segmentation
print("=== Test Segmentation ===")
lettres = testerSegmentation('images/test_ocr.png')

# Test lettre isolée
print("\n=== Test Lettre Isolée ===")
img = Image.open('images/lettre_A.png')
resultat = ocrLettreIsolee(img)
print(f"Lettre détectée : {resultat}")

# Test texte complet
print("\n=== Test Texte Complet ===")
img = Image.open('images/texte.png')
texte = ocrTexteComplet(img)
print(f"Texte reconnu : {texte}")