from PIL import Image,ImageFilter
import numpy as np
import os

img = Image.open('../backend/static/uploads/image_test.png')

# Afficher les informations
print("=== INFORMATIONS DE L'IMAGE ===")
print("Taille (largeur, hauteur):", img.size)
print("Mode (RGB, L, etc.):", img.mode)
print("Format (PNG, JPEG, etc.):", img.format)

# Afficher les dimensions séparément
largeur, hauteur = img.size
print(f"\nDimensions détaillées:")
print(f"  Largeur: {largeur} pixels")
print(f"  Hauteur: {hauteur} pixels")

def passageEnGris(image):
    imagePath = image.filename
    nameWithExt= os.path.basename(imagePath)
    nameWithoutExt = os.path.splitext(nameWithExt)[0]
    print(f"nom fichier : {nameWithoutExt}")
    ext = os.path.splitext(nameWithExt)[1]
    print(f"extension : {ext}")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    arrImg= np.array(image)
    hauteur, largeur = arrImg.shape[:2]  
    print("Mode:", image.mode)
    totalPixels = hauteur * largeur 
    print(f"nombre de pixels total : {totalPixels}")
    arrGray = np.zeros((hauteur, largeur),dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            r,g,b = arrImg[y,x]
            r=int(r)
            g=int(g)
            b=int(b)
            gray = (r+g+b)/3
            gray = int(gray)
            arrGray[y,x] = gray
    imageResult= Image.fromarray(arrGray)
    imageResult.save(f"./processImg/{nameWithoutExt}_gray{ext}")
    return imageResult

#passageEnGris(img)

def imageVersMatrice(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    largeur, hauteur = image.size
    print(f"Conversion image {largeur}×{hauteur} en matrice")
    arrMatrice = np.zeros((hauteur, largeur,3),dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            r,g,b = image.getpixel((x,y))
            arrMatrice[y,x,0] = r
            arrMatrice[y,x,1] = g
            arrMatrice[y,x,2] = b
    print(f"Matrice créée : shape {arrMatrice.shape}, dtype {arrMatrice.dtype}")
    return arrMatrice

imagef = passageEnGris(img)
mat = imageVersMatrice(img)


def normaliserMatrice(matrice):
    hauteur, largeur, canaux = matrice.shape
    print(f"largeur {largeur} - hauteur {hauteur}")
    arrNorm = np.zeros((hauteur, largeur,3),dtype=np.float32)
    for x in range(hauteur):
        for y in range(largeur):
            for z in range(canaux):
                arrNorm[x,y,z] = matrice[x, y, z] / 255.0
    return arrNorm

def decouper(image,posx,posy,hauteur,largeur):
    matrice = imageVersMatrice(image)
    h_source, l_source = matrice.shape[:2]
     # Vérification
    if (posx < 0 or posy < 0 or 
        posx + largeur > l_source or 
        posy + hauteur > h_source):
        print(f"Zone invalide : dépassement des limites")
        return None
    matriceZone = np.zeros((hauteur,largeur,3),dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            matriceZone[y,x] = matrice[y+posy,x+posx]
    return Image.fromarray(matriceZone)

# === Test des différentes fonctions ===

# Découper une zone
zone = decouper(img, posx=200, posy=100, hauteur=250, largeur=350)
print(f"Zone découpée : {zone.size}")
zone.show()

# Passage en nuance de gris
image = passageEnGris(img)
image.show()

# image pillow vers matrice numpy
mat = imageVersMatrice(img)
print(f"pixel : {mat[216,500]}")

# normaliser la matrice (valeur entre 0 et 1)
matnorm = normaliserMatrice(mat)
print(f"pixel : {matnorm[216,500]}")

