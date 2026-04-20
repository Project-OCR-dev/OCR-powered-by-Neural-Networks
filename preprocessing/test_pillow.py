from PIL import Image,ImageFilter
import numpy as np
import os

img = Image.open('../backend/static/uploads/img_test.png')

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

def redimensionner(image, taille=(32,32)):
    largeur_dest, hauteur_dest = taille
    matrice_src = imageVersMatrice(image)
    hauteur_src,largeur_src = matrice_src.shape[:2]
    matrice_dest = np.zeros((hauteur_dest,largeur_dest,3), dtype=np.uint8)
    ratio_x = largeur_src / largeur_dest
    ratio_y = hauteur_src / hauteur_dest
    for x in range(largeur_dest):
        for y in range(hauteur_dest):
            x_calc = int(x * ratio_x)
            y_calc = int(y * ratio_y)
            matrice_dest[y, x] = matrice_src[y_calc, x_calc]
    return Image.fromarray(matrice_dest)    

def preprocess_pour_ocr(image,taille=(32,32)):
    return normaliserMatrice(
        imageVersMatrice(
            redimensionner(
                passageEnGris(image),
                taille=taille
            )
        )
    )



# === Test des différentes fonctions ===

# Découper une zone
zone = decouper(img, posx=60, posy=50, hauteur=230, largeur=280)
print(f"Zone découpée : {zone.size}")
zone.show()

# Redimensionner une image en. 32x32
image_red = redimensionner(zone,taille=(32,32))
image_red.show()


# Passage en nuance de gris
image = passageEnGris(image_red)
image.show()

# image pillow vers matrice numpy
mat = imageVersMatrice(img)
print(f"pixel : {mat[216,500]}")

# normaliser la matrice (valeur entre 0 et 1)
matnorm = normaliserMatrice(mat)
print(f"pixel : {matnorm[216,500]}")


# Test complet
print("=== TEST PIPELINE ===")
print(f"Image originale : {img.size}, mode : {img.mode}")

matrice = preprocess_pour_ocr(img)

print(f" Shape : {matrice.shape}")
print(f" Type : {matrice.dtype}")
print(f" Min : {matrice.min():.3f}")
print(f" Max : {matrice.max():.3f}")
print(f" Prête pour ML : OUI")

