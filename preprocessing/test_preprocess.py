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
            gray = (int)(r+g+b)/3
            arrGray[y,x] = gray
    imageResult= Image.fromarray(arrGray)
    return imageResult

#passageEnGris(img)

def imageVersMatrice(image):
    largeur, hauteur = image.size
    # Cas 2 : Grayscale (mode 'L')
    if image.mode == 'L':
        print(f"Conversion image grayscale {largeur}×{hauteur} en matrice 2D")
        arrMatrice = np.zeros((hauteur, largeur), dtype=np.uint8)
        for y in range(hauteur):
            for x in range(largeur):
                gray = image.getpixel((x, y))
                arrMatrice[y, x] = gray
        
        print(f"Matrice créée : shape {arrMatrice.shape}, dtype {arrMatrice.dtype}")
        return arrMatrice
    # Cas 2 : RGB (mode 'RGB')
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
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

def calculerSeuilOptimal(matrice):
    # Méthode Otsu #

    # Calculer histogramme (combien de pixels de chaque valeur)
    histogramme = np.zeros(256, dtype=int)
    hauteur, largeur = matrice.shape
    
    for y in range(hauteur):
        for x in range(largeur):
            valeur = matrice[y, x]
            histogramme[valeur] += 1
    
    # Total pixels
    total_pixels = hauteur * largeur
    
    # Calculer somme totale pondérée
    somme_totale = 0
    for i in range(256):
        somme_totale += i * histogramme[i]
    
    # Trouver le seuil optimal
    somme_fond = 0
    poids_fond = 0
    variance_max = 0
    seuil_optimal = 0
    
    for t in range(256):
        # Poids fond (pixels <= t)
        poids_fond += histogramme[t]
        if poids_fond == 0:
            continue
        
        # Poids objet (pixels > t)
        poids_objet = total_pixels - poids_fond
        if poids_objet == 0:
            break
        
        # Somme fond
        somme_fond += t * histogramme[t]
        
        # Moyennes
        moyenne_fond = somme_fond / poids_fond
        moyenne_objet = (somme_totale - somme_fond) / poids_objet
        
        # Variance inter-classes
        variance = poids_fond * poids_objet * (moyenne_fond - moyenne_objet) ** 2
        
        # Garder le seuil avec variance maximale
        if variance > variance_max:
            variance_max = variance
            seuil_optimal = t

    return seuil_optimal

def binariser(matrice, seuil=None):
    hauteur, largeur = matrice.shape[:2]
    if seuil is None:
        seuil = calculerSeuilOptimal(matrice)
        print(f"Seuil optimal calculé : {seuil}")
    matrice_bin = np.zeros((hauteur, largeur), dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            if matrice[y, x] >= seuil:
                matrice_bin[y, x] = 255  # Blanc (fond)
            else:
                matrice_bin[y, x] = 0    # Noir (texte)
    return matrice_bin

#pipeline temporaire en attendant la segmentation
def preprocess_pour_ocr(image,taille=(32,32)):
    # 1. Image RGB → Grayscale (Image Pillow)
    img_gray = passageEnGris(image)
    # 2. Image → Matrice (NumPy)
    matrice_gray = imageVersMatrice(img_gray)
    # 3. Binarisation Otsu (Matrice NumPy)
    matrice_bin = binariser(matrice_gray)
    # 4. Matrice → Image (pour découper)
    img_bin = Image.fromarray(matrice_bin)
    # 5. Découper (Image Pillow)
    lettre = decouper(img_bin, posx=440 , posy=30, hauteur=350, largeur=280)
    # 6. Redimensionner (Image Pillow)
    img_resized = redimensionner(lettre, taille=taille)
    img_resized.show()
    # 7. Convertir en matrice (NumPy)
    matrice = imageVersMatrice(img_resized)
    # 8. Normaliser (NumPy)
    matrice_norm = normaliserMatrice(matrice)
    return matrice_norm
    



# === Test des différentes fonctions ===


img = Image.open('../backend/static/uploads/img_test2.png')
lettre = decouper(img, posx=440 , posy=30, hauteur=350, largeur=280)
lettre.show()
lettref=preprocess_pour_ocr(img)




