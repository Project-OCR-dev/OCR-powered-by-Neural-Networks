from PIL import Image
import numpy as np
import os

def passageEnGris(image):
    """
    Convertit une image RGB en niveaux de gris (grayscale) from scratch.
    
    Utilise la méthode de la moyenne simple : gray = (R + G + B) / 3
    Parcourt tous les pixels de l'image et calcule la valeur de gris
    pour chaque pixel en faisant la moyenne des canaux RGB.
    
    Args:
        image (PIL.Image ou str): Image Pillow en mode RGB ou chemin vers l'image.
                                  Si l'image n'est pas en RGB, elle sera convertie.
        sauvegarder (bool, optional): Si True, sauvegarde l'image résultante dans
                                       le dossier './processImg/'. Par défaut True.
    
    Returns:
        PIL.Image: Image en niveaux de gris, valeurs de 0 à 255.
    
    """
    img = Image.open(f"../backend/static/uploads/{image}")
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
    arrGray = np.zeros((hauteur, largeur),dtype=np.uint8)
    for x in range(hauteur):
        for y in range(largeur):
            r,g,b = arrImg[x,y]
            r=int(r)
            g=int(g)
            b=int(b)
            gray = (r+g+b)/3
            gray = int(gray)
            arrGray[y,x] = gray
    imageResult= Image.fromarray(arrGray)
    imageResult.save(f"./processImg/{nameWithoutExt}_gray{ext}")
    return imageResult

def imageVersMatrice(image):
    """
    Convertit une image Pillow en matrice NumPy from scratch.
    
    Parcourt tous les pixels de l'image un par un et construit manuellement
    une matrice NumPy 3D contenant les valeurs RGB de chaque pixel.
    
    Args:
        image (PIL.Image): Image Pillow.
                                  L'image sera automatiquement convertie en RGB
                                  si elle est dans un autre mode (L, RGBA, etc.).
    
    Returns:
        numpy.ndarray: Matrice 3D de shape (hauteur, largeur, 3) contenant
                       les valeurs RGB des pixels. Type uint8, valeurs 0-255.
                       - arrMatrice[y, x, 0] = Rouge
                       - arrMatrice[y, x, 1] = Vert
                       - arrMatrice[y, x, 2] = Bleu

    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    largeur, hauteur = image.size
    arrMatrice = np.zeros((hauteur, largeur,3),dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            r,g,b = image.getpixel((x,y))
            arrMatrice[y,x,0] = r
            arrMatrice[y,x,1] = g
            arrMatrice[y,x,2] = b
    return arrMatrice

def normaliserMatrice(matrice):
    """
    Normalise une matrice de pixels (0-255) vers (0-1) from scratch.
    
    Args:
        matrice (numpy.ndarray): Matrice shape (hauteur, largeur, 3), dtype uint8
    
    Returns:
        numpy.ndarray: Matrice normalisée shape (hauteur, largeur, 3), dtype float32
    
    Notes:
        Divise chaque pixel par 255.0 pour obtenir des valeurs entre 0 et 1,
        format requis par les réseaux de neurones pour accélérer l'apprentissage et 
        améliorer la précision des prédictions
    """
    hauteur, largeur, canaux = matrice.shape
    print(f"largeur {largeur} - hauteur {hauteur}")
    arrNorm = np.zeros((hauteur, largeur,3),dtype=np.float32)
    for x in range(hauteur):
        for y in range(largeur):
            for z in range(canaux):
                arrNorm[x,y,z] = matrice[x, y, z] / 255.0
    return arrNorm

