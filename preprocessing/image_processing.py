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
    
    Returns:
        PIL.Image: Image en niveaux de gris, valeurs de 0 à 255.
    
    """
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
    return Image.fromarray(arrGray)
     

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

def decouper(image,posx,posy,hauteur,largeur):
    """
    Découpe une zone rectangulaire d'une image from scratch.
    
    Args:
        image (PIL.Image): Image source
        posx (int): Position X du coin supérieur gauche
        posy (int): Position Y du coin supérieur gauche
        hauteur (int): Hauteur de la zone à extraire
        largeur (int): Largeur de la zone à extraire
    
    Returns:
        PIL.Image ou None: Zone découpée ou None si invalide
    
    Notes:
        La zone ne doit pas dépasser les limites de l'image source.
        Retourne None si la zone est invalide.
    """
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
    """
    Redimensionne une image à la taille spécifiée from scratch.
    Utilise l'algorithme Nearest Neighbor (plus proche voisin).
    
    Pour chaque pixel de l'image destination, calcule quelle position
    il représente dans l'image source et copie le pixel le plus proche.
    
    Args:
        image (PIL.Image): Image source
        taille (tuple): (largeur, hauteur) de destination, par défaut (32, 32)
    
    Returns:
        Image redimensionnée à la taille spécifiée
    
    Notes:
        - Algorithme : Nearest Neighbor (rapide mais pixelisé si agrandissement)
        - Pour chaque pixel destination, prend le pixel source le plus proche
        - Ratio calculé : largeur_source / largeur_destination
        - Utilise NumPy pour manipuler les matrices de pixels
        - La fonction construit la nouvelle image from scratch avec des boucles
      
    """
    largeur_dest, hauteur_dest = taille
    matrice_src = imageVersMatrice(image)
    hauteur_src,largeur_src = matrice_src.shape[:2]
    if len(matrice_src.shape) == 2:
        matrice_dest = np.zeros((hauteur_dest, largeur_dest), dtype=np.uint8)
    else:
        nb_canaux = matrice_src.shape[2]
        matrice_dest = np.zeros((hauteur_dest, largeur_dest, nb_canaux), dtype=np.uint8)
    ratio_x = largeur_src / largeur_dest
    ratio_y = hauteur_src / hauteur_dest
    for x in range(largeur_dest):
        for y in range(hauteur_dest):
            x_calc = int(x * ratio_x)
            y_calc = int(y * ratio_y)
            matrice_dest[y, x] = matrice_src[y_calc, x_calc]
    return Image.fromarray(matrice_dest)    


#============== FONCTION PIPELINE COMPLET POUR PREPROCESS ================

def preprocess_pour_ocr(image,taille=(32,32)):
    return normaliserMatrice(
        imageVersMatrice(
            redimensionner(
                passageEnGris(image),
                taille=taille
            )
        )
    )


