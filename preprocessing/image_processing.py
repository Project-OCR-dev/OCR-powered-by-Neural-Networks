"""
Module de preprocessing pour OCR
Contient toutes les fonctions de traitement d'image
"""

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


# ============================================
# BINARISATION
# ============================================

def calculerSeuilOptimal(matrice):
    """
    Calcule le seuil optimal de binarisation par la méthode d'Otsu.
    
    La méthode d'Otsu détermine automatiquement le meilleur seuil en maximisant
    la variance inter-classes entre les pixels clairs (fond) et les pixels 
    foncés (texte). Cela permet une binarisation adaptative sans seuil manuel.
    
    Principe :
        Pour chaque seuil possible (0-255), l'algorithme :
        1. Sépare les pixels en deux groupes (fond/texte)
        2. Calcule la variance entre ces groupes
        3. Garde le seuil qui maximise cette variance
        
    Args:
        matrice (numpy.ndarray): Matrice grayscale 2D avec valeurs 0-255
    
    Returns:
        int: Seuil optimal (0-255) qui sépare le mieux fond et texte

    """
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
    """
    Binarise une image grayscale (convertit en noir et blanc pur).
    
    Transforme chaque pixel en noir (0) ou blanc (255) selon un seuil.
    Si aucun seuil n'est fourni, utilise la méthode d'Otsu pour calculer
    automatiquement le seuil optimal.
    
    Args:
        matrice (numpy.ndarray): Matrice grayscale 2D avec valeurs 0-255
        seuil (int, optional): Seuil de binarisation (0-255).
            Si None, calcule automatiquement le seuil par méthode d'Otsu.
            Défaut : None
    
    Returns:
        numpy.ndarray: Matrice binarisée 2D avec seulement deux valeurs :
            - 0 (noir) pour les pixels < seuil (texte)
            - 255 (blanc) pour les pixels >= seuil (fond)
    """
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

# ============================================
# SEGMENTATION
# ============================================

def projectionVerticale(matrice):
    """
    Compte les pixels noirs par colonne (projection verticale).
    
    Cette fonction analyse chaque colonne de l'image binarisée et compte
    combien de pixels noirs (texte) elle contient. Le résultat est un
    histogramme qui permet de détecter où se trouvent les lettres.
    
    Principe :
        - Parcourir chaque colonne de gauche à droite
        - Compter les pixels noirs (valeur 0) dans chaque colonne
        - Une colonne avec beaucoup de pixels noirs = lettre
        - Une colonne avec peu/pas de pixels noirs = espace
    
    Args:
        matrice_bin (numpy.ndarray): Matrice binarisée 2D (0=noir, 255=blanc)
            Issue de la fonction binariser()
    
    Returns:
        list: Liste de longueur = largeur de l'image.
            Chaque élément = nombre de pixels noirs dans la colonne correspondante.
    """
    hauteur, largeur = matrice.shape
    projection = []
    
    # Pour chaque colonne
    for x in range(largeur):
        compteur = 0
        
        # Compter les pixels noirs dans cette colonne
        for y in range(hauteur):
            if matrice[y, x] == 0:  # Pixel noir
                compteur += 1
        
        projection.append(compteur)

    return projection

def projectionHorizontale(matrice, x_debut, x_fin):
    """
    Compte les pixels noirs par ligne dans une zone donnée (projection horizontale).
    
    Cette fonction analyse chaque ligne dans une zone verticale délimitée
    par x_debut et x_fin, et compte combien de pixels noirs elle contient.
    Utilisée après projectionVerticale pour trouver les limites haut/bas
    d'une lettre.
    
    Principe :
        - Analyser uniquement les colonnes entre x_debut et x_fin
        - Pour chaque ligne, compter les pixels noirs dans cette zone
        - Les lignes avec beaucoup de pixels = partie de la lettre
        - Les lignes avec peu/pas de pixels = espace au-dessus/en-dessous
    
    Args:
        matrice (numpy.ndarray): Matrice binarisée 2D (0=noir, 255=blanc)
        x_debut (int): Colonne de début de la zone à analyser (incluse)
        x_fin (int): Colonne de fin de la zone à analyser (exclue)
    
    Returns:
        list: Liste de longueur = hauteur de l'image.
            Chaque élément = nombre de pixels noirs dans la ligne correspondante
            (uniquement dans la zone x_debut à x_fin).
    """
    hauteur, largeur = matrice.shape
    projection = []
    
    # Pour chaque colonne
    for x in range(hauteur):
        compteur = 0
        
        # Compter les pixels noirs dans cette colonne
        for x in range(x_debut, x_fin):
            if matrice[y, x] == 0:  # Pixel noir
                compteur += 1
        
        projection.append(compteur)

    return projection


#============== FONCTION PIPELINE COMPLET POUR PREPROCESS ================

#pipeline temporaire en attendant la segmentation
def preprocess_pour_ocr(image,taille=(32,32)):
    # 1. Image RGB -> Grayscale (Image Pillow)
    img_gray = passageEnGris(image)
    # 2. Image -> Matrice (NumPy)
    matrice_gray = imageVersMatrice(img_gray)
    # 3. Binarisation Otsu (Matrice NumPy)
    matrice_bin = binariser(matrice_gray)
    # 4. Matrice -> Image (pour découper)
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


