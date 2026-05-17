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
    
    Args:
        image: Image Pillow en mode RGB
    
    Returns:
        PIL.Image: Image en niveaux de gris
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    arrImg = np.array(image)
    hauteur, largeur = arrImg.shape[:2]  
    arrGray = np.zeros((hauteur, largeur), dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            r, g, b = arrImg[y, x]
            r = int(r)
            g = int(g)
            b = int(b)
            gray = (r + g + b) / 3
            gray = int(gray)
            arrGray[y, x] = gray
    return Image.fromarray(arrGray)
     

def imageVersMatrice(image):
    """
    Convertit une image Pillow grayscale en matrice NumPy from scratch.
    
    Args:
        image: Image Pillow en mode 'L' (grayscale)
    
    Returns:
        numpy.ndarray: Matrice 2D (hauteur, largeur)
    """
    largeur, hauteur = image.size
    arrMatrice = np.zeros((hauteur, largeur), dtype=np.uint8)
    
    for y in range(hauteur):
        for x in range(largeur):
            pixel = image.getpixel((x, y))
            if isinstance(pixel, tuple):
                arrMatrice[y, x] = pixel[0]
            else:
                arrMatrice[y, x] = pixel
    
    return arrMatrice


def normaliserMatrice(matrice):
    """
    Normalise une matrice grayscale (0-255 vers 0-1) from scratch.
    
    Args:
        matrice: Matrice 2D, dtype uint8
    
    Returns:
        numpy.ndarray: Matrice 2D normalisée, dtype float32
    """
    hauteur, largeur = matrice.shape
    arrNorm = np.zeros((hauteur, largeur), dtype=np.float32)
    
    for y in range(hauteur):
        for x in range(largeur):
            arrNorm[y, x] = matrice[y, x] / 255.0
    
    return arrNorm


def decouper(image, posx, posy, hauteur, largeur):
    """
    Découpe une zone rectangulaire d'une image from scratch.
    
    Args:
        image: Image source
        posx: Position X du coin supérieur gauche
        posy: Position Y du coin supérieur gauche
        hauteur: Hauteur de la zone
        largeur: Largeur de la zone
    
    Returns:
        PIL.Image ou None: Zone découpée ou None si invalide
    """
    matrice = imageVersMatrice(image)
    h_source, l_source = matrice.shape
    
    if (posx < 0 or posy < 0 or 
        posx + largeur > l_source or 
        posy + hauteur > h_source):
        return None
    
    matriceZone = np.zeros((hauteur, largeur), dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            matriceZone[y, x] = matrice[y + posy, x + posx]
    
    return Image.fromarray(matriceZone)


def redimensionner(image, taille=(32, 32)):
    """
    Redimensionne une image from scratch.
    Utilise l'algorithme Nearest Neighbor.
    
    Args:
        image: Image source
        taille: (largeur, hauteur) de destination
    
    Returns:
        numpy.ndarray: Matrice 2D redimensionnée
    """
    largeur_dest, hauteur_dest = taille
    matrice_src = imageVersMatrice(image)
    hauteur_src, largeur_src = matrice_src.shape
    
    matrice_dest = np.zeros((hauteur_dest, largeur_dest), dtype=np.uint8)
    
    ratio_x = largeur_src / largeur_dest
    ratio_y = hauteur_src / hauteur_dest
    
    for y in range(hauteur_dest):
        for x in range(largeur_dest):
            x_calc = int(x * ratio_x)
            y_calc = int(y * ratio_y)
            matrice_dest[y, x] = matrice_src[y_calc, x_calc]
    
    return matrice_dest


########### BINARISATION ###########

def calculerSeuilOptimal(matrice):
    """
    Calcule le seuil optimal de binarisation par la méthode d'Otsu.
    
    Args:
        matrice: Matrice grayscale 2D
    
    Returns:
        int: Seuil optimal (0-255)
    """
    histogramme = np.zeros(256, dtype=int)
    hauteur, largeur = matrice.shape

    for y in range(hauteur):
        for x in range(largeur):
            valeur = matrice[y, x]
            histogramme[valeur] += 1

    total_pixels = hauteur * largeur

    somme_totale = 0
    for i in range(256):
        somme_totale += i * histogramme[i]

    somme_fond = 0
    poids_fond = 0
    variance_max = 0
    seuil_optimal = 0

    for t in range(256):
        poids_fond += histogramme[t]
        if poids_fond == 0:
            continue
        
        poids_objet = total_pixels - poids_fond
        if poids_objet == 0:
            break
        
        somme_fond += t * histogramme[t]
        
        moyenne_fond = somme_fond / poids_fond
        moyenne_objet = (somme_totale - somme_fond) / poids_objet
        
        variance = poids_fond * poids_objet * (moyenne_fond - moyenne_objet) ** 2
        
        if variance > variance_max:
            variance_max = variance
            seuil_optimal = t

    return seuil_optimal


def binariser(matrice, seuil=None):
    """
    Binarise une image grayscale.
    
    Args:
        matrice: Matrice grayscale 2D
        seuil: Seuil de binarisation (None = Otsu auto)
    
    Returns:
        numpy.ndarray: Matrice binarisée 2D (0=noir, 255=blanc)
    """
    hauteur, largeur = matrice.shape
    if seuil is None:
        seuil = calculerSeuilOptimal(matrice)
    
    matrice_bin = np.zeros((hauteur, largeur), dtype=np.uint8)
    for y in range(hauteur):
        for x in range(largeur):
            if matrice[y, x] >= seuil:
                matrice_bin[y, x] = 255
            else:
                matrice_bin[y, x] = 0
    
    return matrice_bin


########### SEGMENTATION ###########

def projectionVerticale(matrice):
    """
    Compte les pixels noirs par colonne.
    
    Args:
        matrice: Matrice binarisée 2D
    
    Returns:
        list: Nombre de pixels noirs par colonne
    """
    hauteur, largeur = matrice.shape
    projection = []
    
    for x in range(largeur):
        compteur = 0
        for y in range(hauteur):
            if matrice[y, x] == 0:
                compteur += 1
        projection.append(compteur)

    return projection


def projectionHorizontale(matrice, x_debut, x_fin):
    """
    Compte les pixels noirs par ligne dans une zone.
    
    Args:
        matrice: Matrice binarisée 2D
        x_debut: Colonne de début
        x_fin: Colonne de fin
    
    Returns:
        list: Nombre de pixels noirs par ligne
    """
    hauteur, largeur = matrice.shape
    projection = []
    
    for y in range(hauteur):
        compteur = 0
        for x in range(x_debut, x_fin):
            if matrice[y, x] == 0:
                compteur += 1
        projection.append(compteur)

    return projection


def detecterZonesLettres(projection, seuil_min=5):
    """
    Détecte les zones contenant des lettres.
    
    Args:
        projection: Projection verticale
        seuil_min: Seuil minimum
    
    Returns:
        list: Liste de tuples (x_debut, x_fin)
    """
    zones = []
    dans_lettre = False
    x_debut = 0
    
    for x in range(len(projection)):
        if projection[x] > seuil_min and not dans_lettre:
            x_debut = x
            dans_lettre = True
        elif projection[x] <= seuil_min and dans_lettre:
            zones.append((x_debut, x))
            dans_lettre = False
    
    if dans_lettre:
        zones.append((x_debut, len(projection)))
    
    return zones


def trouverLimitesVerticales(projection, seuil_min=3):
    """
    Trouve les limites haut et bas d'une lettre.
    
    Args:
        projection: Projection horizontale
        seuil_min: Seuil minimum
    
    Returns:
        tuple: (y_debut, y_fin) ou None
    """
    y_debut = None
    y_fin = None
    
    for y in range(len(projection)):
        if projection[y] > seuil_min and y_debut is None:
            y_debut = y
        if projection[y] <= seuil_min and y_debut is not None and y_fin is None:
            y_fin = y
            break
    
    if y_debut is not None and y_fin is None:
        y_fin = len(projection)
    
    if y_debut is None:
        return None
    
    return (y_debut, y_fin)


def detecterLettres(matrice):
    """
    Détecte toutes les lettres dans l'image.
    
    Args:
        matrice: Matrice binarisée 2D
    
    Returns:
        list: Liste de dict avec clés 'x', 'y', 'w', 'h'
    """
    proj_v = projectionVerticale(matrice)
    zones = detecterZonesLettres(proj_v, seuil_min=5)
    
    lettres = []
    
    for x_debut, x_fin in zones:
        proj_h = projectionHorizontale(matrice, x_debut, x_fin)
        limites = trouverLimitesVerticales(proj_h, seuil_min=3)
        
        if limites is not None:
            y_debut, y_fin = limites
            lettres.append({
                'x': x_debut,
                'y': y_debut,
                'w': x_fin - x_debut,
                'h': y_fin - y_debut
            })
    
    return lettres


########### PIPELINES ###########

def ocrLettreIsolee(image, taille=(30, 30)):
    """
    OCR pour une seule lettre isolée.
    
    Args:
        image: Image Pillow d'une seule lettre
        taille: Taille de redimensionnement
    
    Returns:
        str: Lettre prédite
    """
    import sys
    import os
    ml_model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model')
    if ml_model_path not in sys.path:
      sys.path.insert(0, ml_model_path)
    from ml_model.predict import predict_image  
    
    img_gray = passageEnGris(image)
    matrice = imageVersMatrice(img_gray)
    matrice = 255 - matrice          # inversion ici
    img_gray = Image.fromarray(matrice)
    matrice = redimensionner(img_gray, taille=taille)
    matrice_norm = normaliserMatrice(matrice)
    
    lettre = predict_image(matrice_norm)
    
    return lettre


def ocrTexteComplet(image, taille=(30, 30), seuil_espace=0.5):
    """
    OCR pour un texte complet (lettres espacées).
    
    Args:
        image: Image Pillow avec texte
        taille: Taille de redimensionnement
        seuil_espace: Coefficient pour détecter espaces
    
    Returns:
        str: Texte reconnu
    """
    import sys
    import os
    ml_model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model')
    if ml_model_path not in sys.path:
        sys.path.insert(0, ml_model_path)
    from ml_model.predict import predict_image
    
    img_gray = passageEnGris(image)
    matrice_gray = imageVersMatrice(img_gray)
    matrice_bin = binariser(matrice_gray)
    lettres = detecterLettres(matrice_bin)
    
    if len(lettres) == 0:
        return ""
    
    img_bin = Image.fromarray(matrice_bin)
    
    liste_matrices = []
    
    for lettre in lettres:
        lettre_img = decouper(img_bin, lettre['x'], lettre['y'], 
                              lettre['h'], lettre['w'])
        if lettre_img is None:
            continue
        mat = imageVersMatrice(lettre_img)
        mat = 255 - mat                  # inversion ici
        lettre_inv = Image.fromarray(mat)   
        matrice = redimensionner(lettre_inv, taille=taille)
        matrice_norm = normaliserMatrice(matrice)
        liste_matrices.append(matrice_norm)
    
    predictions = []
    for matrice in liste_matrices:
        pred = predict_image(matrice)
        predictions.append(pred)
    
    texte = ""
    position_precedente = None
    largeur_precedente = None
    
    for i, lettre in enumerate(lettres):
        if position_precedente is not None:
            ecart = lettre['x'] - (position_precedente + largeur_precedente)
            largeur_moyenne = (lettre['w'] + largeur_precedente) / 2
            
            if ecart > largeur_moyenne * seuil_espace:
                texte += " "
        
        texte += predictions[i]
        
        position_precedente = lettre['x']
        largeur_precedente = lettre['w']
    
    return texte


####### TEST SEGMENTATION #######

def testerSegmentation(chemin_image):
    """
    Teste la segmentation sur une image.
    """
    print(f"Chargement : {chemin_image}")
    
    image = Image.open(chemin_image)
    img_gray = passageEnGris(image)
    matrice_gray = imageVersMatrice(img_gray)
    matrice_bin = binariser(matrice_gray)
    lettres = detecterLettres(matrice_bin)
    
    print(f"\nLettres détectées : {len(lettres)}")
    
    for i, lettre in enumerate(lettres):
        print(f"Lettre {i+1}: x={lettre['x']}, y={lettre['y']}, w={lettre['w']}, h={lettre['h']}")
    
    img_bin = Image.fromarray(matrice_bin)
    nom_sortie = chemin_image.replace('.png', '_binarise.png')
    img_bin.save(nom_sortie)
    print(f"\nImage binarisée sauvegardée : {nom_sortie}")
    img_bin.show()
    
    return lettres