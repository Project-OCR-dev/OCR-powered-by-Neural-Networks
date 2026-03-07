from PIL import Image,ImageFilter
import numpy as np

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
    arr = np.array(image)
    hauteur, largeur = arr.shape[:2]
    largeur, longueur = image.size
    
    print("Mode:", img.mode)
    totalPixels = largeur * longueur
    #image.show()
    for y in range(hauteur):
        for x in range(largeur):
            r,g,b,a = arr[y, x] 
            if r == 255 and g == 255 and b == 255:
                image.putpixel(arr[y,x],(0,0,0))
            
    

passageEnGris(img)

        