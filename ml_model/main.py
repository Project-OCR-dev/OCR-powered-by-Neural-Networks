from torch import classes

from convolution import convolution
from convolution import maxpooling
from utils import relu, normalized, softmax

from PIL import Image
import numpy as np
import random
import os



#ouvre l'image initial
im = Image.open("./input/8.png")

#passage en niveau de gris et affichage
gray = im.convert('L')

#conversion image en matrice numpy
map = np.asarray(gray)

def forward():
    folder_path = "./output"
    folder_path2 = "./output2"
    # Crée le dossier s'il n'existe pas
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path2, exist_ok=True)

    output = convolution(map,32)
    all_fmap_maxpool = []
    all_fmap_maxpool2 = []

    for i in range(32):

        fmap = relu(output[i])
        fmap_maxpool = maxpooling(fmap)

        # Normalisation de la feature map pour Pillow (0-255)
        fmap_normalized = normalized(fmap_maxpool) 
        fmap_image = Image.fromarray(fmap_normalized)

        # Chemin complet pour sauvegarder l'image
        file_path = os.path.join(folder_path, f"fmap_{i+1}.png")
        img = fmap_image

        # Sauvegarde l'image
        img.save(file_path) 

        all_fmap_maxpool.append(fmap_maxpool)

    all_fmap_maxpool = np.array(all_fmap_maxpool)
    output2 = convolution(all_fmap_maxpool,64,canaux=32)
    
    for i in range(64):

        fmap2 = relu(output2[i])
        fmap_maxpool2 = maxpooling(fmap2)

        fmap_normalized2 = normalized(fmap_maxpool2)
        fmap_image2 = Image.fromarray(fmap_normalized2)

        # Chemin complet pour sauvegarder l'image
        file_path = os.path.join(folder_path2, f"fmap2_{i+1}.png")
        img = fmap_image2

        # Sauvegarde l'image
        img.save(file_path) 
        all_fmap_maxpool2.append(fmap_maxpool2)



    flat = np.array(all_fmap_maxpool2).flatten()

    W1 = np.random.randn(256, 2304) * 0.01
    b1 = np.zeros(256)

    W2 = np.random.randn(128, 256) * 0.01
    b2 = np.zeros(128)

    W3 = np.random.randn(62, 128) * 0.01
    b3 = np.zeros(62)

    x = relu(np.dot(W1, flat) + b1)
    x = relu(np.dot(W2, x) + b2)
    x = np.dot(W3, x) + b3

    return x
    

def backward():
    pass


def predict(x):
    x = softmax(x)
    classes = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
    pred = classes[np.argmax(x)]
    print(f"la lettre prédite est : {pred}")


#déclencheur du script :
if __name__ == "__main__":
    proba = forward()
    predict(proba)