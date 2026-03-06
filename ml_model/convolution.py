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



#convolution
def convolution(features_map,repetition):
    features_map_3D = []
    for i in range(repetition):
        #déclaration kernel
        kernel = np.array([[random.randint(-1,1),random.randint(-1,1),random.randint(-1,1)],
                    [random.randint(-1,1),random.randint(-1,1),random.randint(-1,1)],
                    [random.randint(-1,1),random.randint(-1,1),random.randint(-1,1)]])
        kernel_h, kernel_l = np.shape(kernel)
        map_h, map_l = np.shape(features_map)
        output_size = map_l - kernel_l + 1 #formule Taille_sortie après convulation = (Taille_entrée - Taille_kernel + 2×Padding) / Stride + 1
        output_array = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                somme = 0
                for x in range(kernel_l):
                    for y in range(kernel_h):
                            somme += kernel[x,y] * features_map[i+x,j+y]
                output_array[i,j] = somme
        features_map_3D.append(output_array);   
    return np.array(features_map_3D)


def maxpooling(arr):
     map_h, map_l = np.shape(arr)
     output_size = map_h // 2
     output_array = np.zeros((output_size,output_size))
     for i in range(output_size):
        for j in range(output_size):
            max =  arr[i*2, j*2]
            for x in range(2):
                for y in range(2):
                        if max < arr[x+i*2,y+j*2]:
                             max = arr[x+i*2,y+j*2]
            output_array[i,j] = max
     return output_array

#fonction d'activation ReLU
def relu(x):
    return np.maximum(0, x)

#normalisation
#normalisation et format necessaire au passge en image via pillow 
#(valeur entiere entre 0 et 255, le astype converti les float en int sur 8 bit)

def normalized(out):
     return ((out - out.min()) / (out.max() - out.min()) * 255).astype('uint8') 

#déclencheur du script :

#affichage de la feature map générée à partir du kernel donnée 
if __name__ == "__main__":
    folder_path = "./output"
    folder_path2 = "./output2"
    # Crée le dossier s'il n'existe pas
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path2, exist_ok=True)

    output = convolution(map,32)
    for i in range(32):
        fmap = relu(output[i])
        fmap_maxpool = maxpooling(fmap)
        fmap_normalized = normalized(fmap_maxpool)
        fmap_image = Image.fromarray(fmap_normalized)
        # Chemin complet pour sauvegarder l'image
        file_path = os.path.join(folder_path, f"map{i+1}.png")
        img = fmap_image
        # Sauvegarde l'image
        img.save(file_path) 

    output2 = convolution(fmap_maxpool,64)
    for i in range(64):
        fmap2 = relu(output2[i])
        fmap_maxpool2 = maxpooling(fmap2)
        fmap_normalized2 = normalized(fmap_maxpool2)
        fmap_image2 = Image.fromarray(fmap_normalized2)
        # Chemin complet pour sauvegarder l'image
        file_path = os.path.join(folder_path2, f"map{i+1}.png")
        img = fmap_image2
        # Sauvegarde l'image
        img.save(file_path) 
