from PIL import Image
import numpy as np


#ouvre l'image initial
im = Image.open("8.png")
im.show()


#passage en niveau de gris et affichage
gray = im.convert('L')
gray.show()
#conversion image en matrice numpy
map = np.asarray(gray)
#déclaration kernel

kernel = np.array([[-1,0,1],
                  [-1,0,1],
                  [-1,0,1]])

#convolution
def convolution():
    kernel_h, kernel_l = np.shape(kernel)
    map_h, map_l = np.shape(map)
    output_size = map_l - kernel_l + 1 #formule Taille_sortie après convulation = (Taille_entrée - Taille_kernel + 2×Padding) / Stride + 1
    output_array = np.zeros((output_size,output_size))
    for i in range(output_size):
        for j in range(output_size):
            somme = 0
            for x in range(kernel_l):
                for y in range(kernel_h):
                        somme += kernel[x,y] * map[i+x,j+y]
            output_array[i,j] = somme
    return output_array


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
    output = convolution()
    output = relu(output)
    output_normalized = normalized(output)
    output_image = Image.fromarray(output_normalized)                                          
    output_image.show()
    output_image.save("features_map.png")
    output2 = maxpooling(output)
    output2 = normalized(output2)
    output_image2 = Image.fromarray(output2)                                          
    output_image2.show()
    output_image2.save("features_map2.png")