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

#fonction d'activation ReLU
def relu(x):
    return np.maximum(0, x)
#déclencheur du script :
#affichage de la feature map générée à partir du kernel donnée 
if __name__ == "__main__":
    output = convolution()
    output = relu(output)
    output = ((output - output.min()) / (output.max() - output.min()) * 255).astype('uint8') #normalisation et format necessaire au passge en image via pillow 
    output_image = Image.fromarray(output)                                                   #(valeur entiere entre 0 et 255, le astype converti les float en int sur 8 bit)
    output_image.show()
    output_image.save("features_map.png")