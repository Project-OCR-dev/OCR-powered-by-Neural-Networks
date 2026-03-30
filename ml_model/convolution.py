from PIL import Image
import numpy as np
import random
import os


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

