from PIL import Image
import numpy as np
import random
import os


#fonction d'activation ReLU
def relu(x):
    return np.maximum(0, x)

#normalisation
#normalisation et format necessaire au passge en image via pillow 
#(valeur entiere entre 0 et 255, le astype converti les float en int sur 8 bit)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def normalized(out):
     if out.max() == out.min():
        return np.zeros_like(out, dtype='uint8')
     return ((out - out.min()) / (out.max() - out.min()) * 255).astype('uint8') 

