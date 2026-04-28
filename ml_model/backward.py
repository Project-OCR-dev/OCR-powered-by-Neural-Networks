
from convolution import params
from scipy.signal import correlate2d
import numpy as np

def backward(proba, y, x1, x2, flat, entree_conv1, entree_conv2, sortie_conv1, sortie_conv2, lr=0.001):

    dL_dZ3 = proba - y
    dL_dW3 = np.outer(dL_dZ3, x2)
    dL_db3 = dL_dZ3

    dL_dx2 = np.dot(params["W3"].T, dL_dZ3)
    dL_dZ2 = dL_dx2 * (x2 > 0)
    dL_dW2 = np.outer(dL_dZ2, x1)
    dL_db2 = dL_dZ2

    dL_dx1 = np.dot(params["W2"].T, dL_dZ2)
    dL_dZ1 = dL_dx1 * (x1 > 0)
    dL_dW1 = np.outer(dL_dZ1, flat)
    dL_db1 = dL_dZ1

    dL_dK2 = np.zeros_like(params["K2"])
    dL_dFlat = np.dot(params["W1"].T, dL_dZ1)
    dL_dFmap2 = dL_dFlat.reshape(64, 6, 6)

    dL_dSortie_conv2 = np.zeros(sortie_conv2.shape)
    for f in range(64):
        for i in range(6):
            for j in range(6):
                dL_dSortie_conv2[f, i*2, j*2] = dL_dFmap2[f, i, j]

    for rep in range(64):
        for c in range(32):
            dL_dK2[rep, c] += correlate2d(entree_conv2[c], dL_dSortie_conv2[rep], mode='valid')

    dL_dK1 = np.zeros_like(params["K1"])
    dL_dEntree_conv2 = np.zeros(entree_conv2.shape)

    for rep in range(64):
        for c in range(32):
            dL_dEntree_conv2[c] += correlate2d(dL_dSortie_conv2[rep], params["K2"][rep,c], mode='full')

    dL_dSortie_conv1 = np.zeros(sortie_conv1.shape)
    for f in range(32):
        for i in range(15):
            for j in range(15):
                dL_dSortie_conv1[f, i*2, j*2] = dL_dEntree_conv2[f, i, j]

    dL_dSortie_conv1 = dL_dSortie_conv1 * (sortie_conv1 > 0)

    for rep in range(32):
        dL_dK1[rep, 0] += correlate2d(entree_conv1, dL_dSortie_conv1[rep], mode='valid')

    params["W3"] -= lr * dL_dW3
    params["b3"] -= lr * dL_db3
    params["W2"] -= lr * dL_dW2
    params["b2"] -= lr * dL_db2
    params["W1"] -= lr * dL_dW1
    params["b1"] -= lr * dL_db1
    params["K2"] -= lr * dL_dK2
    params["K1"] -= lr * dL_dK1