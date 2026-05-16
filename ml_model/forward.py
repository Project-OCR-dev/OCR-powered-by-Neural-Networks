from convolution import convolution, maxpooling
from utils import relu
from convolution import params
import numpy as np


def forward(map):
    output = convolution(map, 32, params["K1"])
    all_fmap_maxpool = []
    all_fmap_maxpool2 = []

    for i in range(32):
        fmap = relu(output[i])
        fmap_maxpool = maxpooling(fmap)
        all_fmap_maxpool.append(fmap_maxpool)

    all_fmap_maxpool = np.array(all_fmap_maxpool)
    output2 = convolution(all_fmap_maxpool, 64, params["K2"], canaux=32)

    for i in range(64):
        fmap2 = relu(output2[i])
        fmap_maxpool2 = maxpooling(fmap2)
        all_fmap_maxpool2.append(fmap_maxpool2)

    flat = np.array(all_fmap_maxpool2).flatten()

    x1 = relu(np.dot(params["W1"], flat) + params["b1"])
    x2 = relu(np.dot(params["W2"], x1) + params["b2"])
    x3 = np.dot(params["W3"], x2) + params["b3"]

    return x3