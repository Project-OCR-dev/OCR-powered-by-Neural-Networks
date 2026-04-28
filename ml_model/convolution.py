import numpy as np

params = {
    "K1": np.random.randn(32, 1, 3, 3) * 0.01,
    "K2": np.random.randn(64, 32, 3, 3) * 0.01,
    "W1": np.random.randn(256, 2304) * 0.01,
    "b1": np.zeros(256),
    "W2": np.random.randn(128, 256) * 0.01,
    "b2": np.zeros(128),
    "W3": np.random.randn(62, 128) * 0.01,
    "b3": np.zeros(62),
}

def convolution(features_map, repetition, kernels, canaux=1):
    features_map_3D = []
    for rep in range(repetition):
        canal = features_map[0] if canaux > 1 else features_map
        map_h, map_l = np.shape(canal)
        kernel_size = kernels[rep][0].shape[0]
        output_size = map_l - kernel_size + 1
        output_array = np.zeros((output_size, output_size))
        for c in range(canaux):
            canal = features_map[c] if canaux > 1 else features_map
            kernel = kernels[rep][c]
            for i in range(output_size):
                for j in range(output_size):
                    somme = 0
                    for x in range(kernel_size):
                        for y in range(kernel_size):
                            somme += kernel[x, y] * canal[i+x, j+y]
                    output_array[i, j] += somme
        features_map_3D.append(output_array)
    return np.array(features_map_3D)


def maxpooling(arr):
    map_h, map_l = np.shape(arr)
    output_size = map_h // 2
    output_array = np.zeros((output_size, output_size))
    for i in range(output_size):
        for j in range(output_size):
            max = arr[i*2, j*2]
            for x in range(2):
                for y in range(2):
                    if max < arr[x+i*2, y+j*2]:
                        max = arr[x+i*2, y+j*2]
            output_array[i, j] = max
    return output_array
