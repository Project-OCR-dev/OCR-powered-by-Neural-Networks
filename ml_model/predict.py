import numpy as np
from forward import forward
from utils import CLASSES


def predict_image(matrix):
    x = matrix.astype(np.float32)[np.newaxis]
    z3 = forward(x)[0][0]
    return str(CLASSES[np.argmax(z3)])
