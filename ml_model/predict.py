import numpy as np
import pickle
import os
from convolution import params
from forward import forward
from utils import CLASSES

_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_weights():
    files = [f for f in os.listdir(_DIR) if f.startswith("params_epoch_") and f.endswith(".pkl")]
    if files:
        ckpt = max(files, key=lambda f: int(f.split("_")[2].split(".")[0]))
        with open(os.path.join(_DIR, ckpt), "rb") as f:
            params.update(pickle.load(f))
        return
    path = os.path.join(_DIR, "params_trained.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            params.update(pickle.load(f))

_load_weights()

def predict_image(matrix):
    x3 = forward(matrix)
    return str(CLASSES[np.argmax(x3)])
