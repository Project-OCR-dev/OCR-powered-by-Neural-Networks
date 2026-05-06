import numpy as np
import os

CLASSES = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def predict(x):
    return CLASSES[np.argmax(x, axis=-1)]

def one_hot(labels, num_classes=62):
    y = np.zeros((len(labels), num_classes))
    y[np.arange(len(labels)), labels] = 1
    return y

def _resize(img, size=32):
    idx = (np.arange(size) * img.shape[0] / size).astype(int)
    return img[np.ix_(idx, idx)]

def load_dataset(split="train"):
    path = os.path.join("dataset", f"emnist-byclass-{split}.csv")
    print(f"Chargement {path}...")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    labels = data[:, 0].astype(np.int32)
    images = data[:, 1:].reshape(-1, 28, 28) / 255.0
    images = np.transpose(images, (0, 2, 1))
    return np.array([_resize(img) for img in images]), labels
