import numpy as np
import os
import gzip
import struct
import zipfile
import urllib.request
from scipy.ndimage import zoom

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(x):
    classes = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
    return classes[np.argmax(x)]

def cross_entropy(sortie, label):
    return -np.sum(label * np.log(sortie + 1e-9))

def one_hot(label, num_classes=62):
    y = np.zeros(num_classes)
    y[label] = 1
    return y

def _download_emnist(dest):
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "gzip.zip")
    print("Téléchargement du dataset EMNIST (peut prendre plusieurs minutes)...")
    req = urllib.request.Request(EMNIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded * 100 // total}%", end="", flush=True)
    print("\nExtraction...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            filename = os.path.basename(name)
            if 'byclass' in filename and filename:
                with z.open(name) as src, open(os.path.join(dest, filename), 'wb') as dst:
                    dst.write(src.read())
    os.remove(zip_path)
    print("Dataset prêt.")

def _parse_idx_images(path):
    with gzip.open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>4i', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(n, rows, cols).astype(np.float32) / 255.0
    return np.transpose(images, (0, 2, 1))

def _parse_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        _, _ = struct.unpack('>2i', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int32)

def _parse_mapping(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            mapping[int(parts[0])] = chr(int(parts[1]))
    return mapping

def load_dataset_MNIST(split="train"):
    dataset_dir = "dataset"
    images_path = os.path.join(dataset_dir, f"emnist-byclass-{split}-images-idx3-ubyte.gz")
    labels_path = os.path.join(dataset_dir, f"emnist-byclass-{split}-labels-idx1-ubyte.gz")
    mapping_path = os.path.join(dataset_dir, "emnist-byclass-mapping.txt")

    if not os.path.exists(images_path):
        _download_emnist(dataset_dir)

    print(f"Chargement des images ({split})...")
    images = _parse_idx_images(images_path)
    labels = _parse_idx_labels(labels_path)
    mapping = _parse_mapping(mapping_path)

    print("Redimensionnement des images (28x28 -> 32x32)...")
    images = np.array([zoom(img, 32 / 28) for img in images])

    return images, labels, mapping
