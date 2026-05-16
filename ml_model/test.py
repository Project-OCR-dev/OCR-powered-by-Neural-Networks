import numpy as np
import pickle
import os
from convolution import params
from forward import forward
from utils import load_dataset, CLASSES

BATCH_SIZE = 256


def _find_latest_checkpoint():
    files = [f for f in os.listdir(".") if f.startswith("params_epoch_") and f.endswith(".pkl")]
    if not files:
        return None
    return max(files, key=lambda f: int(f.split("_")[2].split(".")[0]))


ckpt = _find_latest_checkpoint()
if ckpt:
    with open(ckpt, "rb") as f:
        params.update(pickle.load(f))
    print(f"Poids chargés : {ckpt}")
elif os.path.exists("params_trained.pkl"):
    with open("params_trained.pkl", "rb") as f:
        params.update(pickle.load(f))
    print("Poids chargés : params_trained.pkl")
else:
    print("Aucun fichier de poids trouvé.")
    exit(1)

images, labels = load_dataset("test")
n = len(images)
print(f"{n} images de test chargées.")

correct = 0
for i in range(0, n, BATCH_SIZE):
    bx = images[i:i + BATCH_SIZE]
    by = labels[i:i + BATCH_SIZE]
    z3 = forward(bx)[0]
    preds = np.argmax(z3, axis=1)
    correct += (preds == by).sum()
    print(f"\r  batch {min(i + BATCH_SIZE, n)}/{n}  acc={correct / min(i + BATCH_SIZE, n):.4f}", end="", flush=True)

print()
print(f"\nPrécision finale sur le test set : {correct}/{n} = {correct / n * 100:.2f}%")

# Précision par classe
print("\nPrécision par classe :")
for cls_idx in range(len(CLASSES)):
    mask = labels == cls_idx
    if mask.sum() == 0:
        continue
    cls_images = images[mask]
    cls_labels = labels[mask]
    cls_correct = 0
    for i in range(0, len(cls_images), BATCH_SIZE):
        bx = cls_images[i:i + BATCH_SIZE]
        z3 = forward(bx)[0]
        preds = np.argmax(z3, axis=1)
        cls_correct += (preds == cls_labels[i:i + BATCH_SIZE]).sum()
    print(f"  '{CLASSES[cls_idx]}' : {cls_correct}/{mask.sum()} ({cls_correct / mask.sum() * 100:.1f}%)")
