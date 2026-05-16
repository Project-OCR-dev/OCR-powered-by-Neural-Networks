import asyncio
import numpy as np
import pickle
import os
from convolution import params
from forward import forward
from backward import backward
from utils import load_dataset, one_hot, CLASSES

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.003
LOG_EVERY = 10  # affiche la loss tous les N batches


def _find_latest_checkpoint():
    files = [f for f in os.listdir(".") if f.startswith("params_epoch_") and f.endswith(".pkl")]
    if not files:
        return None, 0
    latest = max(files, key=lambda f: int(f.split("_")[2].split(".")[0]))
    return latest, int(latest.split("_")[2].split(".")[0])


_ckpt, START_EPOCH = _find_latest_checkpoint()
if _ckpt:
    with open(_ckpt, "rb") as f:
        params.update(pickle.load(f))
    print(f"Reprise depuis {_ckpt} (epoch {START_EPOCH})")
elif os.path.exists("params_trained.pkl"):
    with open("params_trained.pkl", "rb") as f:
        params.update(pickle.load(f))
    START_EPOCH = 0
    print("Poids chargés depuis params_trained.pkl")
else:
    START_EPOCH = 0


def step(bx, by):
    return backward(*forward(bx), by, lr=LR)


async def _prefetch(queue, images, labels):
    idx = np.random.permutation(len(images))
    for i in range(0, len(idx), BATCH_SIZE):
        b = idx[i:i+BATCH_SIZE]
        await queue.put((images[b], one_hot(labels[b])))
    await queue.put(None)


async def train_epoch(images, labels, epoch):
    queue = asyncio.Queue(maxsize=8)
    n_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE
    asyncio.create_task(_prefetch(queue, images, labels))
    total, n = 0.0, 0
    while (batch := await queue.get()) is not None:
        loss = await asyncio.to_thread(step, *batch)
        total += loss
        n += 1
        if n % LOG_EVERY == 0:
            print(f"\rEpoch {epoch+1}  batch {n}/{n_batches}  loss={total/n:.4f}", end="", flush=True)
    print()  # saut de ligne après la barre de progression
    return total / n


def predict_image(path):
    from PIL import Image
    img = Image.open(path).convert('L').resize((32, 32))
    x = np.array(img, dtype=np.float32)[np.newaxis] / 255.0
    z3 = forward(x)[0][0]
    print(f"Prédiction : {CLASSES[np.argmax(z3)]}")
    return CLASSES[np.argmax(z3)]


async def main():
    images, labels = load_dataset("train")
    for epoch in range(START_EPOCH, EPOCHS):
        loss = await train_epoch(images, labels, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss finale={loss:.4f}")
        if (epoch + 1) % 5 == 0:
            path = f"params_epoch_{epoch+1}.pkl"
            with open(path, "wb") as f:
                pickle.dump(dict(params), f)
            print(f"  -> Checkpoint : {path}")


if __name__ == "__main__":
    asyncio.run(main())
