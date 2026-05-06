import asyncio
import numpy as np
import pickle
import os
from convolution import params
from forward import forward
from backward import backward
from utils import load_dataset, one_hot, CLASSES

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

if os.path.exists("params_trained.pkl"):
    with open("params_trained.pkl", "rb") as f:
        params.update(pickle.load(f))
    print("Poids chargés")


def step(bx, by):
    return backward(*forward(bx), by, lr=LR)


async def _prefetch(queue, images, labels):
    idx = np.random.permutation(len(images))
    for i in range(0, len(idx), BATCH_SIZE):
        b = idx[i:i+BATCH_SIZE]
        await queue.put((images[b], one_hot(labels[b])))
    await queue.put(None)


async def train_epoch(images, labels):
    queue = asyncio.Queue(maxsize=8)
    asyncio.create_task(_prefetch(queue, images, labels))
    total, n = 0.0, 0
    while (batch := await queue.get()) is not None:
        loss = await asyncio.to_thread(step, *batch)
        total += loss
        n += 1
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
    for epoch in range(EPOCHS):
        loss = await train_epoch(images, labels)
        print(f"Epoch {epoch+1}  loss={loss:.4f}")
        with open(f"params_epoch_{epoch+1}.pkl", "wb") as f:
            pickle.dump(dict(params), f)


if __name__ == "__main__":
    asyncio.run(main())
