from convolution import convolution, maxpooling, params
from utils import relu, softmax, cross_entropy, one_hot, load_dataset_MNIST
from forward import forward
from PIL import Image
import numpy as np
import os
import pickle


if os.path.exists("params_trained.pkl"):
    with open("params_trained.pkl", "rb") as f:
        params_charges = pickle.load(f)
    for key in params_charges:
        params[key] = params_charges[key]
    print("Poids chargés")
else:
    print("Pas de poids sauvegardés, démarrage entrainement")


file_path = "dataset/"
total_loss = 0


def global_loss_function(L):
    global total_loss
    total_loss += L


if __name__ == "__main__":
    images, labels, mapping = load_dataset_MNIST("train")
    total_iterations = 0

    for epoch in range(5):
        total_loss = 0
        shuffle = np.random.permutation(len(images))

        for j in range(len(images)):
            image = images[shuffle[j]]
            label = labels[shuffle[j]]

            x3, x1, x2, flat, entree_conv1, sortie_conv1, entree_conv2, sortie_conv2 = forward(image)
            proba = softmax(x3)
            y_one_hot = one_hot(label)
            loss = cross_entropy(proba, y_one_hot)
            global_loss_function(loss)
            backward(proba, y_one_hot, x1, x2, flat, entree_conv1, entree_conv2, sortie_conv1, sortie_conv2)

            total_iterations += 1
            if total_iterations % 50000 == 0:
                with open(f"params_iter_{total_iterations}.pkl", "wb") as f:
                    pickle.dump(params, f)
                print(f"Sauvegarde intermédiaire à {total_iterations} itérations")

        with open(f"params_epoch_{epoch+1}.pkl", "wb") as f:
            pickle.dump(params, f)
        print(f"Epoch {epoch+1} \n Loss moyenne : {total_loss/len(images):.4f} \n sauvegardé")
