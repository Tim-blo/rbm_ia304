import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.datasets import load_digits
import pandas as pd


def lire_mnist(caracteres=[]):
    if not (all(x in range(0, 10) for x in caracteres)):
        print("MNIST digits range from 0 to 9.\n")
        return None
    mnist = load_digits()
    df = pd.DataFrame(mnist.data)

    # select data for target characters
    indices = [idx for idx, target in enumerate(mnist.target) if target in caracteres]
    df = df.iloc[indices]

    # convert grayscale to binary
    df[:] = np.where(df < 8, 0, 1)

    return np.array(df), 8, 8


def afficher_mnist(chiffre):
    imgs, h, l = lire_mnist([chiffre])
    for i in range(5):
        v = imgs[i].reshape((h, l))
        afficher(v)


def lire_alpha_digits(chemin, caracteres=[]):
    full_data_json = scipy.io.loadmat(chemin)
    donnees_4d = full_data_json['dat'][caracteres]
    donnees_4d = donnees_4d.reshape(np.prod(donnees_4d.shape))
    hauteur, largeur = donnees_4d[0].shape
    donnees_3d = [donnee_4d.reshape(np.prod(donnee_4d.shape)) for donnee_4d in donnees_4d]
    return np.array(donnees_3d), hauteur, largeur


def afficher(image):
    plt.imshow(image)
    plt.show()


def enregistrer_rbm(rbm, specs):
    file_name = 'models/trained_rbm'
    for key, value in specs.items():
        file_name += '_' + key + '_' + str(value)
    pickle.dump(rbm, open(file_name, "wb"))


def charger_rbm(specs):
    file_name = 'models/trained_rbm'
    for key, value in specs.items():
        file_name += '_' + key + '_' + str(value)
    return pickle.load(open(file_name, "rb"))
