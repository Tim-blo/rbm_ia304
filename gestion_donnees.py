import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle


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
