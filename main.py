from rbm import RBM
from gestion_donnees import lire_alpha_digits, enregistrer_rbm, charger_rbm
from generation import generer_image_RBM
from hyperparametres import *

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
GRILLE_CONVERSION = {str(i): i for i in range(10)}
for idx in range(len(ALPHABET)):
    GRILLE_CONVERSION[ALPHABET[idx]] = 10 + idx


def convertir_caracteres_en_indices(caracteres):
    return [GRILLE_CONVERSION[caractere] for caractere in caracteres]


if __name__ == '__main__':
    indices_entrainement = convertir_caracteres_en_indices(CARACTERES_ENTRAINEMENT)
    donnees_entrainement, hauteur, largeur = lire_alpha_digits(
        'data/binaryalphadigs.mat',
        caracteres=indices_entrainement
    )
    try:
        machine = charger_rbm(specs=SPECS_ENTRAINEMENT)
    except:
        machine = RBM(donnees_entrainement.shape[1], Q)
        machine.train_RBM(donnees_entrainement, EPSILON, NB_EPOCHS, TAILLE_BATCH)
        enregistrer_rbm(machine, specs=SPECS_ENTRAINEMENT)
    generer_image_RBM(machine, NB_DONNEES_GENEREES, NB_ITER_GIBBS,
                      hauteur, largeur, commencer_par_h=COMMENCER_PAR_H_GIBBS)