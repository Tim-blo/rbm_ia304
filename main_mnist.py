from rbm import RBM
from gestion_donnees import enregistrer_rbm, charger_rbm, lire_mnist
from generation import generer_image_RBM
from hyperparametres import *

if __name__ == '__main__':
    donnees_entrainement, hauteur, largeur = lire_mnist(
        caracteres=MNIST_ENTRAINEMENT
    )
    try:
        machine = charger_rbm(specs=SPECS_ENTRAINEMENT_MNIST)
    except:
        machine = RBM(donnees_entrainement.shape[1], Q)
        machine.train_RBM(donnees_entrainement, EPSILON, NB_EPOCHS, TAILLE_BATCH)
        enregistrer_rbm(machine, specs=SPECS_ENTRAINEMENT_MNIST)
    generer_image_RBM(machine, NB_DONNEES_GENEREES, NB_ITER_GIBBS,
                      hauteur, largeur, commencer_par_h=COMMENCER_PAR_H_GIBBS)
