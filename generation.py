import numpy as np
from gestion_donnees import afficher


def generer_image_RBM(RBM, nb_donnees, nb_iter_gibbs, hauteur, largeur, seed=0, commencer_par_h=False):
    rng = np.random.default_rng(seed)
    p = len(RBM.a)
    q = len(RBM.b)
    images_generees = []
    for i in range(nb_donnees):
        if commencer_par_h:
            h = (rng.uniform(size=q) < rng.uniform())
            for j in range(nb_iter_gibbs - 1):
                v = (rng.uniform(size=p) < RBM.sortie_entree_RBM(h)) * 1
                h = (rng.uniform(size=q) < RBM.entree_sortie_RBM(v)) * 1
            v = (rng.uniform(size=p) < RBM.sortie_entree_RBM(h)) * 1
        else:
            v = (rng.uniform(size=p) < rng.uniform())
            for j in range(nb_iter_gibbs):
                h = (rng.uniform(size=q) < RBM.entree_sortie_RBM(v)) * 1
                v = (rng.uniform(size=p) < RBM.sortie_entree_RBM(h)) * 1
        v = v.reshape((hauteur, largeur))
        images_generees.append(v)
        afficher(v)
    return images_generees
