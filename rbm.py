import numpy as np
import matplotlib.pyplot as plt
from hyperparametres import PREMIERE_EPOCH_PLOT


class RBM:
    def __init__(self, p, q, seed=0):
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = self.rng.normal(scale=0.1, size=(p, q))

    def entree_sortie_RBM(self, X):
        return 1 / (1 + np.exp(-(X @ self.W + self.b)))

    def sortie_entree_RBM(self, H):
        return 1 / (1 + np.exp(-(H @ self.W.T + self.a)))

    def train_RBM(self, X, epsilon, nb_epochs, taille_batch):
        n_echantillons = X.shape[0]
        rec_errors = []
        for epoch in range(nb_epochs):
            self.rng.shuffle(X, axis=0)
            for batch in range(0, n_echantillons, taille_batch):
                X_batch = X[batch:min(batch + taille_batch, n_echantillons)]
                taille_batch_actuel = X_batch.shape[0]
                v_0 = X_batch
                p_h_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = (self.rng.uniform(size=p_h_v_0.shape) < p_h_v_0) * 1
                p_v_h_0 = self.sortie_entree_RBM(h_0)
                v_1 = (self.rng.uniform(size=p_v_h_0.shape) < p_v_h_0) * 1
                p_h_v_1 = self.entree_sortie_RBM(v_1)
                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1
                self.W += epsilon / taille_batch_actuel * grad_w
                self.a += epsilon / taille_batch_actuel * grad_a
                self.b += epsilon / taille_batch_actuel * grad_b
            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            error = np.sum((X - X_rec) ** 2) / n_echantillons
            rec_errors.append(error)
            print("Reconstruction error at epoch", epoch, "is", error)
        plt.plot(np.arange(PREMIERE_EPOCH_PLOT, nb_epochs), rec_errors[PREMIERE_EPOCH_PLOT:])
        plt.title('Reconstruction error', fontsize=14)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Reconstruction error', fontsize=10)
        plt.show()
