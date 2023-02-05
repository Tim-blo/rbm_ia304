import torch
from torch.utils.data import DataLoader
from gan import Generator, Discriminator
from hyperparametres import *
from main import convertir_caracteres_en_indices
from gestion_donnees import afficher
from pytorch_dataset import AlphaDigitsDataset


class LossMeter(object):
    def __init__(self):
        self.number = 0.
        self.sum = 0.
        self.avg = 0.
        self.last_ratio = 0.
        self.first = 1.

    def update(self, value, first=False):
        self.number += 1.
        self.sum += value
        self.avg = self.sum / self.number
        if first:
            self.first = value
        self.last_ratio = value / (self.first + 1e-8)

    def reset(self):
        self.number, self.sum, self.avg = 0., 0., 0.


def train_D_batch(batch, G, D, D_optimiser):
    D_optimiser.zero_grad()
    batch_size = batch.shape[0]

    # Train the discriminator on the true example
    d = D(batch)
    # the dataset examples are all real examples (1) or a high score for Wasserstein
    loss = -(torch.mean(d))
    # Train the discriminator on generated examples
    g = G(torch.randn(batch_size, 100))
    d = D(g)
    # we want D to say that the examples are fake
    loss = loss + (torch.mean(d))

    loss.backward()
    D_optimiser.step()

    for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)

    return loss.item() / 2  # over 2 because D is trained on twice as many examples without proper averaging


def train_G_batch(batch, G, D, G_optimiser):
    G_optimiser.zero_grad()
    batch_size = batch.shape[0]  # we expect a batch of [batch_size, sequence_length, input_size]

    g = G(torch.randn(batch_size, 100))
    d = D(g)
    # We want G to fool D
    loss = -(torch.mean(d))

    loss.backward()
    G_optimiser.step()
    return loss.item()


def train():
    indices_entrainement = convertir_caracteres_en_indices(CARACTERES_ENTRAINEMENT)
    train_dataset = AlphaDigitsDataset('data/binaryalphadigs.mat', indices_entrainement)
    train_loader = DataLoader(train_dataset, batch_size=TAILLE_BATCH,
                              shuffle=True)

    G = Generator().train()

    D = Discriminator().train()

    G_optimiser = torch.optim.Adam(G.parameters(), lr=EPSILON)
    G_trainer = train_G_batch
    D_optimiser = torch.optim.Adam(D.parameters(), lr=EPSILON)

    D_loss, G_loss = LossMeter(), LossMeter()

    for epoch in range(1, NB_EPOCHS + 1):
        D_loss.reset()
        G_loss.reset()

        for idx, batch in enumerate(train_loader):

            # if D is too good we skip this batch for it
            freeze_D = D_loss.last_ratio < 0.7 * G_loss.last_ratio and idx != 0
            # if G is too good we skip this batch for it
            freeze_G = G_loss.last_ratio < 0.7 * D_loss.last_ratio and idx != 0
            x = batch

            if not freeze_D:
                D_loss_batch = train_D_batch(x, G, D, D_optimiser)
                D_loss.update(D_loss_batch, first=idx == 0)

            if not freeze_G and idx % 5 == 0:
                G_loss_batch = G_trainer(x, G, D, G_optimiser)
                G_loss.update(G_loss_batch, first=idx == 0)

        print('[{}/{}]\tD: {:.5f}\tG: {:.5f}'.format(epoch, NB_EPOCHS,
                                                     D_loss.avg if D_loss.avg != 0 else D_loss.last_ratio,
                                                     G_loss.avg if G_loss.avg != 0 else G_loss.last_ratio))

    for i in range(NB_DONNEES_GENEREES):
        afficher(G(torch.randn(1, 100))[0].detach().numpy().reshape(20,16))


if __name__ == '__main__':
    train()

