import torch
import numpy as np
from torch.utils.data import Dataset
from gestion_donnees import lire_alpha_digits


class AlphaDigitsDataset(Dataset):
    def __init__(self, chemin, caracteres):
        self.donnees = lire_alpha_digits(chemin, caracteres)[0]

    def __len__(self):
        return len(self.donnees)

    def __getitem__(self, idx):
        data = torch.tensor(self.donnees[idx])
        return data
