from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(320, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.float()
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(100, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 320)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x[x > 0] = 1.
        x[x <= 0] = 0.
        return x
