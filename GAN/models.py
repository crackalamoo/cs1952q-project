import torch.nn as nn
from constants import*

dis_h1 = 256
dis_h2 = 256

gen_h1 = 1000
gen_h2 = 1000


class Discriminator(nn.Module):
    def __init__(self, img_size, name="dis_model"):
        super(Discriminator, self).__init__()
        self.name = name
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size*img_size, dis_h1),
            nn.LeakyReLU(0.01),
            nn.Linear(dis_h1, dis_h2),
            nn.LeakyReLU(0.01),
            nn.Linear(dis_h2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, img_size, z_dim, name="gen_model"):
        super(Generator, self).__init__()
        self.name = name
        self.model = nn.Sequential(
            nn.Linear(z_dim, gen_h1),
            nn.LeakyReLU(0.01),
            nn.Linear(gen_h1, gen_h2),
            nn.LeakyReLU(0.01),
            nn.Linear(gen_h2, img_size*img_size),
            nn.Tanh(),
            nn.Unflatten(1, (img_size, img_size, channels))
        )

    def forward(self, x):
        return self.model(x)
