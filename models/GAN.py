import numpy as np
from torch import Tensor
from torch.autograd import Variable

import torch.nn as nn
import torch


class GAN:
    def __init__(self, args, input_dim, output_dim, device=None):
        assert device is not None, "Please specify 'device'!"
    
        # config
        self.device = device
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(input_dim)

    def train(self, epochs, X, adversarial_loss=torch.nn.BCELoss()):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
        
        for epoch in range(epochs):
            
            valid = Variable(Tensor(X.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(X.size(0), 1).fill_(0.0), requires_grad=False)
            
            real_X = X

            z = Tensor(np.random.normal(0, 1, X.shape))
            gen_X = self.generator(z)

            # === Train Generator ===
            # 让判别器把生成数据识别为真的损失
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(self.discriminator(gen_X), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # === Train Discriminator ===
            optimizer_D.zero_grad()
            # 训练判别器识别真样本为真，假样本为假
            real_loss = adversarial_loss(self.discriminator(real_X), valid)
            fake_loss = adversarial_loss(self.discriminator(gen_X.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

    def generate(self, X):
        return self.generator(X)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.model(X)