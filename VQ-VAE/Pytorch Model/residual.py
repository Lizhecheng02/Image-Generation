import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([
            ResidualLayer(in_dim, h_dim, res_h_dim)
        ] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    x = np.random.random_sample((50, 3, 224, 224))
    """
        (batch, channel, ..., ...)
    """
    x = torch.tensor(x).float()

    res = ResidualLayer(3, 3, 32)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)

    res_stack = ResidualStack(3, 3, 32, 4)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
