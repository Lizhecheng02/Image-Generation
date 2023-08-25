import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResidualStack


class Encoder(nn.Module):

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel -
                      1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_block(x)


if __name__ == "__main__":
    x = np.random.random_sample((50, 3, 224, 224))
    x = torch.tensor(x).float()

    encoder = Encoder(3, 128, 4, 96)
    print(encoder)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
