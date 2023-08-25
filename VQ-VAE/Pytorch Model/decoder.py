import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual import ResidualStack


class Decoder(nn.Module):

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_block(x)


if __name__ == "__main__":
    x = np.random.random_sample((50, 3, 224, 224))
    x = torch.tensor(x).float()

    decoder = Decoder(3, 128, 2, 64)
    print(decoder)
    decoder_out = decoder(x)
    print('Decoder out shape:', decoder_out.shape)
