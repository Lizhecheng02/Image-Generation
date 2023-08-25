import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class VectorQuantizer(nn.Module):
    """
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x) - sg[e]|| ^ 2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B, C, H, W)
            2. flatten input to (B * H * W, C)

        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        print("z_flattened shape:", z_flattened.shape)

        print(torch.sum(z_flattened ** 2, dim=1, keepdim=True).shape)
        print(torch.sum(self.embedding.weight ** 2, dim=1).shape)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        print("d shape:", d.shape)

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        print("min_codings shape:", min_encodings.shape)
        """
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)：这行代码使用torch.argmin函数在d张量的维度1上找到最小值的索引。具体来说，它返回一个形状为(d.shape[0], 1)的张量，其中d.shape[0]表示d张量的第一个维度的大小。.unsqueeze(1)操作将张量的维度扩展为(d.shape[0], 1)，以便与后续的编码张量对齐。

        min_encoding = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)：这行代码创建一个全零张量min_encoding，形状为(min_encoding_indices.shape[0], self.n_e)，其中min_encoding_indices.shape[0]表示最小值索引张量的第一个维度的大小，self.n_e表示编码张量的第二个维度的大小。该张量将用于存储最小值编码结果。

        min_encoding.scatter_(1, min_encoding_indices, 1)：这行代码使用scatter_函数在min_encoding张量的第1维度上，根据min_encoding_indices中的索引，将对应位置设置为1。具体来说，它将min_encoding_indices中的每个值作为索引，在min_encoding中将相应位置的值设置为1。这实现了最小值编码的效果，其中只有最小值的位置被设置为1，其他位置为0。

        总结起来，这段代码的目的是将输入张量d进行最小值编码，生成一个编码张量min_encoding，其中只有最小值的位置被设置为1，其他位置为0。这在某些情况下可以用于表示最小值的位置或生成离散的最小值编码表示。
        """

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        print("z_q shape:", z_q.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + \
            self.beta * torch.mean((z_q - z.detach()) ** 2)
        """
        z_q.detach() - z和z_q - z.detach()的区别在于是否对z和z_q进行分离操作，以及分离操作对梯度的影响。z_q.detach() - z会保留z_q的梯度，而z_q - z.detach()会保留z_q和z的梯度中只与z_q有关的部分。
        """

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


if __name__ == "__main__":
    vq = VectorQuantizer(49, 64, 0.5)

    z = np.random.random_sample((1, 128, 7, 7))
    z = torch.tensor(z).float()

    vq(z)
