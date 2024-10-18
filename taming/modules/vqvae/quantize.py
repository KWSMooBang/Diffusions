import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn
from torch import einsum


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, beta):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel)
        z = z.premute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_embed).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients 
        # value: z_q, gradients: z
        z_q = z + (z_q - z).detach()

        embed_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(embed_mean * torch.log(embed_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
    
    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_embed).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

