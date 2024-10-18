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
    

    class VectorQuantizer2(nn.Module):
        def __init__(
            self, 
            n_embed, 
            embed_dim, 
            beta, remap=None, 
            unknown_index='random',
            sane_index_shape=False,
            legacy=True
        ):
            super().__init__()

            self.n_embed = n_embed
            self.embed_dim = embed_dim
            self.beta = beta
            self.legacy = legacy

            self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

            self.remap = remap
            if self.remap is not None:
                self.register_buffer('used', torch.tensor(np.load(self.remap)))
                self.re_embed = self.used.shape[0]
                self.unknown_index= unknown_index
                if self.unknown_index == 'extra':
                    self.unknown_index = self.re_embed
                    self.re_embed = self.re_embed + 1  
                print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                         f"Using {self.unknown_index} for unknown indices.")
            else:
                self.re_embed = n_embed
        
            self.sane_index_shape = sane_index_shape

        def remap_to_used(self, indices):
            indice_shape = indices.shape
            assert len(indice_shape) > 1
            indices = indices.reshape(indice_shape[0], - 1)
            used = self.used.to(indices)
            match = (indices[:, :, None] == used[None, None, ...]).long()
            new = match.argmax(-1)
            unknown = match.sum(2) < 1
            if self.unknown_index == 'random':
                new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
            else:
                new[unknown] = self.unknown_index
            return new.reshape(indice_shape)

        def unmap_to_all(self, indices):
            indice_shape = indices.shape
            assert len(indice_shape) > 1
            indices = indices.reshape(indice_shape[0], -1)
            used = self.used.to(indices)
            if self.re_embed > self.used.shape[0]:
                indices[indices >= self.used.shape[0]] = 0
            back = torch.gather(used[None, :][indices.shape[0]*[0], :], 1, indices)
            return back.reshape(indice_shape)

        def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
            assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
            assert rescale_logits==False, "Only for interface compatible with Gumbel"
            assert return_logits==False, "Only for interface compatible with Gumbel"

            z = rearrange(z, 'b c h w -> b h w c').contiguous()
            z_flattened = z.view(-1, self.embed_dim)

            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight ** 2, dim=1) - \
                   2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = self.embedding(min_encoding_indices).view(z.shape)
            perplexity = None
            min_encodings = None

            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
            else:
                loss =torch.mean((z_q.detach()  - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

            z_q = z + (z_q - z).detach()

            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

            if self.remap is not None:
                min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
                min_encoding_indices = self.remap_to_used(min_encoding_indices)
                min_encoding_indices = min_encoding_indices.reshape(-1, 1)

            if self.sane_index_shape:
                min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

            return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

        def get_codebook_entry(self, indices, shape):
            if self.remap is not None:
                indices = indices.reshape(shape[0], -1)
                indices = self.unmap_to_all(indices)
                indices = indices.reshape(-1)

            z_q = self.embedding(indices)

            if shape is not None:
                z_q = z_q.view(shape)
                z_q = z_q.permute(0, 3, 1, 2).contiguous()

            return z_q        
