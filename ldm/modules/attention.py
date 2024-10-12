import math
import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat


# FeedForward
class GEGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
    
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        hidden_dim = int(in_dim * mult)
        if out_dim is None:
            out_dim = in_dim
        proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU()
        ) if not glu else GEGLU(in_dim, hidden_dim)

        self.net = nn.Sequential(
            proj,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, head_dim=32, heads=4):
        super().__init__()
        self.heads = heads
        hidden_dim = head_dim * heads
        self.qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.out = nn.Conv2d(hidden_dim, dim, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv =3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)     # [b, heads, c, c]
        out = torch.einsum('bhde,bhdn->bhen', context, q)   # [b, heads, c, hw]
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.out(out)

    
class SpatialSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj = torch.nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        sim = einsum('bij,bjk->bik', q, k)   # [b, hw, hw]

        sim = sim * (int(c) ** (-0.5))
        attn = F.softmax(sim, dim=2)

        v = rearrange(v, 'b c h w -> b c (h w)')    # [b, c, hw]
        attn = rearrange(attn, 'b i j -> b j i')    # [b, hw, hw]
        out = einsum('bij,bjk->bik', v, attn)    # [b, c, hw]
        out = rearrange(out, 'b c (h w) -> b c h w', h=h)
        out = self.proj(out)

        return x + out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, head_dim=64, heads=8, dropout=0.):
        super().__init__()
        hidden_dim = head_dim * heads
        if context_dim is None:
            context_dim = query_dim
        
        self.scale = head_dim ** (-0.5)
        self.heads = heads

        self.q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.q(x)
        if context is None:
            context = x
        k = self.k(context)
        v = self.v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale    # [bh, n, n]

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        head_dim, 
        heads, 
        dropout=0., 
        context_dim=None, 
        gated_ff=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, head_dim=head_dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, head_dim=head_dim,
                                    heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data
    """
    def __init__(
        self,
        dim,
        head_dim,
        heads,
        depth=1,
        dropout=0.,
        context_dim=None
    ):
        super().__init__()

        self.dim = dim
        hidden_dim = heads * head_dim

        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.transformers = nn.ModuleList(
            [
                BasicTransformerBlock(hidden_dim, head_dim, heads,
                                      dropout=dropout, context_dim=context_dim)
                                      for d in range(depth)
            ]
        )
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0)

        for parameter in self.proj_out.parameters():
            parameter.detach().zero_()

    def forward(self, x, context=None):
        b, c, h, w = x.shape

        h = self.norm(x)
        h = self.proj_in(h)
        h = rearrange(h, 'b c h w -> b (h w) c')
        for transformer in self.transformers:
            h = transformer(h, context=context)
        h = rearrange(h, 'b (h w) c -> b c h w', h=h, w=w)
        h = self.proj_out(h)
        return h + x

