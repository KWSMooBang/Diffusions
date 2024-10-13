import torch
import torch.nn.functional as F

from collections import namedtuple
from functools import partial
from inspect import isfunction
from einops import rearrange, repeat, reduce
from torch import nn, einsum
from attention import FeedForward, 


DEFAULT_HEAD_DIM = 64

Intermediates = namedtuple(
    'Intermediates', [
        'pre_softmax_attn',
        'post_softmax_attn'
    ]
)

LayerIntermediates = namedtuple(
    'Intermediates', [
        'hiddens',
        'attn_intermediates'
    ]
)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]

    
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim , 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = einsum('i,j ->ij', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]
    

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, *kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)
    

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * (self.g)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g
    

class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )
        
        return gated_output.reshape_as(x)
    

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=DEFAULT_HEAD_DIM,
        heads=8,
        causal=False,
        mask=None,
        talking_heads=False,
        sparse_topk=None,
        use_entmax15=False,
        num_mem_kv=0,
        dropout=0.,
        on_attn=False
    ):
        super().__init__()
        if use_entmax15:
            raise NotImplementedError("Check out entmax activation instead of softmax activation")
        self.scale = head_dim ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        hidden_dim = head_dim * heads

        self.q = nn.Linear(dim, hidden_dim, bias=False)
        self.k = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        self.sparse_topk = sparse_topk

        self.attn_fn = F.softmax

        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, head_dim))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, head_dim))

        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        rel_pos=None,
        sinusoidal_emb=None,
        prev_attn=None,
        mem=None
    ):
        b, n, _ = x.shape
        h = self.heads
        talking_heads = self.talking_heads
        device = x.device

        kv_input = context if context is not None else x

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if mem is not None:
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)
        
        if sinusoidal_emb is not None:
            offset = k_input.shape[-2] -  q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.q(q_input)
        k = self.k(k_input)
        v = self.v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        input_mask = None
        q_mask = mask if mask is not None else torch.ones((b, n), device=device).bool()
        k_mask = context_mask if context is not None else q_mask
        k_mask = k_mask if k_mask is not None else torch.ones((b, k.shape[-2]), device=device).bool()
        q_mask = rearrange(q_mask, 'b i -> b () i ()')
        k_mask = rearrange(k_mask, ' b j -> b () () j')
        input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if input_mask is not None:
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)
            
        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(attn.dtype).max

        if prev_attn is not None:
            attn = attn + prev_attn

        pre_softmax_attn = attn

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.pre_softmax_proj).contiguous()

        if rel_pos is not None:
            attn.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = attn.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            attn.masked_fill_(mask, mask_value)
            del mask

        if self.sparse_top is not None and self.sparse_topk < attn.shape[-1]:
            top, _ = attn.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(attn)
            mask = attn < vk
            attn.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(attn, dim=-1)
        post_softmax_attn = attn

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        return self.out(out), intermediates


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        position_infused_attn=False,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        head_dim = attn_kwargs.get('head_dim', DEFAULT_HEAD_DIM)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None
        self.rotary_pos_emb = None

        assert rel_pos_num_buckets <= rel_pos_max_distance, "number of relative position buckets must be less than the relative position max distance"
        self.rel_pos = None

        self.pre_norm = pre_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim) 
        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f', ) + default_block

        if custom_layers is not None:
            layer_types = custom_layers
        elif par_ratio is not None:
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(lambda x: x == 'f', default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, "default block is too large for par_ratio"
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif sandwich_coef is not None:
            assert sandwich_coef > 0 and sandwich_coef <= depth, "sandwich coefficient should be less than the depth"
            layer_types = ('a',) * sandwich_coef + default_block * (depth -sandwich_coef)
        else:
            layer_types = default_block * depth

        self.layers_types = layer_types
        self.num_attn_layers = len(list(filter(lambda x: x == 'a', layer_types)))

        for layer_type in self.layers_types:
            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"Invalid Layer Type: {layer_type}")
            
            if isinstance(layer, Attention) and branch_fn is not None:
                layer = branch_fn(layer)

            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))
    
    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        mems=None,
        return_hiddens=False
    ):
        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if mems is not None else [None] * self.num_attn_layers

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask, sinusoidal_emb=self.pia_pos_emb, 
                                   rel_pos=self.rel_pos, prev_attn=prev_attn, mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask,
                                   prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0,
        emb_dropout=0.,
        num_memory_tokens=None,
        tie_embedding=False,
        use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = emb_dim if emb_dim is not None else dim

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.num_tokens = num_tokens

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (use_pos_emb and
                    not attn_layers.has_pos_emb) else 0
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()
        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        num_memory_tokens = num_memory_tokens if num_memory_tokens is not None else 0
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.num_memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        **kwargs
    ):
        b, n = x.shape
        device = x.device
        num_mem = self.num_memory_tokens

        x = self.token(x)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.num_memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            if mask is not None:
                mask = F.pad(mask, (num_mem, 0), value=True)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if mems is not None else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems
    
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps
        
        return out
            