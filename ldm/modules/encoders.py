import torch


from functools import partial
from einops import rearrange, repeat
from torch import nn
from torchvision.transforms import v2, InterpolationMode
from transformers import CLIPTokenizer, CLIPTextModel
from transformer import Encoder, TransformerWrapper


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class TransformerEmbedder(AbstractEncoder):
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device='cuda'):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        return z
    
    def encode(self, x):
        return self(x)
    

class BERTTokenizer(AbstractEncoder):
    def __init__(self, vq_interface=True, max_length=77, device='cuda'):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                        return_overflowing_tokens=False, padding='max_length',
                                        return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(self.device)
        return tokens
    
    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text
    
class BERTEmbedder(AbstractEncoder):
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device='cuda', use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tokenizer = use_tokenizer
        if self.use_tokenizer:
            self.tokenizer = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_mem_len=,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tokenizer:
            tokens = self.tokenizer(text).to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z
    
    def encode(self, text):
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages=1,
        method='bilinear',
        multiplier=0.5,
        in_dim=3,
        out_dim=None,
        bias=False
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_dim is not None
        if self.remap_output:
            print(f"Spatial Rescaler mapping from {in_dim} to {out_dim} dimension after resizing.")
            self.dimension_mapper = nn.Conv2d(in_dim, out_dim, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.dimension_mapper(x)

        return x
    
    def encode(self, x):
        return self(x)
    

class FrozenCLIPEmbedder(AbstractEncoder):
    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda', max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                        return_length=True, return_overflowing_tokens=False,
                                        padding='max_length', return_tensors='pt')
        tokens = batch_encoding['input_ids'].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z
    
    def encoder(self, text):
        return self(text)
    

class FrozenCLIPTextEmbedder(nn.Module):
    def __init__(self, version='ViT-L/14', device='cuda', max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device='cpu')
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z
    
    def encoder(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    def __init__(
        self,
        model,
        jit=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)
        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

        self.resize = v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=self.antialias)
        self.normalize = v2.Normalize(mean=self.mean, std=self.std)

    def preprocess(self, x):
        x = self.resize(x)
        x = (x + 1.) / 2.
        x = self.normalize(x)
        return x

    def forward(self, x):
        return self.model.encode_image(self.preprocess(x))