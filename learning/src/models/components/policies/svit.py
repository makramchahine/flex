from typing import Optional
from omegaconf import DictConfig # type: ignore
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from .base import BasePolicy

dir_vocab = {'left': 0, 'right': 1, 'above': 2, 'below': 3, 'towards': 4}

# helpers

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SViTPolicy(BasePolicy):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        image_height, image_width = cfg.image_size
        patch_height, patch_width = cfg.patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = cfg.channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, cfg.dim),
            nn.LayerNorm(cfg.dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = cfg.dim,
        )

        self.dir_proj = nn.Linear(len(dir_vocab), cfg.dim)
        self.transformer = Transformer(cfg.dim, cfg.depth, cfg.heads, cfg.dim_head, cfg.mlp_dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(cfg.dim, cfg.num_classes)

    def forward(self, img, dir_text: Optional[list[str]] = None):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        if dir_text is not None:
            dir_idx = torch.tensor([dir_vocab[dir_] for dir_ in dir_text], device=device)
            dir_ohe = torch.nn.functional.one_hot(dir_idx, num_classes=len(dir_vocab)).to(dtype=x.dtype)
            dir_emb = self.dir_proj(dir_ohe).unsqueeze(1)
            x = torch.cat((x, dir_emb), dim=1)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        x = self.linear_head(x)
        return x

