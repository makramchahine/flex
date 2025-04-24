from typing import Optional
from omegaconf import DictConfig # type: ignore
import torch
from torch import nn
from einops.layers.torch import Rearrange

from .base import BasePolicy
from .svit import posemb_sincos_2d, Transformer

dir_vocab = {'left': [1, -1, 0], 'right': [1, 1, 0], 'above': [1, 0, 1], 'below': [1, 0, -1], 'towards': [1, 0, 0]}

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

        self.dir_proj = nn.Linear(len(dir_vocab['above']), cfg.dim)
        self.transformer = Transformer(cfg.dim, cfg.depth, cfg.heads, cfg.dim_head, cfg.mlp_dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(cfg.dim, cfg.num_classes)

    def forward(self, img, dir_text: Optional[list[str]] = None):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        if dir_text is not None:
            dir_enc = torch.tensor([dir_vocab[dir_] for dir_ in dir_text], device=device, dtype=torch.float32)
            dir_emb = self.dir_proj(dir_enc).unsqueeze(1)
            x = torch.cat((x, dir_emb), dim=1)

        x = self.transformer(x)
        rep = x.detach().clone()
        if dir_text is not None:
            rep = torch.cat((rep, dir_emb), dim=1)
        self.save_rep(rep)

        x = x.mean(dim = 1)
        x = self.to_latent(x)
        x = self.linear_head(x)
        return x

