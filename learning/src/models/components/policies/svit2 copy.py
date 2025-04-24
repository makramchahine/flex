from typing import Optional
from omegaconf import DictConfig # type: ignore
import torch
from torch import nn

from .base import BasePolicy
from vit_pytorch import SimpleViT # type: ignore

dir_vocab = {'left': 0, 'right': 1, 'above': 2, 'below': 3, 'towards': 4}
dir_dim = 5

class SViTPolicy(BasePolicy):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.net = SimpleViT(
            image_size = tuple(cfg.image_size),
            patch_size = tuple(cfg.patch_size),
            num_classes = cfg.num_classes,
            dim = cfg.dim,
            depth = cfg.depth,
            heads = cfg.heads,
            mlp_dim = cfg.mlp_dim,
            channels = cfg.channels + len(dir_vocab),
            dim_head = cfg.dim_head,
        )

    def forward(self, x, dir_text: list[str]):
        device = x.device
        dir_idx = torch.tensor([dir_vocab[dir_] for dir_ in dir_text], device=device)
        dir_emb = torch.nn.functional.one_hot(dir_idx, num_classes=len(dir_vocab)).to(dtype=x.dtype)
        dir_emb = dir_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, dir_emb), dim=1)

        return self.net(x)

