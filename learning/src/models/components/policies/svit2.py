from typing import Optional
from omegaconf import DictConfig # type: ignore
import torch
from torch import nn

from .base import BasePolicy
from vit_pytorch import SimpleViT # type: ignore

dir_vocab = {'left': [1, -1, 0], 'right': [1, 1, 0], 'above': [1, 0, 1], 'below': [1, 0, -1], 'towards': [1, 0, 0]}
dir_dim = 3

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
            channels = cfg.channels + dir_dim,
            dim_head = cfg.dim_head,
        )
        self.save_rep_hooked = False

    def forward(self, x, dir_text: list[str]):
        device = x.device
        dir_enc = torch.tensor([dir_vocab[dir_] for dir_ in dir_text], device=device, dtype=torch.float32)
        dir_enc = dir_enc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, dir_enc), dim=1)

        if not self.save_rep_hooked:
            def hook(module, input, output):
                self.save_rep(output)
            self.net.transformer.register_forward_hook(hook)
            self.save_rep_hooked = True

        return self.net(x)

