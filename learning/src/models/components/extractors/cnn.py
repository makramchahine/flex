from typing import List
from omegaconf import DictConfig
import torch.nn as nn

from src.models.components.net_utils import get_norm, get_activation
from src.models.components.extractors.base import BaseExtractor


class CNNExtractor(BaseExtractor):
    def __init__(
        self,
        conv_cfg: List[List[int]],
        norm_cfg: DictConfig,
        act_cfg: DictConfig,
    ):
        super().__init__()
        
        net = []
        for conv_cfg_i in conv_cfg:
            # conv
            in_channels, out_channels, kernel_size, stride, padding = conv_cfg_i
            net.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ))

            # norm
            norm = get_norm(norm_cfg, out_channels)
            if norm is not None:
                net.append(norm)

            # activation
            act = get_activation(act_cfg, out_channels)
            if act is not None:
                net.append(act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        super().forward(x)
        inp = x["image"]

        return self.net(inp)

    @property
    def modalities(self):
        return ["image", "image_raw"]
