from typing import List
from omegaconf import DictConfig
import torch.nn as nn

from src.models.components.net_utils import get_norm, get_activation
from src.models.components.policies.base import BasePolicy


class ConvPolicy(BasePolicy):
    def __init__(
            self,
            fc_cfg: List[int],
            fc_bias: bool,
            norm_cfg: DictConfig,
            act_cfg: DictConfig,
            dropout_p: float,
    ):
        super().__init__()

        # Fully connected network configuration
        net = []
        for i in range(1, len(fc_cfg)-1):
            # Convolutional layer (1x1)
            in_channels, out_channels = fc_cfg[i - 1], fc_cfg[i]
            net.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=fc_bias,
            ))

            # Normalization
            norm = get_norm(norm_cfg, out_channels)
            if norm is not None:
                net.append(norm)

            # Activation
            act = get_activation(act_cfg, out_channels)
            if act is not None:
                net.append(act)

            # Dropout
            if dropout_p > 0:
                net.append(nn.Dropout(p=dropout_p))

        # last layer should be linear
        net.append(nn.Flatten())
        net.append(nn.Linear(
            in_features=16*16 * fc_cfg[-2],
            out_features=fc_cfg[-1],
            bias=fc_bias,
        ))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out
