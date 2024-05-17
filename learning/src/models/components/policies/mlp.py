from typing import List
from omegaconf import DictConfig
import torch.nn as nn

from src.models.components.net_utils import get_norm, get_activation
from src.models.components.policies.base import BasePolicy


class MLPPolicy(BasePolicy):
    def __init__(
        self,
        reduction_cfg: DictConfig,
        fc_cfg: List[int],
        fc_bias: bool,
        norm_cfg: DictConfig,
        act_cfg: DictConfig,
        dropout_p: float,
    ):
        super().__init__()
        
        reduction = []
        if reduction_cfg.type in [None, "None"]:
            pass
        elif reduction_cfg.type == "avgpool":
            reduction.append(nn.AdaptiveAvgPool2d(
                output_size=reduction_cfg.get("output_size", (1, 1)),
            ))
        else:
            raise ValueError(f"Unrecognized reduction {reduction_cfg.type}")
        reduction.append(nn.Flatten())
        self.reduction = nn.Sequential(*reduction)
        
        net = []
        for i in range(1, len(fc_cfg)):
            # fc
            in_features, out_features = fc_cfg[i-1], fc_cfg[i]
            net.append(nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=fc_bias,
            ))
            
            is_last_layer = i == (len(fc_cfg) - 1)
            if not is_last_layer: # NOTE: last layer should be linear
                # norm
                norm = get_norm(norm_cfg, out_features)
                if norm is not None:
                    net.append(norm)
                
                # activation
                act = get_activation(act_cfg, out_features)
                if act is not None:
                    net.append(act)

                # dropout
                if dropout_p > 0:
                    net.append(nn.Dropout(p=dropout_p))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        z = self.reduction(x)
        out = self.net(z)

        return out
