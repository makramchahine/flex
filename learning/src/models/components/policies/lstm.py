from typing import Optional
from omegaconf import DictConfig
import torch.nn as nn

from src.models.components.policies.base import BasePolicy


class LSTMPolicy(BasePolicy):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        input_dim = cfg.channels * int(cfg.spatial_dim / 4) ** 2

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=False  # Batch dim is sequence dim, can be changed later
        )
        self.hidden_state = None
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        out, hidden_state = self.lstm(x, self.hidden_state)
        self.hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        out = self.fc(out)
        return out