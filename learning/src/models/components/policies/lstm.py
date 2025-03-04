from typing import Optional
from omegaconf import DictConfig
import torch
import torch.nn as nn
from collections import deque

from src.models.components.policies.base import BasePolicy


class LSTMPolicy(BasePolicy):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        
        self.cfg = cfg
        self.single_step = cfg.single_step

        self.hidden_state = None

        # Linear layer to reduce number of channels
        self.linear = nn.Linear(cfg.channels, cfg.reduced_dim)
        # length of vector for a single element in the input sequence 
        input_dim = (cfg.spatial_dim ** 2) * cfg.reduced_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )
        # final layer to return num actions
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.fc_stop = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        
        seq_len = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(-1, self.cfg.channels)
        x = self.linear(x)
        x = x.reshape(seq_len, -1)

        if self.single_step:
            # Single-step mode: process one time step at a time (one input at a time)
            out, hidden_state = self.lstm(x.unsqueeze(0), self.hidden_state)
            self.hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            out = out.squeeze(0)
        else:
            self.hidden_state = None
            # this should reset the hidden state but we reset in case
            out, _ = self.lstm(x)

        out_action = self.fc(out)
        out_stop = self.fc_stop(out)
        # out_stop = torch.softmax(out_stop, dim=-1)

        return out_action, out_stop