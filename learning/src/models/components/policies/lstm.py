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
        # Linear layer to reduce number of channels
        self.linear = nn.Linear(cfg.channels, cfg.hidden_channels)
        # length of vector for a single element in the input sequence 
        input_dim = (cfg.spatial_dim ** 2) * cfg.hidden_channels
        # buffer to store sequence
        # self.buffer = deque(maxlen=cfg.seq_length)
        self.seq_count = 0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=False
        )
        self.hidden_state = None
        # final layer to return num actions
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, x:torch.Tensor, reset_hidden=False):
        # reseting hidden state by fixed sequence length if needed
        if self.cfg.reset_count is not None:
            self.seq_count = (self.seq_count + 1) % self.cfg.reset_count
            if self.seq_count == 0: reset_hidden = True

        # reshaping x for the linear layer and then policy
        b = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(-1, self.cfg.channels)
        x = self.linear(x)
        x = x.reshape(b, -1)

        # preparing sequence of features
        # self.buffer.append(x)
        # seq = torch.stack(list(self.buffer), dim=1)

        # applying policy on the sequence of features
        out, hidden_state = self.lstm(x, self.hidden_state)
        self.hidden_state = None if reset_hidden else (hidden_state[0].detach(), hidden_state[1].detach())
        out = self.fc(out)
        # self.buffer = deque([t.detach() for t in self.buffer], maxlen=self.cfg.seq_length)
        return out