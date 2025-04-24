import torch
import torch.nn as nn
from .svit import Transformer
from .base import BaseStopPolicy

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, scale=True):
        super().__init__()
        context_dim = context_dim or query_dim
        self.scale = scale
        self.temperature = query_dim ** 0.5 if scale else 1.0
        self.to_q = nn.Identity()
        self.to_k = nn.Identity()
        self.to_v = nn.Identity()
        if context_dim is not None:
            self.to_k = nn.Linear(context_dim, query_dim, bias=False)
            self.to_v = nn.Linear(context_dim, query_dim, bias=False)

    def forward(self, query, context):
        q = self.to_q(query)                  
        k = self.to_k(context)                 
        v = self.to_v(context)                    

        attn_weights = torch.softmax((q @ k.transpose(-1, -2)) / self.temperature, dim=-1)  
        out = attn_weights @ v                  
        return out


class StopFlaggeSimpleConv(BaseStopPolicy):
    def __init__(self, in_dim = 128, hidden_dim =32, cfg = None, *args, **kwargs):
        super().__init__(hidden_dim, cfg, *args, **kwargs)
        self.flagger = nn.Sequential(
            nn.Conv2d(in_channels= in_dim, out_channels = hidden_dim, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(in_channels= hidden_dim, out_channels=hidden_dim, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        b, n, c = x.shape
        h = int(n ** 0.5)
        y = x.permute(0, 2, 1)[:, :, :h * h].view(b, c, h, h)
        y = self.flagger(y).permute(0, 2, 3, 1).view(b, h * h, -1)
        y = self.fc_layer(y).mean(dim=1).view(-1, 1)
        return y


class StopFlaggerAttnConv(BaseStopPolicy):
    def __init__(self, in_dim = 128, hidden_dim =32, cfg = None, *args, **kwargs):
        super().__init__(hidden_dim, cfg, *args, **kwargs)
        self.flagger = nn.Sequential(
            nn.Conv2d(in_channels= in_dim, out_channels = hidden_dim, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(in_channels= hidden_dim, out_channels=hidden_dim, kernel_size = 3, padding = 1)
        )
        self.dir_q = nn.Linear(in_dim, hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, hidden_dim)

    def forward(self, x):
        b, n, c = x.shape
        h = int(n ** 0.5)
        y = x.permute(0, 2, 1)
        dir_q = self.dir_q(y[:, :, -1]).unsqueeze(1)
        y = y[:, :, : h * h].view(b, c, h, h)
        y = self.flagger(y).permute(0, 2, 3, 1).view(b, h * h, -1)
        y = self.cross_attn(dir_q, y)
        y = self.fc_layer(y).mean(dim=1).view(-1, 1)
        return y
    
class StopFlaggerTransformer(BaseStopPolicy):
    def __init__(self, in_dim = 128, hidden_dim =32, cfg = None, *args, **kwargs):
        super().__init__(in_dim, cfg, *args, **kwargs)
        self.flagger = Transformer(in_dim, 1, 1, hidden_dim, 1)

    def forward(self, x):
        x = self.flagger(x)
        x = self.fc_layer(x)
        return x.mean(dim=(-1, -2)).view(-1, 1)
    
    
class TemporalStopPolicy(BaseStopPolicy):
    def __init__(self, in_dim=128, hidden_dim=32, cfg=None, *args, **kwargs):
        super().__init__(hidden_dim, cfg, *args, **kwargs)
        if cfg is None: cfg = {}
        self.single_step = cfg.get('single_step', True)
        self.patch_count = cfg.get('patch_count', 65)
        self.h_dim = hidden_dim
        self.hidden_state = None

        self.linear = nn.Linear(in_dim, 8)
        self.lstm = nn.LSTM(input_size=8 * self.patch_count, hidden_size=hidden_dim, num_layers=1)

    def forward(self, x):
        """
        x: [B, N, D] with N == self.patch_count
        """
        b, n, d = x.shape
        assert n == self.patch_count, f"Expected {self.patch_count} tokens, got {n}"
        x = self.linear(x).view(b, -1)

        if self.single_step:
            out, hidden = self.lstm(x.unsqueeze(0), self.hidden_state)
            self.hidden_state = (hidden[0].detach(), hidden[1].detach())
            out = out.squeeze(0)
        else:
            self.hidden_state = None
            out, _ = self.lstm(x)
        return self.fc_layer(out)

    def reset_state(self):
        self.hidden_state = None


def select_stop_flagger(flagger_type):
    if flagger_type == 'conv':
        return StopFlaggeSimpleConv
    elif flagger_type == 'attn_conv':
        return StopFlaggerAttnConv
    elif flagger_type == 'transformer':
        return StopFlaggerTransformer
    elif flagger_type == 'temporal':
        return TemporalStopPolicy
    else:
        raise NotImplementedError