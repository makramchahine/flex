import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
