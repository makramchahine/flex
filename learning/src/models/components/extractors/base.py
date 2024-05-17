import torch.nn as nn


class BaseExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        assert set(x.keys()).issubset(set(self.modalities))

    @property
    def modalities(self):
        raise NotImplementedError # return a list
