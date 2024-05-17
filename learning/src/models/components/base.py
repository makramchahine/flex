import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @property
    def modalities(self):
        raise NotImplementedError
    
    @property
    def output_names(self):
        raise NotImplementedError
