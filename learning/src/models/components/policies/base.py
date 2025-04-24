import torch.nn as nn
import torch

class BasePolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__last_rep = None
    
    def save_rep(self, rep):
        self.__last_rep = rep.detach().clone()

    def get_last_rep(self):
        if self.__last_rep is None: raise ValueError("No Last Representation Saved")
        temp = self.__last_rep
        self.__last_rep = None
        return temp

class BaseStopPolicy(nn.Module):
    def __init__(self, fc_dim, cfg = None, *args, **kwargs):
        super().__init__()
        self.step_remain_flag = False
        self.fc = nn.Linear(fc_dim, 1)
        if cfg is not None and cfg.get('step_remain_flag', False):
            self.step_remain_flag = True
            self.fc_step = nn.Sequential(
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, 1),
                nn.Sigmoid(),
            )
    
    def fc_layer(self, x):
        out = self.fc(x)
        if self.step_remain_flag:
            step_out = self.fc_step(x)
            return torch.cat([out, step_out], dim=-1)
        return out
