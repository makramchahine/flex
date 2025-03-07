from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BaseCriterion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def names(self):
        raise NotImplementedError


class FMCriterion(BaseCriterion):
    def __init__(self, weights: DictConfig, reduction: str, weights_class:Optional[DictConfig]=None):
        super(FMCriterion, self).__init__()
        self.weights = weights
        self.reduction = reduction
        self.weights_class = weights_class

    def forward(self, preds, targets):
        assert set(preds.keys()) == set(targets.keys()), \
            f"Model outputs {list(preds.keys())} is not compatible with the targets {list(targets.keys())}"

        losses = dict(total=0.)
            
        for key in self.weights.keys():
            loss = F.mse_loss(preds[key], targets[key], reduction=self.reduction)
            losses[f"{key}/MSE"] = loss
            losses["total"] += self.weights[key] * loss

        if self.weights_class is not None:
            for key in self.weights_class.keys():
                loss = F.binary_cross_entropy_with_logits(preds[key], targets[key], reduction=self.reduction)
                losses[f"{key}/BCE"] = loss
                losses["total"] += self.weights_class[key] * loss

        return losses
    
    @property
    def names(self):
        return [f"{k}/MSE" for k in self.weights.keys()] + [f"{k}/BCE" for k in self.weights_class.keys()] + ["total"]
