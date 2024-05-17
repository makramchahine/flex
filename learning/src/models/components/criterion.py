from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F


class BaseCriterion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def names(self):
        raise NotImplementedError


class MSECriterion(BaseCriterion):
    def __init__(self, weights: DictConfig, reduction: str):
        super(MSECriterion, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, preds, targets):
        assert set(preds.keys()) == set(targets.keys()), \
            f"Model outputs {list(preds.keys())} is not compatible with the targets {list(targets.keys())}"

        losses = dict(total=0.)
        for key in targets.keys():
            loss = F.mse_loss(preds[key], targets[key], reduction=self.reduction)
            losses[f"{key}/MSE"] = loss
            losses["total"] += self.weights[key] * loss

        return losses
    
    @property
    def names(self):
        return [f"{k}/MSE" for k in self.weights.keys()] + ["total"]
