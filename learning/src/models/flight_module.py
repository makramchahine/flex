from typing import Any, Union, List, Optional
import copy
from omegaconf import DictConfig
import hydra

import torch
from torch.nn import ModuleDict
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError

from src.models.components.base import BaseNet
from src.models.components.criterion import BaseCriterion


class FlightLitModule(LightningModule):
    def __init__(
        self,
        net: BaseNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: BaseCriterion,
        closed_loop_eval_cfg: Optional[Union[DictConfig, None]] = None,
        in_distribution_env: Optional[Union[Any, None]] = None,
        out_of_distribution_env: Optional[Union[Any, None]] = None,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "criterion", "in_distribution_env", "out_of_distribution_env"],
        )

        self.net = net

        if hasattr(self.net.extractor, "set_backbone"):
            self.net.extractor.set_backbone(self)
        self.criterion = criterion
        
        loss_tracker = MetricCollection({k: MeanMetric() for k in self.criterion.names})
        self.train_loss_tracker = loss_tracker.clone(prefix="train_loss/")
        self.val_loss_tracker = loss_tracker.clone(prefix="val_loss/")
        
        metric_mse = MetricCollection({k: MeanSquaredError() for k in self.net.output_names})
        metric_mae = MetricCollection({k: MeanAbsoluteError() for k in self.net.output_names})
        self.train_metric_mse = metric_mse.clone(prefix="train_metric/mse/")
        self.train_metric_mae = metric_mae.clone(prefix="train_metric/mae/")
        self.val_metric_mse = metric_mse.clone(prefix="val_metric/mse/")
        self.val_metric_mae = metric_mae.clone(prefix="val_metric/mae/")

        self.closed_loop_eval_cfg = closed_loop_eval_cfg
        self.in_distribution_env_cfg = in_distribution_env
        self.out_of_distribution_env_cfg = out_of_distribution_env
        self.env_is_instantiated = False

        self.closed_loop_metric_names = ["out_of_lane", "exceed_max_rot", "complete", "steps_traveled", "crashed"]
        cls_metric_tracker = MetricCollection({k: MeanMetric() for k in self.closed_loop_metric_names})
        self.closed_loop_metrics = ModuleDict({
            "in_distr": cls_metric_tracker.clone(f"closed_loop_eval_in_distr/"),
            "out_of_distr": cls_metric_tracker.clone(f"closed_loop_eval_out_of_distr/"),
        })
        
        self.closed_loop_eval_transform = dict()

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def model_step(self, batch: Any):
        x, y = batch
        # x_is_last = x['is_last'].to(torch.float32).view(-1, 1)
        preds = self.forward(x)
        losses = self.criterion(preds, y)
        # print(x_is_last, preds_stop, x_is_last.shape, preds_stop.shape)
        # pause = input('sanity check')
        # losses['total'] += self._stop_criterion(preds_stop, x_is_last)

        return losses, preds, y
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_tracker.reset()
        self.val_metric_mse.reset()
        self.val_metric_mae.reset()
    
    def training_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.model_step(batch)

        self._log_loss_tracker(self.train_loss_tracker, losses)

        self._log_metrics(self.train_metric_mse, preds, targets)
        self._log_metrics(self.train_metric_mae, preds, targets)
        
        return losses["total"]
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {
            'policy' : self.net.policy.state_dict(),
            'extractor_ll' : self.net.extractor.last_linear_layer.state_dict()
        }
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_start(self):
        pass
        # NOTE: Only can instantiate envs here (rather than at model.init) otherwise causing
        #       ValueError: ctypes objects containing pointers cannot be pickled
        # if self.global_rank == 0 and not self.env_is_instantiated:
        #     self.in_distribution_env = self._instantiate_env(self.in_distribution_env_cfg)
        #     self.out_of_distribution_env = self._instantiate_env(self.out_of_distribution_env_cfg)
        #     self.env_is_instantiated = True
        #
        #     self.closed_loop_eval_transform["image"] = self.trainer.datamodule.val_dataloader().dataset._transform_rgb
    
    def validation_step(self, batch: Any, batch_idx: int):
        losses, preds, targets = self.model_step(batch)

        self._log_loss_tracker(self.val_loss_tracker, losses)

        self._log_metrics(self.val_metric_mse, preds, targets)
        self._log_metrics(self.val_metric_mae, preds, targets)
        
        # if self.global_rank == 0:
        #     self.run_closed_loop_eval()

    def on_validation_epoch_end(self):
        pass
    
    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError
    
    def on_test_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def run_closed_loop_eval(self):
        envs = dict()
        if self.in_distribution_env is not None:
            envs["in_distr"] = self.in_distribution_env
        if self.out_of_distribution_env is not None:
            envs["out_of_distr"] = self.out_of_distribution_env

        for env_name, env in envs.items():
            NotImplementedError("Closed loop evaluation is not implemented yet!")

    def _log_loss_tracker(self, loss_tracker, losses):
        for key in self.criterion.names:
            loss_tracker[key](losses[key])
        self.log_dict(loss_tracker, on_step=True, prog_bar=True) #, on_epoch=True)
    
    def _log_metrics(self, metrics, preds, targets):
        for key in self.net.output_names:
            metrics[key](preds[key], targets[key])
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def _instantiate_env(self, env_cfg):
        if env_cfg is None:
            env = None
        else:
            env_cfg = copy.deepcopy(env_cfg)
            env_cfg["_target_"] = env_cfg.pop("cls")
            env = hydra.utils.instantiate(env_cfg)

        return env
    