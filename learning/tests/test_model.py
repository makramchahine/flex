import hydra
from omegaconf import DictConfig
import pyrootutils
import torch
import lightning as L
from lightning import LightningDataModule, LightningModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    datamodule.prepare_data()
    datamodule.setup()
    
    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path:
        import torch
        ckpt = torch.load(cfg.ckpt_path, map_location=model.device)
        for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
            if dropped_key in ckpt["state_dict"].keys():
                ckpt["state_dict"].pop(dropped_key) # HACK: remove param used for determining device
        model.load_state_dict(ckpt["state_dict"])
    log.info(f"\n{model}")
    
    pred = model(x)
    if isinstance(y, dict):
        assert set(pred.keys()) == set(y.keys()), f"Model output {list(pred.keys())} is not compatible with the target {list(y.keys())}"
        for k in y.keys():
            assert pred[k].shape == y[k].shape, f"Model output shape {pred[k].shape} is not consistent with the target shape {y[k].shape}"
            log.info(f"[{k}] Model output shape: {pred[k].shape}")
            log.info(f"[{k}] Target shape: {y[k].shape}")
    else:
        assert pred.shape == y.shape, f"Model output shape {pred.shape} is not consistent with the target shape {y.shape}"
        log.info(f"Model output shape: {pred.shape}")
        log.info(f"Target shape: {y.shape}")


if __name__ == "__main__":
    main()
