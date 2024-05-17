import hydra
from omegaconf import DictConfig
import pyrootutils
import lightning as L
from lightning import LightningDataModule

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
    if isinstance(x, dict):
        for k, v in x.items():
            if hasattr(v, "shape"):
                print(f"x[{k}] shape: {v.shape}")
            else:
                print(f"x[{k}] shape: None")
    else:
        log.info(f"x shape: {x.shape}")
    if isinstance(y, dict):
        for k, v in y.items():
            log.info(f"y[{k}] shape: {v.shape}")
    else:
        log.info(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
