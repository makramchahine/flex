import hydra
from omegaconf import DictConfig
import pyrootutils
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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    log.info(f"\n{model}")
    
    model.on_train_start()
    
    train_loss = model.training_step(batch, 0)
    log.info(f"Training loss: {train_loss}")
    
    model.validation_step(batch, 0)
    log.info("Validation step done")
    
    model.on_validation_epoch_end()
    log.info("Validation epoch done")


if __name__ == "__main__":
    main()
