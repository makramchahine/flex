import os
import tqdm
import hydra
from omegaconf import DictConfig
import pyrootutils
import numpy as np
import cv2
import lightning as L
from lightning import LightningDataModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from vista.core.Display import curvature2noodle, plot_roi

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.data.num_workers == 0, "Please set data.num_workers to 0"
    
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    datamodule.prepare_data()
    datamodule.setup()
    
    dataloader = datamodule.val_dataloader()
    data_iter = iter(dataloader)
    camera_param = None
    
    n_frames = cfg.get("n_frames", 100)
    
    out_path = cfg.get("out_path", "./test.mp4")
    video_writer = None

    for i in tqdm.tqdm(range(n_frames), total=n_frames, desc="Writing to video"):
        try:
            batch = next(data_iter)
            x, y = batch

            if camera_param is None: # need to be called after querying at least one data sample otherwise the sim is not initialized yet
                camera_param = dataloader.dataset.world.agents[0].sensors[0].camera_param
            
            img = (x["image"][0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            curvature = y["curvature"][0].item()
            noodle = curvature2noodle(curvature, camera_param, mode='camera')
            img = plot_roi(img.copy(), camera_param.get_roi())
            img = cv2.polylines(img, [noodle], False, (255, 0, 0), 2)
            
            if video_writer is None:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(out_path, fourcc, 30,
                        (camera_param.get_width(), camera_param.get_height()))
            video_writer.write(img)
        except KeyboardInterrupt:
            break
        
    video_writer.release()


if __name__ == "__main__":
    main()
