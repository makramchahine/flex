from typing import List, Tuple
import os
import copy
import tqdm
import traceback
import numpy as np
from skvideo.io import FFmpegWriter
import lightning as L

import hydra
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
import vista

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    
    if cfg.get("pytorch_sharing_strategy"):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy(cfg.pytorch_sharing_strategy)
    
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    modify_cfg_for_backward_compatibility(cfg)

    # datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    # model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt_path:
        import torch
        ckpt = torch.load(cfg.ckpt_path, map_location=model.device)
        for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
            if dropped_key in ckpt["state_dict"].keys():
                ckpt["state_dict"].pop(dropped_key) # HACK: remove param used for determining device
        model.load_state_dict(ckpt["state_dict"])
    if torch.cuda.is_available():
        model.cuda()

    # misc
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    # DictConfig({
    #     "csv": {
    #         "_target_": "lightning.pytorch.loggers.csv_logs.CSVLogger",
    #         "save_dir": "${paths.output_dir}",
    #         "name": "csv/",
    #         "prefix": "",
    #     }
    # }))
    os.makedirs(logger[0].log_dir, exist_ok=True)

    # trainer and logger
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)
    
    # env
    for env_cfg in [model.in_distribution_env_cfg, model.out_of_distribution_env_cfg]:
        if "car_configs" in env_cfg.keys(): # multi-agent env
            for car_cfg in env_cfg.car_configs:
                car_cfg.dynamics = "bicycle" # HACK: to sidestep the weird bug where the ado car gradually drifts in curvilinear dynamics
    envs = {
        "id": instantiate_env(model.in_distribution_env_cfg),
        "ood": instantiate_env(model.out_of_distribution_env_cfg),
    }
    if cfg.get("save_video", False):
        for env_name, env in envs.items():
            video_dir = os.path.join(cfg.paths.output_dir, "video", env_name)
            envs[env_name] = SaveVideoWrapper(env, video_dir)
    
    data_transforms = {
        "image": datamodule.val_dataloader().dataset._transform_rgb,
    }
    
    if cfg.get("save_features", False):
        out_path = os.path.join(cfg.paths.output_dir, "feats.pkl")
        feature_saver = utils.FeatureSaver(out_path, write_freq=cfg.get("write_features_freq", 10))
    else:
        feature_saver = None
        
    if cfg.get("save_info", False):
        out_path = os.path.join(cfg.paths.output_dir, "info.pkl")
        info_saver = utils.InfoSaver(out_path, write_freq=cfg.get("write_info_freq", 10))
    else:
        info_saver = None

    # run    
    step_evaled = [None, None]
    n_epsiodes = model.closed_loop_eval_cfg.n_episodes
    pbar = tqdm.tqdm(range(n_epsiodes), total=n_epsiodes)
    for ep_i in pbar:
        try:
            pbar.set_description(f"Steps Traversed {step_evaled}")
            step_evaled = []
            for env_tag, env in envs.items():
                results = eval_an_episode(
                    env,
                    model,
                    model.closed_loop_eval_cfg.max_steps,
                    data_transforms,
                    feature_saver=feature_saver,
                    info_saver=info_saver,
                )
                results["env"] = env_tag
                step_evaled.append(results["steps_traveled"])
                
                logger[0].log_metrics(results)
                logger[0].save()
        except KeyboardInterrupt:
            if cfg.get("save_features", False):
                feature_saver.close()
            if cfg.get("save_info", False):
                info_saver.close()
                
            break
            
    if cfg.get("save_features", False):
        feature_saver.close()
    if cfg.get("save_info", False):
        info_saver.close()

    return dict(), dict()


def eval_an_episode(env, model, max_steps, transforms, agent_id=None, sensor=None, feature_saver=None, info_saver=None):
    agent_id = env.world.agents[0].id if agent_id is None else agent_id
    agent = [v for v in env.world.agents if v.id == agent_id][0]
    sensor = agent.sensors[0] if sensor is None else sensor
    device = model.device
    
    model.eval()

    try:
        obs = env.reset()
    except AssertionError:
        obs = env.reset()
    ep_rew = 0.
    if info_saver is not None:
        info_saver.set_new_ep()
    for step in range(max_steps + 1):
        obs = obs[agent_id]["camera_front"]
        model_inp = {
            "image": transforms["image"](img=obs, sensor=sensor, train=False)[None, ...].to(device),
            "image_raw": obs, # for visualization only
        }
        with torch.no_grad():
            model_out = model.forward(model_inp)
            
        if feature_saver is not None:
            assert hasattr(model.net.extractor, "intermediate_features"), "No attribute intermediate_features in the feature extractor"
            feats = model.net.extractor.intermediate_features.data.cpu().numpy()
            feats = feats[0].reshape(feats.shape[1], -1).transpose(1, 0) # drop batch dim and flatten h,w
            # feats = feats[0].transpose(1, 2, 0) # drop batch dim and transpose to (h, w, c)
            feature_saver.write({"feats": feats})
        
        action = {agent_id: [model_out["curvature"].item()]} # NOTE: no speed now
        for _a in env.world.agents:
            if _a.id not in action.keys():
                action[_a.id] = np.zeros((2,))
        obs, rew, done, info = env.step(action, dt=1/30.)
        done = np.any(list(done.values()))
        rew = rew[agent_id]
        if info_saver is not None:
            info["ego_agent_id"] = agent_id
            info_saver.write(info)
        info = info[agent_id]
        
        # ## DEBUG
        # if step > 0:
        #     model.net.extractor.image_to_text_model.stop = True
        
        ep_rew += rew
        if done:
            break
    
    if hasattr(env, "close_video_writer"):
        env.close_video_writer()
        
    complete = (step / max_steps) if done else 1.0
    results = {
        "steps_traveled": step,
        "complete": complete,
        "exceed_max_rot": info.get("exceed_max_rot", None),
        "out_of_lane": info.get("out_of_lane", None),
        "crashed": info.get("crashed", None),
    }
    
    return results


class SaveVideoWrapper:
    def __init__(self, env, video_dir):
        self.env = env
        
        os.makedirs(video_dir, exist_ok=True)
        self.video_dir = video_dir
        
        display_config = dict(road_buffer_size=1000, )
        self.display = vista.Display(self.env.world, display_config=display_config)
        
        self.video_writer = None
        self.episode_i = 0
        
    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        self.display.reset()
        
        video_path = os.path.join(self.video_dir, f'episode_{self.episode_i:04d}.mp4')
        dt = 1 / 10.
        rate = f'{(1. / dt)}'
        self.video_writer = FFmpegWriter(video_path,
                                         inputdict={'-r': rate},
                                         outputdict={'-vcodec': 'libx264',
                                                     '-pix_fmt': 'yuv420p',
                                                     '-r': rate,})
        
        self.episode_i += 1
        
        return out
    
    def step(self, *args, **kwargs):
        out = self.env.step(*args, **kwargs)
        
        # dump to frame
        assert self.video_writer is not None, "Video writer is not set yet. Please run reset first"
        img = self.display.render()
        self.video_writer.writeFrame(img)
        
        return out

    def close_video_writer(self):
        self.video_writer.close()
    
    @property
    def world(self):
        return self.env.world


def instantiate_env(env_cfg):
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg["_target_"] = env_cfg.pop("cls")
    if "MultiAgentBase" in env_cfg["_target_"]: # HACK
        if "car_config" in env_cfg.keys():
            del env_cfg["car_config"] # use car_configs instead
    env = hydra.utils.instantiate(env_cfg)
    
    return env


def modify_cfg_for_backward_compatibility(cfg):
    if hasattr(cfg.data, "standardize"):
        dc = {k: v for k, v in cfg.data.items() if k != "standardize"}
        dc["use_standardize"] = cfg.data.standardize
        cfg.data = DictConfig(dc)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
