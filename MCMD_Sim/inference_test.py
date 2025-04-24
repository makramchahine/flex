import torch
from PIL import Image
from torchvision import transforms # type: ignore
from tqdm import tqdm
from simulator import MCMDSimEval
import hydra # type: ignore
from omegaconf import OmegaConf # type: ignore
from lightning import LightningModule # type: ignore
from simulator.utils.gen_init_cond import ICCLISchema2SimEnvSchema, generate_closed_loop_1drone_nobject_env_init
from simulator.utils.tasks import get_task_delta
from enum import Enum
from utils import generate_init_conditions_closed_loop_inference_2choice, generate_init_conditions_closed_loop_inference_3choice_random
import random
from datetime import datetime
import os
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt

class Task(Enum):
    RB_2CHOICE = "2rb"
    COLOR_5 = "2colors"
    R_SHAPE = "2rshape"
    M_SHAPE = "2mshape"
    OPEN_DICT = "3open_dict"

def get_meta(task):
    if task == Task.RB_2CHOICE:
        objects = ["red ball", "blue ball"]
        num_obj = 2
        generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
    elif task == Task.COLOR_5:
        objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
        num_obj = 2
        generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
    elif task == Task.R_SHAPE:
        objects = ["red ball", "red cube", "red pyramid"]
        num_obj = 2
        generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
    elif task == Task.M_SHAPE:
        # objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball", "red cube", "blue cube", "yellow cube", "green cube", "purple cube", "red pyramid", "blue pyramid", "yellow pyramid", "green pyramid", "purple pyramid"]
        objects = ["yellow ball", "green ball", "purple ball", "red cube", "blue cube", "yellow cube", "green cube", "purple cube", "red pyramid", "blue pyramid", "yellow pyramid", "green pyramid", "purple pyramid"]
        num_obj = 2
        generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
    elif task == Task.OPEN_DICT:
        # objects = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
        objects = ["red cube", "blue pyramid", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
        num_obj = 3
        generate_init_conditions = generate_init_conditions_closed_loop_inference_3choice_random
    else:
        raise ValueError("Invalid task")
    return objects, num_obj, generate_init_conditions

def trajectory_score(obs_arr, exp_point, ref_point):
    exp_vec = exp_point - ref_point
    exp_unit = exp_vec / np.linalg.norm(exp_vec)

    obs_vecs = obs_arr - ref_point
    obs_norms = np.linalg.norm(obs_vecs, axis=1)
    valid_mask = obs_norms > 0
    obs_units = obs_vecs[valid_mask] / obs_norms[valid_mask][:, None]

    cos_sims = obs_units @ exp_unit
    dists = np.linalg.norm(obs_arr[valid_mask] - exp_point, axis=1)
    scale = np.linalg.norm(exp_vec)
    rel_dists = dists / scale

    scores = cos_sims * np.exp(- 0.35 * (rel_dists ** 4))
    return np.max(scores), np.max(cos_sims), np.min(rel_dists)

def cfg_2_model(cfg_path, feature_extraction = True):
    d, t, c = cfg_path
    cfg_path = os.path.join('/home/alex/flex/local/train_flight', d, t)
    cfg = OmegaConf.load(os.path.join(cfg_path, "config")).model
    cfg.net.extractor.extract_features_flag = feature_extraction
    if not feature_extraction: cfg.net.extractor.checkpoint = None
    model: LightningModule = hydra.utils.instantiate(cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path_base = os.path.join(cfg_path, "checkpoints")
    if c  == -1:
        ckpt_paths = sorted([f for f in os.listdir(ckpt_path_base) if f.endswith('.ckpt')])
        cpath = os.path.join(ckpt_path_base, ckpt_paths[-1])
    else:
        cpath = os.path.join(ckpt_path_base, f'step_{c}.ckpt')

    if not os.path.exists(cpath): raise ValueError(f"Check the checkpoint path {cpath}")
    ckpt = torch.load(cpath, map_location=torch.device('cpu'), weights_only=False)
    ckpt = ckpt['state_dict']
    model.net.policy.load_state_dict(ckpt['policy'])
    model.net.extractor.last_linear_layer.load_state_dict(ckpt['extractor_ll'])
    if model.net.stop_flagger is not None and "stop_flagger" in ckpt.keys():
        model.net.stop_flagger.load_state_dict(ckpt['stop_flagger'])
        if cfg.net.stop_flagger_cfg.name == 'temporal': model.net.stop_flagger.single_step = True
    
    return model.to(device), cfg

def cfg2prefix(cfg, env_name):
    pol_name = cfg.net.policy.get('_target_').split('.')[-2]
    pol_meta = ''
    if pol_name.lower().endswith('vit') or pol_name.lower()[:-1].endswith('vit'):
        psize = cfg.net.extractor.get('patch_size', 2)
        nhead = cfg.net.policy.cfg.get('heads', 4)
        depth = cfg.net.policy.cfg.get('depth', 3)
        pol_meta = f'p{psize}h{nhead}d{depth}'
    
    spol = cfg.net.get('stop_flagger_cfg', '')
    if spol != '': spol = spol.name

    dt = datetime.now().strftime("%S")
    prefix = f'{pol_name}{pol_meta}_{spol}_{env_name}{dt}_'
    return prefix

def closed_loop_render_set(
        model, 
        init_conditions, 
        record_hz, 
        text_instr=None,
        prefix = 'Experiment',
        log_text=None,
        max_steps = 120,
        stop_threshold = 100
    ):
    # ! Setup Simulator
    sim = MCMDSimEval(init_conditions, record_hz, log_prefix=prefix)
    image = sim.setup_simulation(verbose=False)[0]
    pred_stop = []

    with torch.no_grad():
        for iter in tqdm(range(max_steps)):
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')

            preds = model({"image": img, "text": [text_instr]})
            # preds = {
            #     "vx": torch.tensor([0.0]),
            #     "vy": torch.tensor([0.0]),
            #     "vz": torch.tensor([0.0]),
            #     "yaw": torch.tensor([0.0]),
            #     "stop": torch.tensor([0.0]),
            #     "step_remain": torch.tensor([1.0])
            # }
            if 'stop' not in preds: preds['stop'] = torch.tensor([-1.0])
            if 'step_remain' not in preds: preds['step_remain'] = torch.tensor([1.0])
            stop_val = torch.sigmoid(preds['stop'])[0].cpu().item()
            pred_stop.append([stop_val, preds['step_remain'][0].cpu().item()])
            if iter > 25 and stop_val > stop_threshold: break

            vel_cmd = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            image = sim.step_action(vel_cmd)
            image = image[0]
    # sim.logger.log_text(f'Stop Model : {}')  
    sim.logger.log_text(f'Text Command : {text_instr}')
    if log_text is not None:
        sim.logger.log_text(f'Log Text : {log_text}')

    if len(pred_stop) > 0:
        pred_stop = np.array(pred_stop)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(pred_stop[:, 0], label='stop')
        ax.plot(pred_stop[:, 1], label='step_remain')
        ax.legend()
        ax.set_title("Stop Prediction")
        ax.set_xlabel("Step")
        ax.set_ylabel("Stop Probability")
        plt.savefig(os.path.join(sim.logger.log_dir, 'stop_pred.png'))
        plt.close()
        np.savetxt(os.path.join(sim.logger.log_dir, 'stop_pred.txt'), pred_stop)

    logger = sim.close()
    return logger


if __name__ == "__main__":
    model_cfg_path = ['2025-04-24', '13-15-24', '125000']
    log_path = f'results_test/{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    model, cfg = cfg_2_model(model_cfg_path)
    tasks = [Task.M_SHAPE, Task.OPEN_DICT]
    envs = ["samurai"]
    commands = ["below", "above", "left", "right", "towards"]
    target_idxs = [0, 1]
    num_exp = 5
    result = []

    for i in range(num_exp):
        for task in tasks:
            for env in envs:
                for cmd in commands:
                    for target_idx in target_idxs:
                        objects, num_obj, generate_init_conditions = get_meta(task)
                        prefix = cfg2prefix(cfg, env)
                        random.shuffle(objects)
                        objects = objects[:num_obj]
                        init_conditions = generate_init_conditions(objects)
                        # init_conditions = generate_closed_loop_1drone_nobject_env_init(objects, env, target_idx, cmd, log_path)
                        init_conditions = ICCLISchema2SimEnvSchema(init_conditions, env, target_idx, cmd, log_path)
                        text_instr = f"{cmd}---{objects[target_idx]}"
                        logger = closed_loop_render_set(model, 
                                                        init_conditions, 
                                                        record_hz=3, 
                                                        text_instr=text_instr, 
                                                        prefix = prefix,
                                                        log_text = f"{model_cfg_path}_{env}_{task.value}_{cmd}_{target_idx}",
                                                    )
                        exp_vec = get_task_delta(init_conditions['command'])
                        ref_point = np.array(init_conditions["objects_loc"][init_conditions["target_idx"]])
                        score, cos_sim, dist = trajectory_score(np.array(logger.global_pos_array[0])[:, :3], exp_vec+ref_point, ref_point)
                        result.append({
                            'env_name': init_conditions['env_name'],
                            'command': init_conditions['command'],
                            'tidx': init_conditions["target_idx"],
                            'score': score,
                            'cos_sim': cos_sim,
                            'dist': dist,
                            'run': logger.log_dir
                        })
    df = pd.DataFrame(result)
    df.to_csv(os.path.join(log_path, 'result.csv'), index=False)
