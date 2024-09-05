import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones", "gym_pybullet_drones", "examples"))
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
import numpy as np
import copy

import torch
from PIL import Image
from torchvision import transforms
import hydra
from lightning import LightningModule
from omegaconf import OmegaConf

from gym_pybullet_drones.examples.simulator_utils import get_x_y_z_yaw_relative_to_base_env
from gym_pybullet_drones.examples.simulator_eval import EvalSimulator

from config import generate_init_conditions

def cfg_2_model(cfg_path):
    cfg = OmegaConf.load(cfg_path[0])
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if cfg.ckpt_path:
        # if its a list load the last one
        cpath = cfg.ckpt_path[0]
        print(f"Loading checkpoint: {cpath}")
        ckpt = torch.load(cpath, map_location=device)
        for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
            if dropped_key in ckpt["state_dict"].keys():
                ckpt["state_dict"].pop(dropped_key)  # HACK: remove param used for determining device
        model.load_state_dict(ckpt["state_dict"])

    return model.to(device)

def closed_loop_render_set(model, init_conditions, record_hz, closed_loop_save_path, text_instr=None,
                           selected_index=None):
    
    assert closed_loop_save_path is not None
    save_path = closed_loop_save_path[0]
    makedirs(save_path, exist_ok=True)


    # ! Setup Simulator
    sim = EvalSimulator(save_path, init_conditions, record_hz, selected_index)
    sim.setup_simulation()

    CLOSED_LOOP_NUM_FRAMES = 80

    # init stabilization
    vel_cmd = np.array([0, 0, 0, 0])
    sim.vel_cmd_world = vel_cmd
    updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
    updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
    pybullet_img = pybullet_img[None, :, :, 0:3]

    init_forward = updated_position[0]

    unnormalized_vel_cmds = []
    for _ in tqdm(range(CLOSED_LOOP_NUM_FRAMES)):

        image = copy.deepcopy(pybullet_img)
        # squeeze first dimension of image
        image = image.squeeze(0)
        image = Image.fromarray(image)
        img = image.resize((224, 224))
        # convert the image to a tensor
        img = transforms.ToTensor()(img).to('cuda:0')

        text = text_instr

        # run inference
        preds = model.forward({"image": img, "text": text})

        # convert dictionnary of 1D tensors to array of floating numbers
        # dictionnary has 4 keys: "vx", "vy", "vz", "yaw"
        out = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()

        unnormalized_vel_cmds.append(out[0])
        vel_cmd = out[0]  # shape: 1 x 4

        # Put into simulator
        updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
        if finished:
            break
        pybullet_img = pybullet_img[None, :, :, 0:3]

        updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
        updated_position -= [init_forward, 0, 0.6, sim.theta_environment]

    print("instruction_text: ", text)
    with open(os.path.join(save_path, "instruction_text.txt"), "w") as file:
        file.write(text)
    sim.export_plots()
    np.savetxt(os.path.join(save_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--cfg_path", nargs='*', default=None, type=str)
    parser.add_argument("--closed_loop_save_path", nargs='*', default=None, type=str)
    parser.add_argument("--objects_color", nargs='*', default=None, type=str)
    parser.add_argument('--text_instr', type=str, default="", help='Text instruction')
    parser.add_argument('--selected_index', type=int, default=0, help='Selected index')
    args = parser.parse_args()

    closed_loop_save_path = getattr(args, "closed_loop_save_path", None)
    print(f"closed_loop_save_path: {closed_loop_save_path}")
    params_paths = getattr(args, "params_paths", None)
    checkpoint_paths = getattr(args, "checkpoint_paths", None)

    init_conditions = generate_init_conditions(args.objects_color,
                                                1, # don't care about this
                                                closed_loop_save_path)

    model = cfg_2_model(args.cfg_path)
    closed_loop_render_set(model, init_conditions, 3, closed_loop_save_path, args.text_instr,
                           args.selected_index)
