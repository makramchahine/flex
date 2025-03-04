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

from gym_pybullet_drones.examples.simulator_utils import get_x_y_z_yaw_relative_to_base_env
from gym_pybullet_drones.examples.simulator_eval import EvalSimulator

from config import generate_init_conditions

# def cfg_2_model(cfg_path):
#     print(cfg_path, "config_path")
#     cfg = OmegaConf.load(cfg_path[0])
#     print(cfg.model, "cfg_model")
#     model: LightningModule = hydra.utils.instantiate(cfg.model)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(f"Device: {device}")

#     if cfg.ckpt_path:
#         # if its a list load the last one
#         cpath = cfg.ckpt_path[0]
#         print(f"Loading checkpoint: {cpath}")
#         ckpt = torch.load(cpath, map_location=device)
#         for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
#             if dropped_key in ckpt["state_dict"].keys():
#                 ckpt["state_dict"].pop(dropped_key)  # HACK: remove param used for determining device
#         model.load_state_dict(ckpt["state_dict"])

#     return model.to(device)

import numpy as np

def make_waypoints(config):
    """
    Extract or define a set of 3D waypoints from the given config.
    For simplicity, we just show a static path that uses the object positions.
    In practice, you might compute these based on 'objects_relative' etc.
    """
    # z start is from 'start_heights'
    z_start = config["start_heights"][0] if "start_heights" in config else 0.6
    
    # We know the objects are at:
    obj1 = config["objects_relative"][0]  # e.g. (1.728..., 0.3)
    obj2 = config["objects_relative"][1]  # e.g. (1.728..., -0.3)
    
    # Suppose we just define a path that arcs around both objects
    waypoints = [
        (0.0,    0.0,   z_start),             # Start
        (obj1[0]+0.07, obj1[1]+0.2, z_start), # Right of right object
        (obj1[0]+0.2,  obj1[1],     z_start), # Straight forward
        ((obj1[0]+obj2[0])/2, 0.0,   z_start),# Some midpoint
        (obj2[0]+0.07, obj2[1]-0.1, z_start), # Curve near left object
        (obj2[0],      obj2[1],     z_start)  # End near left object
    ]
    
    return waypoints

def trajectory_generator(config, Kp=0.2, threshold=0.05):
    """
    A generator that yields [vx, vy, vz, yaw] at each discrete step.
    Uses a simple P-control with delta_t = 1.
    
    Arguments:
      config (dict): the JSON-like dictionary with task info.
      Kp (float): proportional gain.
      threshold (float): distance threshold to consider a waypoint reached.
    """
    # 1) Build or retrieve the waypoints
    wpts = make_waypoints(config)  # list of (x, y, z)
    
    # 2) Initialize drone's state
    # We'll start at the first waypoint, or (0,0, z_start) if we prefer.
    pos = np.array(wpts[0], dtype=float)
    yaw = 0.0  # start with yaw=0. you could also derive an initial yaw from config

    # 3) For each waypoint in sequence
    for target_index in range(1, len(wpts)):
        wpt = np.array(wpts[target_index], dtype=float)
        
        while True:
            # Error to current waypoint
            error = wpt - pos  # (ex, ey, ez)
            dist = np.linalg.norm(error)
            
            # Check if we've reached the waypoint
            if dist < threshold:
                # "Snap" to the waypoint and break
                pos = wpt.copy()
                break
            
            # P-control: velocity = Kp * error (since dt=1, velocity is just scaled position difference)
            vx, vy, vz = Kp * error

            # Yaw from velocity direction
            speed_xy = np.hypot(vx, vy)
            if speed_xy > 1e-6:
                yaw = np.arctan2(vy, vx)
            
            # Yield [vx, vy, vz, yaw]
            yield np.array([vx, vy, vz, yaw])
            
            # Update position (pos_new = pos_old + v * dt, dt=1)
            pos += np.array([vx, vy, vz])


def closed_loop_render_set(init_conditions, record_hz, closed_loop_save_path, text_instr=None,
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
    tgen = trajectory_generator(init_conditions)

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
        # preds = model.forward({"image": img, "text": text})

        # convert dictionnary of 1D tensors to array of floating numbers
        # dictionnary has 4 keys: "vx", "vy", "vz", "yaw"
        # out = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
        try:
            out = next(tgen)

            unnormalized_vel_cmds.append(out)
            vel_cmd = out  # shape: 1 x 4

            # Put into simulator
            updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
            if finished:
                break
            pybullet_img = pybullet_img[None, :, :, 0:3]

            updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
            updated_position -= [init_forward, 0, 0.6, sim.theta_environment]
        except:
            print('stop iteration error')

    print("instruction_text: ", text)
    with open(os.path.join(save_path, "instruction_text.txt"), "w") as file:
        file.write(text)
    sim.export_plots()
    np.savetxt(os.path.join(save_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
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
    print(init_conditions, "init_conditions_err")

    closed_loop_render_set(init_conditions, 3, closed_loop_save_path, args.text_instr,
                           args.selected_index)
