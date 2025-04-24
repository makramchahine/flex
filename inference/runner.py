import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

import torch
from PIL import Image
from torchvision import transforms # type: ignore
import hydra # type: ignore
from lightning import LightningModule # type: ignore
from omegaconf import OmegaConf # type: ignore
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from MCMD_Sim.simulator.simulator_mcmd_exp import MCMDSimEval
from MCMD_Sim.simulator.utils.gen_init_cond import ICCLISchema2SimEnvSchema
from MCMD_Sim.simulator.utils.tasks import generate_instruction
from config import generate_init_conditions, env_name, command

def cfg_2_model(cfg_path):
    cfg = OmegaConf.load(os.path.join(cfg_path[0], "config"))
    cfg.model.net.extractor.extract_features_flag = True
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path_base = os.path.join(cfg_path[0], "checkpoints")
    # raise ValueError(f"Check the checkpoint path {ckpt_path_base}")
    ckpt_paths = sorted([f for f in os.listdir(ckpt_path_base) if f.endswith('.ckpt')])

    if ckpt_paths:
        cpath = os.path.join(ckpt_path_base, ckpt_paths[-1])
        ckpt = torch.load(cpath, map_location=torch.device('cpu'), weights_only=False)
        model.net.policy.load_state_dict(ckpt['state_dict']['policy'])
        model.net.extractor.last_linear_layer.load_state_dict(ckpt['state_dict']['extractor_ll'])

    pol_name = cfg.model.net.policy.get('model_type', 'LSTM')[-4:]
    num_action = cfg.model.net.policy.cfg.get('num_classes', 4)
    pol_meta = ''
    if pol_name.lower().endswith('vit'):
        psize = cfg.model.net.extractor.get('patch_size', 2)
        nhead = cfg.model.net.policy.cfg.get('heads', 4)
        depth = cfg.model.net.policy.cfg.get('depth', 3)
        pol_meta = f'p{psize}h{nhead}d{depth}'

    dt = datetime.now().strftime("%S")
    prefix = f'{pol_name}{num_action}{pol_meta}_{env_name}{dt}_'

    return model.to(device), prefix

def closed_loop_render_set(
        model, 
        init_conditions, 
        record_hz, 
        text_instr=None,
        prefix = 'Experiment',
        log_text=None,
    ):
    # ! Setup Simulator
    sim = MCMDSimEval(init_conditions, record_hz, log_prefix=prefix)
    image = sim.setup_simulation()[0]
    CLOSED_LOOP_NUM_FRAMES = 80

    with torch.no_grad():
        for iter in tqdm(range(CLOSED_LOOP_NUM_FRAMES)):
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')

            preds = model({"image": img, "text": text_instr})

            vel_cmd = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            image = sim.step_action(vel_cmd)
            image = image[0]
        
        sim.logger.log_text(f'Text Command : {text_instr}')
        if log_text is not None:
            sim.logger.log_text(f'Log Text : {log_text}')

    logger = sim.close()
    return logger



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--cfg_path", nargs='*', default=None, type=str)
    parser.add_argument("--closed_loop_save_path", nargs='*', default=None, type=str)
    parser.add_argument("--objects_color", nargs='*', default=None, type=str)
    parser.add_argument('--selected_index', type=int, default=0, help='Selected index')
    args = parser.parse_args()

    closed_loop_save_path = getattr(args, "closed_loop_save_path", None)

    init_conditions = generate_init_conditions(args.objects_color)
    init_conditions = ICCLISchema2SimEnvSchema(
        init_conditions, 
        env_name=env_name, 
        target_idx=args.selected_index, 
        command=command,
        log_path=closed_loop_save_path[0],
    )

    model, prefix = cfg_2_model(args.cfg_path)
    text_instr = generate_instruction(direction=command, obj_type=args.objects_color[args.selected_index])
    closed_loop_render_set(model, init_conditions, 3, text_instr, prefix=prefix, log_text=args.cfg_path[0])
