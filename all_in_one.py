import os
import torch
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import copy
import json
from PIL import Image
from learning.src.models.components.extractors.blip import BLIPExtractor
from learning.src.models.components.policies.lstm import LSTMPolicy
from learning.src.models.components.policies.transformer import TransformerPolicy as VITPolicy
from learning.src.data.components.flight_il_dataset_clean import FlightILDataset
from dataclasses import dataclass, asdict
from torchvision import transforms # type: ignore
from omegaconf import OmegaConf # type: ignore
import argparse
    
class CKPT:
    def __init__(self, train_day, train_time, train_step, policy = None, mode = None):
        self.train_day = train_day
        self.train_time = train_time
        self.train_step = train_step
        self.policy_name = policy[0]
        self.num_actions = policy[1]
        self.mode = mode
        self.path = None
        self.policy_cfg = None
        
        base = '/home/alex/flex/local/train_flight'
        cpath = os.path.join(base, self.train_day, self.train_time, 'checkpoints', f'step_{self.train_step}.ckpt')
        if os.path.exists(cpath):
            self.path = cpath
            config = OmegaConf.load(os.path.join(base, self.train_day, self.train_time, 'config'))
            self.policy_name = config.model.net.policy.get('model_type', 'LSTM')
            if self.policy_name == 'SimpleViT': self.policy_name = 'VIT'
            self.policy_cfg = config.model.net.policy.cfg
            self.num_actions = config.model.net.policy.cfg.num_classes

    @property
    def prefix(self):
        if not self.path: return ''
        day = self.train_day.split('-')[-1]
        tm = '_'.join(self.train_time.split('-')[:2])
        st = int(int(self.train_step) / 1000)
        return f'{self.policy_name}{self.num_actions}_{day}_{tm}_{st}k_'
# ----------------- CONFIGURATION CLASSES ---------------------------------------
mode = 'infer' # train, eval, infer, infer_eval
policy_name = 'LSTM' # LSTM, VIT
num_actions = 5
ckpt = CKPT('2025-04-09', '10-14-08', '325000', policy = (policy_name, num_actions), mode = mode)

@dataclass
class InferenceConfig:
    env_name:str = 'samurai' # arena, samurai
    obj0:str = 'colorless rocket'
    obj1:str = 'red ball'
    target_idx:int = 0
    text_cmd:str = 'right'
    max_step:int = 100
    stop_thresh:float = 0.4

@dataclass
class VITConfig: 
    image_size:tuple[int, int] = (32, 1)
    patch_size:tuple[int, int] = (1, 1)
    dim:int = 128
    depth:int = 3
    heads:int = 4
    mlp_dim:int = 256
    channels:int = 64
    dim_head:int = 32
    num_classes:int = ckpt.num_actions

@dataclass
class LSTMConfig: 
    channels:int = 64
    reduced_dim:int = 8
    spatial_dim:int = 8
    seq_length:int = 32
    hidden_dim:int = 256
    num_layers:int = 2
    num_classes:int = ckpt.num_actions
    single_step:bool = False
    dropout:float = 0.5

@dataclass
class DataConfig:
    data_path:str = 'BLIP2_DATASET/train/' if mode == 'train' else 'BLIP2_DATASET/train/'
    snippet_size:int = 100
    seq_length:int = 32 if mode == 'train' else 1
    stride:int = 5 if mode == 'train' else 1
    batch_size:int = 32 if mode == 'train' else 1
    num_workers:int = 1 if mode == 'train' else 0
    shuffle:bool = mode == 'train'

@dataclass
class TrainingConfig:
    learning_rate:int = 1e-3
    checkpoint_interval:int = 600
    seed_value:int = 42
    num_epochs:int = 1000

@dataclass
class ExperimentConfig:
    policy_cfg:LSTMConfig = ckpt.policy_cfg if ckpt.policy_cfg is not None else LSTMConfig() if ckpt.policy_name == 'LSTM' else VITConfig()
    train_cfg:TrainingConfig = TrainingConfig()
    data_cfg:DataConfig = DataConfig()
    infer_cfg:InferenceConfig = InferenceConfig()

    device:str = torch.device('cuda:0')
    policy_name:str = ckpt.policy_name
    mode:str = mode
    log_dir:str = os.path.join('local2', mode, ckpt.policy_name +'325k' + datetime.now().strftime('%Y_%m_%d'))
    checkpoint_path:str = ckpt.path if ckpt else None
    new_ckpt_mode:bool = True
    meta_data:str = ''

    def __post_init__(self):
        if self.mode != 'train':
            assert self.checkpoint_path is not None, "Checkpoint is required for evaluation and inference"

    def json_dump(self):
        self.meta_data = ExperimentConfig.meta_data
        self.device = str(self.device)
        with open(os.path.join(self.log_dir, f'exp_cfg_{datetime.now().strftime("%H_%M")}.json'), "w") as f:
            json.dump(asdict(self), f, indent=4)
        self.device = torch.device(self.device)

# ----------------- SEED FUNCTION -----------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)    

# ----------------- End to End Model -----------------
class E2ENet(nn.Module):
    def __init__(self, policy, device, feature_extraction = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = BLIPExtractor(
            'blip2_feature_extractor',
            'pretrain',
            freeze_blip=True,
            use_low_dim_feature=False,
            last_linear_layer=[768, 64],
            use_continuous_pe=False,
            stride=14,
            checkpoint='~/.cache/torch/hub/checkpoints/blip2_pretrained.pth',
            patch_size=2,
            all_q_dims= False,
            use_masked_patch_wise_feature= True,
            use_visual_encoder_only= False,
            extract_features_flag = feature_extraction
        )
        self.policy = policy
        self._output_names = ['vx', 'vy', 'vz', 'yaw', 'stop']
        self.device = device
        self.to(device)

    def forward(self, x):
        x = {'image': x['image'].to(self.device), 'text':x['text']}
        z = self.extractor(x)
        out = self.policy(z)

        out_dim = out.shape[-1]
        out = {k: out[...,i] for i, k in enumerate(self._output_names[:out_dim])}

        return out
    
    def save_policy(self, path):
        torch.save({'state_dict':{
            'policy' : self.policy.state_dict(),
            'extractor_ll': self.extractor.last_linear_layer.state_dict()
        }}, path)

    def load_checkpoint(self, path, new_ckpt_mode=False):
        ckpt_dict = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_dict = ckpt_dict['state_dict']
        if new_ckpt_mode:
            self.policy.load_state_dict(ckpt_dict['policy'])
            self.extractor.last_linear_layer.load_state_dict(ckpt_dict['extractor_ll'])
        else: 
            # old ckpt mode used to store everything
            self.policy.load_state_dict({
                key: ckpt_dict[f'net.policy.{key}']
                for key in self.policy.state_dict().keys()
            })

            self.extractor.last_linear_layer.load_state_dict({
                key: ckpt_dict[f'net.extractor.last_linear_layer.{key}']
                for key in self.extractor.last_linear_layer.state_dict().keys()
            })

        print(f"Checkpoint loaded: {path}")

# ----------------- CSV Models -----------------
class CSVNet:
    def __init__(self, csv_path):
        self._output_names = ['vx', 'vy', 'vz', 'yaw', 'stop']
        self.df = pd.read_csv(csv_path)
        self.cur_idx = 0

    def __call__(self, x=None):
        row = self.df.iloc[self.cur_idx]
        out = {k: torch.tensor([row[k]], dtype=torch.float32) for k in self._output_names if k in row}
        out['stop'] = torch.log(out.get('stop', torch.tensor([0.2])))
        self.cur_idx += 1
        if self.cur_idx == len(self.df):
            out['stop'] = torch.tensor([10.0], dtype=torch.float32)
        return out
    
class CSVNet2:
    def __init__(self, csv_path):
        self.out_keys = ['vx', 'vy', 'vz', 'yaw']
        self.df = pd.read_csv(csv_path)
        self.cur_idx = 0

    def __call__(self, x=None):
        row = self.df.iloc[self.cur_idx]
        out = {k: torch.tensor([row[i]], dtype=torch.float32) for i, k in enumerate(self.out_keys)}
        out['stop'] = torch.log(out.get('stop', torch.tensor([0.2])))
        self.cur_idx += 1
        if self.cur_idx == len(self.df):
            out['stop'] = torch.tensor([10.0], dtype=torch.float32)
        return out


# ----------------- LOSS FUNCTION -----------------

# ----------------- TRAINING FUNCTION -----------------
def train(model, train_loader, writer, config:TrainingConfig, ckpt_path):
    #TODO recently there are few changes, need to modify here before using
    device = config.device
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def get_loss(pred, y, i, writer):
        loss = {'total': 0}
        for key, value in y.items():
            loss[key] = nn.functional.mse_loss(pred[key], value.to(device), reduction='mean')
            loss['total'] += loss[key]/len(y)
            writer.add_scalar(f"Loss/{key}", loss[key].item(), i)
        writer.add_scalar(f"Loss/total", loss['total'].item(), i)
        return loss

    for epoch in range(config.num_epochs):
        total_loss = 0
        loop = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for i, data in loop:
            x, y = data
            pred = model(x)
            loss = get_loss(pred, y, i, writer)

            optim.zero_grad()
            loss['total'].backward()
            optim.step()

            total_loss += loss['total'].item()
            average_loss = total_loss/(i+1)
            loop.set_postfix(average_loss=average_loss)
            writer.add_scalar(f"Loss/running_average", average_loss, i)
            loop.set_postfix(loss=loss['total'].item())

            if i % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(ckpt_path, f"policy_{i}.ckpt")
                model.save_policy(checkpoint_path)
                # print(f"Checkpoint saved: {checkpoint_path}")

# ----------------- Evaluation FUNCTION -----------------
def evaluate(model, eval_loader, max_eval_num=10, prefix=''):
    runs = []
    cur_path = ExperimentConfig.log_dir
    num_eval = 0
    plot_toggle = False
    # losses = {'vx': [], 'vy': [], 'vz': [], 'yaw': [], 'total': []}
    vals = {'vx': [], 'vy': [], 'vz': [], 'yaw': [], 
            'vx_y': [], 'vy_y': [], 'vz_y': [], 'yaw_y': [], 
            'stop':[], 'stop_y':[]}
    with torch.no_grad():
        loop = tqdm(enumerate(eval_loader), desc="Evaluating")
        for i, data in loop:
            if num_eval >= max_eval_num: break
            x, y = data
            if y['stop'].sum()>0 and not plot_toggle:
                runs.append(eval_loader.dataset.cur_run)
                cur_path = os.path.join(ExperimentConfig.log_dir, prefix+runs[-1])
                os.makedirs(cur_path, exist_ok=True)
                plot_toggle = True
            if plot_toggle and y['stop'].sum() == 0:
                num_eval += 1
                plot_toggle = False
                ax, fig = plt.subplots(3, 2, figsize=(15, 10))
                for j, key in enumerate(['vx', 'vy', 'vz', 'yaw', 'stop']):
                    plt.subplot(3, 2, j+1)
                    plt.plot(vals[key], label='pred')
                    plt.plot(vals[key+'_y'], label='gt')
                    plt.legend()
                    plt.title(key)
                plt.savefig(os.path.join(cur_path, "eval_vel_cmds.png"))
                df = pd.DataFrame(vals)
                df.to_csv(os.path.join(cur_path, "eval_vel_cmds.csv"), index=False)

                for key in vals:
                    vals[key] = []
            
            pred = model(x)
            pred['stop'] = torch.sigmoid(pred.get('stop', torch.tensor([-1.0])))
            for key, value in pred.items():
                vals[key].append(value.cpu().item())
                vals[key+'_y'].append(y[key].cpu().item())

    ExperimentConfig.meta_data = '\n'.join(runs)
    return runs
# ----------------- Infer Function -----------------
def infer(model, init_cond, text_cmd, prefix=''):
    """
    Runs inference on a policy model and controls a simulator to follow the instructions.

    Args:
        model: The policy model to run inference on.
        infer_path: The path to save the output images and plots.
        infer_cfg: The config for the inference, including the environment name, instruction text,
            and max steps to run.
    """
    from MCMD_Sim.simulator.simulator_mcmd_exp import MCMDSimEval
    sim = MCMDSimEval(init_cond, 3, log_prefix=prefix)

    image = sim.setup_simulation()[0]
    pred_stops = []
    with torch.no_grad():
        for iter in tqdm(range(InferenceConfig.max_step)):
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')

            preds = model({"image": img, "text": text_cmd})
            pred_stop = torch.sigmoid(preds.get('stop', torch.tensor([-1.0])))
            pred_stops.append(pred_stop[0].cpu().item())

            vel_cmd = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            image = sim.step_action(vel_cmd)
            image = image[0]

            if pred_stop > InferenceConfig.stop_thresh and iter > 30: break

    logger = sim.close()

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(pred_stops)
    plt.title('Stop Predition')
    plt.subplot(2, 1, 2)
    plt.plot(np.log(np.array(pred_stops)))
    plt.title('Stop Predition Log Scale')
    plt.savefig(os.path.join(logger.log_dir, 'pred_stop.jpg'))
    return logger.log_dir

def infer2(model, init_cond, text_cmd, prefix=''):
    import sys
    sys.path.append('/home/alex/flex/gym_pybullet_drones')
    sys.path.append('/home/alex/flex/gym_pybullet_drones/gym_pybullet_drones')
    sys.path.append('/home/alex/flex/gym_pybullet_drones/gym_pybullet_drones/examples')
    from simulator_eval import EvalSimulator # type: ignore
    from simulator_utils import get_x_y_z_yaw_relative_to_base_env # type: ignore

    init_cond['log_dir'] = os.path.join(init_cond['log_dir'], prefix[3:])
    init_cond['start_heights'] = init_cond['drones_loc'][0][2]
    init_cond['target_heights'] = init_cond['objects_loc'][init_cond['target_idx']][2]
    init_cond['objects_color'] = [f'{colr} {obj_type}' for colr, obj_type in zip(init_cond['objects_color'], init_cond['objects_type'])]
    init_cond['objects_relative'] = [(loc[0], loc[1]) for loc in init_cond['objects_loc']]
    init_cond['start_dist'] = np.linalg.norm(np.array(init_cond['drones_loc'][0]) - np.array(init_cond['objects_loc'][init_cond['target_idx']]))

    sim = EvalSimulator(init_cond['log_dir'], init_cond, 3, init_cond['target_idx'], init_cond['env_name'])

    sim.setup_simulation()

    vel_cmd = np.array([0,0,0,0])
    sim.vel_cmd_world = vel_cmd
    updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
    updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
    pybullet_img = pybullet_img[None, :, :, 0:3]

    init_forward = updated_position[0]
    unnormalized_cmds = []
    text = text_cmd
    pred_stops = []
    with torch.no_grad():
        for iter in tqdm(range(InferenceConfig.max_step)):
            image = copy.deepcopy(pybullet_img)
            image = image.squeeze(0)
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')

            # run inference
            preds = model({"image": img, "text": text})
            out = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            vel_cmd = out[0]
            pred_stop = torch.sigmoid(preds['stop'])

            pred_stops.append(pred_stop[0].cpu().numpy())
            unnormalized_cmds.append([*vel_cmd, pred_stop[0].cpu().numpy().item()])

            updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
            if finished or pred_stop > InferenceConfig.stop_thresh:
                break
            pybullet_img = pybullet_img[None, :, :, 0:3]

            updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
            updated_position -= [init_forward, 0, 0.6, sim.theta_environment]


    sim.export_plots()

    log_dir = init_cond['log_dir']

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(pred_stops)
    plt.title('Stop Predition')
    plt.subplot(2, 1, 2)
    plt.plot(np.log(np.array(pred_stops)))
    plt.title('Stop Predition Log Scale')
    plt.savefig(os.path.join(log_dir, 'pred_stop.jpg'))
    return init_cond['log_dir']

# ----------------- MAIN EXECUTION -----------------
if __name__ == '__main__':
    config = ExperimentConfig() # Initialize Configuration
    prefix = ckpt.prefix 
    # Load DataLoader, Policy and Model
    if mode in {'train', 'eval', 'infer_eval'}:
        dataset = FlightILDataset(
            [DataConfig.data_path], train= mode == 'train', shuffle=DataConfig.shuffle, 
            snippet_size=DataConfig.snippet_size, seq_length=DataConfig.seq_length, 
            stride=DataConfig.stride, load_features_directly=True
        )
        # dataset.run_types = ['right_blue']
        # dataset.run_types_wts = [1.0]   
        data_loader = DataLoader(dataset, batch_size=DataConfig.batch_size, num_workers=DataConfig.num_workers)

    policy = VITPolicy(cfg = config.policy_cfg) if config.policy_name == 'VIT' else LSTMPolicy(config.policy_cfg)
    model = E2ENet(policy, config.device, feature_extraction=True)
    if config.checkpoint_path is not None:
        model.load_checkpoint(config.checkpoint_path, config.new_ckpt_mode)
    # set_seed(config.train_cfg.seed_value)
    model.eval()
        
    if config.mode == 'train':
        tensorboard_path = os.path.join(config.log_dir, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_path)
        model.train()
        train(model, data_loader, writer, config.train_cfg, config.log_dir)
        writer.close()
    elif config.mode == 'infer':
        from MCMD_Sim.simulator.utils import generate_closed_loop_1drone_2ball_env_init, generate_instruction
        for _ in range(30):
            sim_objs = random.sample(['red ball', 'colorless rocket', 'colorless jeep', 'colorless dog', 'green ball'], 2)
            command = random.choice(['above', 'below',])
            target_idx = random.choice([0, 1])
            init_cond = generate_closed_loop_1drone_2ball_env_init(
                env_name=InferenceConfig.env_name,
                command=command,
                target_idx=target_idx,
                objs = sim_objs,
                log_path=os.path.join(config.log_dir),
                data_path=None
            )
            target, obj_type = sim_objs[target_idx].split(" ")
            text = generate_instruction(target, command, obj_type = obj_type if target == 'colorless' else None)
            log_dir = infer(model, init_cond, text, prefix)
            ExperimentConfig.log_dir = log_dir
            with open(os.path.join(log_dir, "instruction_text.txt"), "w") as file:
                file.write(text)
    elif config.mode == 'infer2':
        from MCMD_Sim.simulator.utils import generate_closed_loop_1drone_2ball_env_init, generate_instruction

        sim_objs = [InferenceConfig.obj0, InferenceConfig.obj1]
        init_cond = generate_closed_loop_1drone_2ball_env_init(
            env_name=InferenceConfig.env_name,
            command=InferenceConfig.text_cmd,
            target_idx=InferenceConfig.target_idx,
            objs = sim_objs,
            log_path=os.path.join(config.log_dir),
            data_path=None
        )
        target, obj_type = sim_objs[InferenceConfig.target_idx].split(" ")
        text = generate_instruction(target, InferenceConfig.text_cmd, obj_type = obj_type if target == 'colorless' else None)

        log_dir = infer(model, init_cond, text, prefix)
        ExperimentConfig.log_dir = log_dir
        with open(os.path.join(log_dir, "instruction_text.txt"), "w") as file:
            file.write(text)

    elif config.mode == 'eval':
        evaluate(model, data_loader, prefix=prefix)

    elif config.mode == 'infer_eval':
        runs = evaluate(model, data_loader, prefix=prefix, max_eval_num=550)
        icond_base = '/home/alex/flex/MCMD_Sim/results'
        for run in runs:
            run_path = os.path.join(ExperimentConfig.log_dir, prefix+run)
            init_cond = json.load(open(
                os.path.join(icond_base, run, 'init_conditions.json'), 'r'
            ))
            init_cond['data_dir'] = None
            init_cond['log_dir'] = ExperimentConfig.log_dir
            # model = CSVNet(os.path.join(run_path, 'eval_vel_cmds.csv'))
            run_type = run.split('_')
            run_type = run_type[0] + '_' + run_type[1][1:]
            model = CSVNet2(os.path.join('/home/alex/flex/BLIP2_DATASET/train', run_type, run, 'data_out.csv'))
            text = '_'.join(run.split("_")[:2])
            infer(model, init_cond, text, prefix = 'ALL'+prefix+run)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")
    
    config.json_dump()



# def infer(model, infer_path, infer_cfg:InferenceConfig):
#     import sys
#     sys.path.append('/home/alex/flex/gym_pybullet_drones')
#     from gym_pybullet_drones.examples.simulator_utils import get_x_y_z_yaw_relative_to_base_env
#     from gym_pybullet_drones.examples.simulator_eval import EvalSimulator
#     from MCMD_Sim.utils import generate_init_conditions_closed_loop_inference_2choice
    
#     init_cond = generate_init_conditions_closed_loop_inference_2choice(
#         [infer_cfg.obj0, infer_cfg.obj1],
#         1,
#         [infer_path]
#     )
#     sim = EvalSimulator(infer_path, init_cond, 3, '0', infer_cfg.env_name)
#     sim.setup_simulation()

#     vel_cmd = np.array([0,0,0,0])
#     sim.vel_cmd_world = vel_cmd
#     updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
#     updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
#     pybullet_img = pybullet_img[None, :, :, 0:3]

#     init_forward = updated_position[0]
#     unnormalized_cmds = []
#     text = infer_cfg.text_cmd
#     pred_stops = []
#     with torch.no_grad():
#         for _ in tqdm(range(infer_cfg.max_step)):
#             image = copy.deepcopy(pybullet_img)
#             image = image.squeeze(0)
#             image = Image.fromarray(image)
#             img = image.resize((224, 224))
#             img = transforms.ToTensor()(img).to('cuda:0')

#             # run inference
#             preds = model({"image": img, "text": text})
#             out = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
#             vel_cmd = out[0]
#             pred_stop = torch.sigmoid(preds['stop'])

#             pred_stops.append(pred_stop[0].cpu().numpy())
#             unnormalized_cmds.append([*vel_cmd, pred_stop[0].cpu().numpy().item()])

#             updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
#             if finished or pred_stop > infer_cfg.stop_thresh:
#                 break
#             pybullet_img = pybullet_img[None, :, :, 0:3]

#             updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
#             updated_position -= [init_forward, 0, 0.6, sim.theta_environment]

#     with open(os.path.join(infer_path,"instruction_text.txt"), "w") as file:
#         file.write(text)
#     fig, ax = plt.subplots(2, 1, figsize=(15, 10))
#     plt.subplot(2, 1, 1)
#     plt.plot(pred_stops)
#     plt.title('Stop Predition')
#     plt.subplot(2, 1, 2)
#     plt.plot(np.log(np.array(pred_stops)))
#     plt.title('Stop Predition Log Scale')
#     plt.savefig(os.path.join(infer_path, 'pred_stop.jpg'))
#     sim.export_plots()
#     np.savetxt(os.path.join(infer_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_cmds), delimiter=",")