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
import argparse

# ----------------- CONFIGURATION CLASSES ---------------------------------------
mode = 'infer' # train, eval, infer, infer_eval
policy_name = 'VIT' # LSTM, VIT
ckpt_path = '/home/alex/flex/local/train_flight/2025-03-25/16-32-20/checkpoints/step_100000.ckpt'    #LSTM 2 layer
# ckpt_path = '/home/alex/flex/local/train_flight/2025-03-25/15-45-15/checkpoints/step_575000.ckpt'  #VIT 5 actions
# ckpt_path = '/home/alex/flex/local/train_flight/LSTM_from_tensor_8x8_flag/03-06-00-09-17/checkpoints/step_200000.ckpt' #LSTM 1 layer

@dataclass
class InferenceConfig:
    env_name:str = 'samurai' # arena, samurai
    obj0:str = 'red ball'
    obj1:str = 'blue ball'
    target_idx:int = 1
    text_cmd:str = 'right'
    max_step:int = 200
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
    num_classes:int = 5  #previously trained with 4

@dataclass
class LSTMConfig: 
    channels:int = 64
    reduced_dim:int = 8
    spatial_dim:int = 8
    seq_length:int = 32
    hidden_dim:int = 128
    num_layers:int = 2
    num_classes:int = 5
    single_step:bool = False
    dropout:float = 0.5


@dataclass
class DataConfig:
    data_path:str = 'BLIP2_DATASET/train/' if mode == 'train' else 'BLIP2_DATASET/eval/'
    snippet_size:int = 100
    seq_length:int = 32 if mode == 'train' else 1
    stride:int = 5 if mode == 'train' else 1
    batch_size:int = 32 if mode == 'train' else 1
    num_workers:int = 1
    shuffle:bool = mode == 'train'

@dataclass
class TrainingConfig:
    learning_rate:int = 1e-3
    checkpoint_interval:int = 600
    seed_value:int = 42
    num_epochs:int = 1000

@dataclass
class ExperimentConfig:
    policy_cfg:LSTMConfig = LSTMConfig() if policy_name == 'LSTM' else VITConfig
    train_cfg:TrainingConfig = TrainingConfig()
    data_cfg:DataConfig = DataConfig()
    infer_cfg:InferenceConfig = InferenceConfig()
    device:str = torch.device('cuda:0')

    policy_name:str = policy_name
    mode:str = mode
    log_dir:str = "local2"
    checkpoint_path:str = ckpt_path
    new_ckpt_mode:bool = True

    def __post_init__(self):
        if self.mode != 'infer': self.log_dir = f"local2/{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

    def json_dump(self, log_dir=None):
        if log_dir is None: log_dir = self.log_dir
        self.device = str(self.train_cfg.device)
        with open(os.path.join(log_dir, 'experiment_config.json'), "w") as f:
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

# ----------------- DATA LOADER FUNCTION -----------------
def get_data_loader():  
    dataset = FlightILDataset(
        [DataConfig.data_path], train= mode == 'train', shuffle=DataConfig.shuffle, 
        snippet_size=DataConfig.snippet_size, seq_length=DataConfig.seq_length, 
        stride=DataConfig.stride
    )   
    return DataLoader(dataset, batch_size=DataConfig.batch_size, num_workers=DataConfig.num_workers)

# ----------------- End to End Model -----------------
class E2ENet(nn.Module):
    def __init__(self, policy, device, *args, **kwargs):
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
            extract_features_flag= True
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

# ----------------- End to End Model -----------------
class CSVNet:
    def __init__(self, csv_path):
        self._output_names = ['vx', 'vy', 'vz', 'yaw', 'stop']
        self.df = pd.read_csv(csv_path)
        self.cur_idx = 0

    def __call__(self, x= None):
        out = self.df.iloc[self.cur_idx]
        out_dim = out.shape[-1]
        out = {k: out[...,i] for i, k in enumerate(self._output_names[:out_dim])}
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
def evaluate(model, eval_loader, eval_path, max_eval_num=10):
    model.eval()
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
            pred = model(x)
            pred['stop'] = torch.sigmoid(pred['stop'])
            for key, value in pred.items():
                vals[key].append(value.cpu().numpy())
                vals[key+'_y'].append(y[key].cpu().numpy())

            if y['stop'].sum()>0 and not plot_toggle:
                plot_toggle = True
            if plot_toggle and y['stop'].sum() == 0:
                num_eval += 1
                plot_toggle = False
                ax, fig = plt.subplots(3, 2, figsize=(15, 10))
                for j, key in enumerate(['vx', 'vy', 'vz', 'yaw', 'stop']):
                    plt.subplot(3, 2, j+1)
                    plt.plot(vals[key][:-1], label='pred')
                    plt.plot(vals[key+'_y'][:-1], label='gt')
                    plt.legend()
                    plt.title(key)
                plt.savefig(os.path.join(eval_path, f"eval_{i}.png"))

                for key in vals:
                    vals[key] = [vals[key][-1]]
# ----------------- Infer Function -----------------
def infer(model, init_cond, text_cmd):
    """
    Runs inference on a policy model and controls a simulator to follow the instructions.

    Args:
        model: The policy model to run inference on.
        infer_path: The path to save the output images and plots.
        infer_cfg: The config for the inference, including the environment name, instruction text,
            and max steps to run.
    """
    from MCMD_Sim.simulator.simulator_mcmd_exp import MCMDSimEval
    sim = MCMDSimEval(init_cond, 3, log_prefix=ExperimentConfig.policy_name)

    vel_cmd = np.array([[0,0,0,0]])
    image = sim.step_action(vel_cmd)[0]
    pred_stops = []
    with torch.no_grad():
        for iter in tqdm(range(InferenceConfig.max_step)):
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')
            # text = 'zoom in on the red ball'

            preds = model({"image": img, "text": text_cmd})
            pred_stop = torch.sigmoid(preds.get('stop', torch.tensor([-1.0])))
            pred_stops.append(pred_stop[0].cpu().numpy())

            vel_cmd = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            image = sim.step_action(vel_cmd)
            image = image[0]

            if pred_stop > InferenceConfig.stop_thresh: break

    log_dir = sim.close()

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(pred_stops)
    plt.title('Stop Predition')
    plt.subplot(2, 1, 2)
    plt.plot(np.log(np.array(pred_stops)))
    plt.title('Stop Predition Log Scale')
    plt.savefig(os.path.join(log_dir, 'pred_stop.jpg'))
    return log_dir

# ----------------- Infer Function -----------------
def infer_eval(infer_path, infer_cfg:InferenceConfig):
    """
    Runs inference on a policy model and controls a simulator to follow the instructions.

    Args:
        model: The policy model to run inference on.
        infer_path: The path to save the output images and plots.
        infer_cfg: The config for the inference, including the environment name, instruction text,
            and max steps to run.
    """
    from MCMD_Sim.simulator.simulator_mcmd_exp import MCMDSimEval
    import pandas as pd
    
    init_cond = json.load(open(infer_cfg.init_cond_path, 'r'))
    init_cond['data_path'] = None
    init_cond['log_path'] = infer_path
    with open(os.path.join(infer_cfg.eval_path, 'label.txt'), 'r') as f:
        text = int(f.read())

    vel_cv = pd.read_csv(os.path.join(infer_cfg.eval_path, 'vel_cmds.csv'))
    sim = MCMDSimEval(init_cond, 3)

    with torch.no_grad():
        for iter in tqdm(range(len(vel_cv))):
            vel_cmd = vel_cv.iloc[iter]
            _ = sim.step_action(vel_cmd)

    log_dir = sim.close()
    with open(os.path.join(log_dir, "instruction_text.txt"), "w") as file:
        file.write(text)

    return log_dir

# ----------------- MAIN EXECUTION -----------------
if __name__ == '__main__':
    # Initialize Configuration
    config = ExperimentConfig()
    # torch.save(config, os.path.join(config.log_dir, "experiment_config.pth"))
    set_seed(config.train_cfg.seed_value)
    if config.mode == 'infer_eval':
        log_dir = infer_eval(config.log_dir, config.infer_cfg)
        config.json_dump(log_dir)
    else:
        # Load Policy and Model
        policy = VITPolicy(cfg = config.policy_cfg) if config.policy_name == 'VIT' else LSTMPolicy(config.policy_cfg)
        model = E2ENet(policy, config.train_cfg.device)
        if config.checkpoint_path is not None:
            model.load_checkpoint(config.checkpoint_path, config.new_ckpt_mode)

        if config.mode == 'train':
            tensorboard_path = os.path.join(config.log_dir, 'tensorboard')
            os.makedirs(tensorboard_path, exist_ok=True)
            config.json_dump()
            writer = SummaryWriter(log_dir=tensorboard_path)
            model.train()
            train_loader = get_data_loader(config.data_cfg)
            train(model, train_loader, writer, config.train_cfg, config.log_dir)
            writer.close()
        elif config.mode == 'infer':
            from MCMD_Sim.simulator.utils import generate_closed_loop_1drone_2ball_env_init, generate_instruction

            sim_objs = [InferenceConfig.obj0, InferenceConfig.obj1]
            init_cond = generate_closed_loop_1drone_2ball_env_init(
                env_name=InferenceConfig.env_name,
                command=InferenceConfig.text_cmd,
                target_idx=InferenceConfig.target_idx,
                objs = sim_objs,
                log_path=os.path.join(config.log_dir, 'infer_log'),
                data_path=None
            )
            target = sim_objs[InferenceConfig.target_idx].split(" ")[0]
            text = generate_instruction(target, InferenceConfig.text_cmd)

            model.eval()
            log_dir = infer(model, init_cond, text)
            with open(os.path.join(log_dir, "instruction_text.txt"), "w") as file:
                file.write(text)
            config.json_dump(log_dir)
        else:
            eval_path = os.path.join(config.log_dir, 'eval')
            os.makedirs(eval_path, exist_ok=True)
            config.json_dump()
            model.eval()
            eval_loader = get_data_loader(config.data_cfg, mode_train=False)
            evaluate(model, eval_loader, eval_path, config.train_cfg)



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