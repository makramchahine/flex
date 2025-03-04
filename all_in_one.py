import os
import torch
import random
import numpy as np
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
from learning.src.data.components.flight_il_dataset_clean import FlightILDataset
from dataclasses import dataclass, asdict
from torchvision import transforms

# ----------------- CONFIGURATION CLASS -----------------



@dataclass
class PolicyConfig:
    channels:int = 64
    reduced_dim:int = 8
    spatial_dim:int = 8
    seq_length:int = 32
    hidden_dim:int = 128
    num_layers:int = 1
    num_classes:int = 4
    single_step:bool = False
    dropout:float = 0.5


@dataclass
class DataConfig:
    train_data_path:str = '/home/alex/flex/BLIP2_DATASET/train/'
    eval_data_path:str = '/home/alex/flex/BLIP2_DATASET/eval/'
    snippet_size:int = 100
    seq_length:int = 32
    stride:int = 5
    batch_size:int = 32
    num_workers:int = 1
    shuffle:bool = True

@dataclass
class TrainingConfig:
    device:str = torch.device('cuda:0')
    learning_rate:int = 1e-3
    checkpoint_interval:int = 600
    seed_value:int = 42
    num_epochs:int = 1000

@dataclass
class ExperimentConfig:
    policy_cfg:PolicyConfig = PolicyConfig()
    train_cfg:TrainingConfig = TrainingConfig()
    data_cfg:DataConfig = DataConfig()
    mode:str = 'infer' # train, eval, infer
    log_dir:str = f"zcheckpoints/{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    checkpoint_path:str = '/home/alex/flex/local/train_flight/2025-03-03/13-12-23/checkpoints/step_075000.ckpt'
    ckpt_has_policy_only:bool = False

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.policy_cfg.single_step = self.mode != 'train'

    def json_dump(self):
        self.train_cfg.device = str(self.train_cfg.device)
        with open(os.path.join(self.log_dir, 'experiment_config.json'), "w") as f:
            json.dump(asdict(self), f, indent=4)
        self.train_cfg.device = torch.device(self.train_cfg.device)

# ----------------- SEED FUNCTION -----------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# ----------------- DATA LOADER FUNCTION -----------------
def get_data_loader(config:DataConfig, mode_train=True):
    data_path = config.train_data_path if mode_train else config.eval_data_path    
    dataset = FlightILDataset(
        [data_path], train=mode_train, shuffle=config.shuffle, 
        snippet_size=config.snippet_size,
        seq_length=config.seq_length, 
        stride=config.stride
    )
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)

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
            use_visual_encoder_only= False
        )
        self.policy = policy
        self._output_names = ['vx', 'vy', 'vz', 'yaw']
        self.device = device
        self.to(device)

    def forward(self, x):
        x = {'image': x['image'].to(self.device), 'text':x['text']}
        z = self.extractor(x)
        out, pred_stop = self.policy(z)

        out_dim = out.shape[-1]
        out = {k: out[...,i] for i, k in enumerate(self._output_names[:out_dim])}

        return out, pred_stop
    
    def to(self, device):
        self.extractor.to(device)
        self.policy.to(device)
        return super().to(device)
    
    def train(self):
        self.extractor.train()
        self.policy.train()
    
    def eval(self):
        self.extractor.eval()
        self.policy.eval()  
    
    def save_policy(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_checkpoint(self, path, has_policy_only=False):
        if has_policy_only:
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
        else:
            policy_torch = torch.load(path, map_location=self.device)
            policy_dict = {}
            for key in self.policy.state_dict().keys():
                policy_dict[key] = policy_torch['state_dict'][f'net.policy.{key}']
            self.policy.load_state_dict(policy_dict)

            extract_ll = {}
            for key in self.extractor.last_linear_layer.state_dict().keys():
                extract_ll[key] = policy_torch['state_dict'][f'net.extractor.last_linear_layer.{key}']
            self.extractor.last_linear_layer.load_state_dict(extract_ll)

        print(f"Checkpoint loaded: {path}")

# ----------------- LOSS FUNCTION -----------------

# ----------------- TRAINING FUNCTION -----------------
def train(model, train_loader, writer, config:TrainingConfig, ckpt_path):
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
def evaluate(model, eval_loader, eval_path, config:TrainingConfig, max_eval_num=10):
    device = config.device
    model.eval()
    num_eval = 0

    # losses = {'vx': [], 'vy': [], 'vz': [], 'yaw': [], 'total': []}
    vals = {'vx': [], 'vy': [], 'vz': [], 'yaw': [], 
            'vx_y': [], 'vy_y': [], 'vz_y': [], 'yaw_y': [], 
            'stop':[], 'stop_y':[]}
    with torch.no_grad():
        loop = tqdm(enumerate(eval_loader), desc="Evaluating")
        for i, data in loop:
            if num_eval >= max_eval_num: break
            x, y = data
            pred, pred_stop = model(x)
            vals['stop'].append(torch.sigmoid(pred_stop)[0].cpu().numpy())
            vals['stop_y'].append(x['is_last'][0].cpu().numpy())
            for key, value in pred.items():
                vals[key].append(value.cpu().numpy())
                vals[key+'_y'].append(y[key].cpu().numpy())

            if x['is_last'].sum()>0:
                num_eval += 1
                ax, fig = plt.subplots(3, 2, figsize=(15, 10))
                for j, key in enumerate(['vx', 'vy', 'vz', 'yaw', 'stop']):
                    plt.subplot(3, 2, j+1)
                    plt.plot(vals[key], label='pred')
                    plt.plot(vals[key+'_y'], label='gt')
                    plt.legend()
                    plt.title(key)
                plt.savefig(os.path.join(eval_path, f"eval_{i}.png"))

                for key in vals:
                    vals[key] = []
# ----------------- Infer Function -----------------
def infer(model, infer_path):
    import sys
    sys.path.append('/home/alex/flex/gym-pybullet-drones')
    from gym_pybullet_drones.examples.simulator_utils import get_x_y_z_yaw_relative_to_base_env
    from gym_pybullet_drones.examples.simulator_eval import EvalSimulator
    from data_collection.utils import generate_init_conditions_closed_loop_inference_2choice
    
    init_cond = generate_init_conditions_closed_loop_inference_2choice(
        ['red ball', 'blue ball'],
        1,
        [infer_path]
    )
    sim = EvalSimulator(infer_path, init_cond, 3, '0', 'arena')
    sim.setup_simulation()

    vel_cmd = np.array([0,0,0,0])
    sim.vel_cmd_world = vel_cmd
    updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
    updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
    pybullet_img = pybullet_img[None, :, :, 0:3]

    init_forward = updated_position[0]
    unnormalized_vel_cmds = []
    text = 'Venture towards the blue ball in a straight line.'
    pred_stops = []
    with torch.no_grad():
        for _ in tqdm(range(200)):
            image = copy.deepcopy(pybullet_img)
            image = image.squeeze(0)
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')

            # run inference
            preds, pred_stop = model({"image": img, "text": text})
            out = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            pred_stop = torch.sigmoid(pred_stop)

            print(pred_stop, "should i stop pred?")
            pred_stops.append(pred_stop[0].cpu().numpy())
            unnormalized_vel_cmds.append(out[0])
            vel_cmd = out[0]
            updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
            if finished or pred_stop > 0.4:
                break
            pybullet_img = pybullet_img[None, :, :, 0:3]

            updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
            updated_position -= [init_forward, 0, 0.6, sim.theta_environment]

    with open(os.path.join(infer_path,"instruction_text.txt"), "w") as file:
        file.write(text)
    plt.plot(pred_stops)
    plt.title('Stop Predition')
    plt.savefig(os.path.join(infer_path, 'pred_stop.jpg'))
    sim.export_plots()
    np.savetxt(os.path.join(infer_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")

# ----------------- MAIN EXECUTION -----------------
if __name__ == '__main__':
    # Initialize Configuration
    config = ExperimentConfig()
    config.json_dump()
    # torch.save(config, os.path.join(config.log_dir, "experiment_config.pth"))
    set_seed(config.train_cfg.seed_value)
    
    # Load Policy and Model
    policy = LSTMPolicy(config.policy_cfg)
    model = E2ENet(policy, config.train_cfg.device)
    if config.checkpoint_path is not None:
        model.load_checkpoint(config.checkpoint_path, config.ckpt_has_policy_only)

    if config.mode == 'train':
        tensorboard_path = os.path.join(config.log_dir, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_path)
        model.train()
        train_loader = get_data_loader(config.data_cfg)
        train(model, train_loader, writer, config.train_cfg, config.log_dir)
        writer.close()
    elif config.mode == 'infer':
        model.eval()
        infer_path = os.path.join(config.log_dir, 'infer')
        os.makedirs(infer_path, exist_ok=True)
        infer(model, infer_path)
    else:
        model.eval()
        config.data_cfg.seq_length = 1
        config.data_cfg.stride = 1
        config.data_cfg.batch_size = 1
        config.data_cfg.shuffle = False
        eval_loader = get_data_loader(config.data_cfg, mode_train=False)
        eval_path = os.path.join(config.log_dir, 'eval')
        os.makedirs(eval_path, exist_ok=True)
        evaluate(model, eval_loader, eval_path, config.train_cfg)