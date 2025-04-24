import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from typing import List, Dict
from omegaconf import OmegaConf # type: ignore
from torchvision import transforms # type: ignore
import pandas as pd
import argparse
from datetime import datetime

from learning.src.models.components.policies.svit import SViTPolicy as svit
from learning.src.models.components.policies.svit2 import SViTPolicy as svit2
from learning.src.models.components.policies.svit3 import SViTPolicy as svit3
from learning.src.models.components.policies.stop_policies import select_stop_flagger
from learning.src.models.components.extractors.blip import BLIPExtractor

from MCMD_Sim.simulator.simulator_mcmd_exp import MCMDSimEval
from MCMD_Sim.simulator.utils.tasks import get_task_delta
from MCMD_Sim.simulator.utils.gen_init_cond import generate_closed_loop_1drone_nobject_env_init

Task_set = {
    '2rb': (['red ball', 'blue ball'], 2),
    '2colors': (['red ball', 'blue ball', 'yellow ball', 'green ball', 'purple ball'], 2),
    'r_shape': (['red ball', 'red cube', 'red pyramid'], 2),
    'm_shape': (["yellow ball", "green ball", "purple ball", "red cube", "blue cube", "yellow cube", "green cube", "purple cube", "red pyramid", "blue pyramid", "yellow pyramid", "green pyramid", "purple pyramid"], 2),
    'open_dict': (["red cube", "blue pyramid", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"], 3),
}
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

class E2ENet(nn.Module):
    def __init__(self, policy_cfg, device, stop_flagger_cfg = None, feature_extraction = False, *args, **kwargs):
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
        policy_name = policy_cfg['_target_'].split('.')[-2]
        self.policy = self.select_policy(policy_name)(policy_cfg.cfg)
        self.stop_flagger = None
        if stop_flagger_cfg is not None:
            self.stop_flagger = select_stop_flagger(stop_flagger_cfg.name)(
                in_dim=stop_flagger_cfg.in_dim,
                hidden_dim=stop_flagger_cfg.hidden_dim,
                cfg = stop_flagger_cfg
            )
        self.output_names = ['vx', 'vy', 'vz', 'yaw', 'stop']
        self.device = device
        # self.ckpt_base = '/home/alex/flex/local/train_flight'
        self.to(device)

    def select_policy(self, policy_name):
        if policy_name == 'svit': return svit
        if policy_name == 'svit2': return svit2
        if policy_name == 'svit3': return svit3
    
    def get_obj_dir_text(self, texts: List[str], meta = None) -> Dict[str, List[str]]:
        dirs, objs = [], []
        if meta is None: meta = texts[::]
        for i, text in enumerate(texts):
            parts = text.split('---')
            if len(parts) != 2: raise ValueError(f"Expected format 'dir---obj', but got: {text} {texts[i]} with meta {meta[i]} ")
            dirs.append(parts[0])
            objs.append(parts[1])
        return {'dir': dirs, 'obj': objs}

    def forward(self, x):
        txt_ = self.get_obj_dir_text(x['text'])
        x = {'image': x['image'], 'text':x['text']}
        z = self.extractor({'image': x['image'], 'text': txt_['obj']})
        out = self.policy(z, txt_['dir'])
        if self.stop_flagger is not None:
            stop = self.stop_flagger(self.policy.get_last_rep())
            out = torch.cat((out, stop), dim=1)
        out_dim = out.shape[-1]
        out = {k: out[...,i] for i, k in enumerate(self.output_names[:out_dim])}

        return out

    def load_checkpoint(self, cpath):
        # cpath = os.path.join(self.ckpt_base, ckpt_date, ckpt_time, 'checkpoints', f'step_{ckpt_step}.ckpt')
        ckpt_dict = torch.load(cpath, map_location=self.device, weights_only=False)
        ckpt_dict = ckpt_dict['state_dict']
        self.policy.load_state_dict(ckpt_dict['policy'])
        self.extractor.last_linear_layer.load_state_dict(ckpt_dict['extractor_ll'])
        if self.stop_flagger is not None and "stop_flagger" in ckpt_dict.keys():
            self.stop_flagger.load_state_dict(ckpt_dict['stop_flagger'])
        print(f"Checkpoint loaded: {cpath}")

def infer(model, init_cond, text_cmd, prefix='', num_step=100, msg = '', logging=False, verbose=False):
    sim = MCMDSimEval(init_cond, 3, log_prefix=prefix, logging=logging)
    image = sim.setup_simulation(verbose=verbose)[0]
    stop_vals = ''
    with torch.no_grad():
        for iter in tqdm(range(num_step), desc=msg):
            image = Image.fromarray(image)
            img = image.resize((224, 224))
            img = transforms.ToTensor()(img).to('cuda:0')

            preds = model({"image": img, "text": text_cmd})
            if 'stop' in preds:
                stop_val = torch.sigmoid(preds['stop'])[0].cpu().item()
                stop_vals += f'{stop_val:.3f}, '
                if iter > 25 and stop_val > 0.5: break

            vel_cmd = torch.stack([preds["vx"], preds["vy"], preds["vz"], preds["yaw"]], dim=1).cpu().detach().numpy()
            image = sim.step_action(vel_cmd, save_image=logging)
            image = image[0]
    sim.logger.log_text(stop_vals)
    return sim.close(save=logging)


def eval_one(model, init_cond, txt_cmd, ckpt, mode, task='2rb', max_step=100, logging=False):
    logger = infer(model, init_cond, [txt_cmd], logging=logging, num_step=max_step)
    exp_vec = get_task_delta(init_cond['command'])
    ref_point = np.array(init_cond["objects_loc"][init_cond["target_idx"]])
    score, cos_sim, dist = trajectory_score(np.array(logger.global_pos_array[0])[:, :3], exp_vec+ref_point, ref_point)
    return {
        'ckpt': ckpt,
        'env_name': init_cond['env_name'],
        'command': init_cond['command'],
        'tidx': init_cond["target_idx"],
        'score': score,
        'cos_sim': cos_sim,
        'dist': dist,
        'run': logger.log_dir
    }

def eval_data(model, ckpt_base, ckpts, mode='train', num_exp = 5):
    result = []
    data_path = f'/home/alex/flex/BLIP2_DATASET/{mode}'
    init_base = '/home/alex/flex/MCMD_Sim/results_accepted'
    for ckpt in tqdm(ckpts, desc='Evaluating'):
        cpath = os.path.join(ckpt_base, ckpt)
        model.load_checkpoint(cpath)
        for run_type in os.listdir(data_path):
            if run_type.startswith('save'): continue
            runs = os.listdir(os.path.join(data_path, run_type))
            random.shuffle(runs)
            for run in runs[:num_exp]:
                max_step = 10 + len(os.listdir(os.path.join(data_path, run_type, run)))
                init_cond = json.load(open(os.path.join(init_base, run, 'init_conditions.json'), 'r'))
                with open(os.path.join(data_path, run_type, run, 'label.txt'), 'r') as f:
                    txt_cmd = f.read().strip()
                result_one = eval_one(model, init_cond, txt_cmd, ckpt, mode=mode, max_step=max_step)
                result_one['run'] = run
                result.append(result_one)
    df = pd.DataFrame(result)
    df.to_csv(os.path.join(ckpt_base, '..', f'{mode}.csv'), index=False)
    return df


def model_evaluation(cfg_date, cfg_time, device):
    cfg_base = '/home/alex/flex/local/train_flight'
    cfg_path = os.path.join(cfg_base, cfg_date, cfg_time)
    config = OmegaConf.load(os.path.join(cfg_path, 'config')).model.net
    sfc = config.get('stop_flagger_cfg', None)
    model = E2ENet(config.policy, device, stop_flagger_cfg= sfc, feature_extraction=True)
    ckpt_base = os.path.join(cfg_path, 'checkpoints')
    ckpts = sorted([f for f in os.listdir(ckpt_base) if f.endswith('.ckpt') and f.startswith('step_')])
    ckpts = ckpts[4:] # skipping the first 4 checkpoints
    #-----------------Evaluation on Training Data-----------------#
    train_df = eval_data(model, ckpt_base, ckpts, mode='train', num_exp = 1)
    train_means = train_df[['ckpt', 'score']].groupby('ckpt').mean().reset_index()
    ckpts = train_means.sort_values(by='score', ascending=False).head(20)['ckpt'].tolist()
    #-----------------Evaluation on Training Data-----------------#
    eval_df = eval_data(model, ckpt_base, ckpts, mode='eval', num_exp = 2)
    eval_means = eval_df[['ckpt', 'score']].groupby('ckpt').mean().reset_index()
    ckpts = eval_means.sort_values(by='score', ascending=False).head(3)['ckpt'].tolist()
    #-----------------Evaluation on Inference-----------------#
    result = []
    cur_dt = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    for ckpt in tqdm(ckpts, desc='Evaluating'):
        cpath = os.path.join(ckpt_base, ckpt)
        model.load_checkpoint(cpath)
        for task in Task_set:
            objs, num_obj = Task_set[task]
            for command in ['above', 'below', 'left', 'right', 'towards']:
                for env_name in ['samurai', 'arena']:
                    num_infer_exp = 1 if task == '2rb' else 3
                    for _ in range(num_infer_exp):
                        objs = random.sample(objs, num_obj)
                        init_cond = generate_closed_loop_1drone_nobject_env_init(objs, 
                                                        env_name=env_name,
                                                        command=command,
                                                        log_path= os.path.join('/home/alex/flex/results_test', cur_dt))
                        txt_cmd = f'{command}---{objs[init_cond["target_idx"]]}'
                        result_one = eval_one(model, init_cond, txt_cmd, ckpt, task=task, mode='infer', logging=True)
                        result.append(result_one)
    result = pd.DataFrame(result)
    result.to_csv(os.path.join(cfg_path, 'result_infer.csv'), index=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2023-08-23')
    parser.add_argument('--time', type=str, default='23-54')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model_evaluation(args.date, args.time, device)
