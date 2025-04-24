import os
from simulator import MCMDSimSampler
from simulator.utils import generate_closed_loop_1drone_2ball_env_init
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import torch
from torchvision import transforms # type: ignore
import sys
sys.path.append('/home/alex/flex/learning/src/models/components/extractors')
from blip import BLIPExtractor # type: ignore

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def data2BLIP(model, data_path, target_path, device, run_types = None):
    if run_types is None: run_types = sorted(os.listdir(data_path))
    for run_type in run_types:
        runs = sorted(os.listdir(os.path.join(data_path, run_type)))
        for i, run in tqdm(enumerate(runs), leave=True, total=len(runs), desc=run_type):
            target_path_run = os.path.join(target_path, run_type, run)
            os.makedirs(target_path_run, exist_ok=True)

            run = os.path.join(data_path, run_type, run)
            with open(os.path.join(run,  'label.txt'), 'r') as f:
                text = f.read()
            text = text.split('---')[1]

            image_names = sorted([f for f in os.listdir(run) if f.endswith('.png')])
            run_loop = tqdm(enumerate(image_names), total=len(image_names), desc=f"Run {i+1}/{len(runs)}", leave=False)
            for j, im_name in run_loop:
                img = Image.open(os.path.join(run, im_name))
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img = transforms.ToTensor()(img)
                data_out = model({"image": img.to(device), "text":text})
                torch.save(data_out.cpu(), os.path.join(target_path_run, im_name.replace('.png', '.pt')))
                run_loop.set_postfix({'Image': im_name})

def run_extraction(run_types = None):
    data_path:str = '/home/alex/flex/BLIP2_DATASET/'
    target_dir:str = "/home/alex/flex/BLIP2_Features1/"
    device = torch.device('cuda:0')
    set_seed(42)

    model = BLIPExtractor(
            'blip2_feature_extractor',
            'pretrain',
            freeze_blip=True,
            use_low_dim_feature=False,
            last_linear_layer=None,      #features before last linear layer
            use_continuous_pe=False,
            stride=14,
            checkpoint='~/.cache/torch/hub/checkpoints/blip2_pretrained.pth',
            patch_size=1,
            all_q_dims= False,
            use_masked_patch_wise_feature= True,
            use_visual_encoder_only= False
        )

    model.to(device)
    model.eval()
    data2BLIP(model, data_path+'train/', target_dir+'train/', device, run_types)
    data2BLIP(model, data_path+'eval/', target_dir+'eval/', device, run_types)

def generate_data(num_data, objs=None, data_path=None, command=None, target_idx=None, pscale=0.0):
    data_path_full = data_path
    for i in tqdm(range(num_data), total=num_data, desc=f'Generating data: {command}{target_idx}'):
        subdir = 'eval5' if i % 10 == 4 else 'train5'
        if data_path is not None: data_path_full = data_path.replace('REPLACE', subdir)
        init_cond = generate_closed_loop_1drone_2ball_env_init(
                        env_name='samurai',
                        progress_scale=pscale,
                        data_path=data_path_full,
                        command=command,
                        target_idx=target_idx,
                        objs=objs
                    )
        sim = MCMDSimSampler(init_cond, 3)
        sim.run_simulation_to_completion(random_walk=True, verbose=False)


if __name__ == '__main__':
    # all_inst = ''
    # for task in ['left', 'right', 'above', 'below']:
    #     for objs in [['blue ball', 'red ball'], ['red ball', 'blue ball']]:
    #         for target_idx in [0, 1]:
    #             num_data = 75
    #             if task == 'left' and target_idx == 0 and objs[0] == 'blue ball': continue
    #             if task == 'left' and target_idx == 1 and objs[0] == 'blue ball': num_data = 35
    #             obj = objs[target_idx].split(' ')[0]
    #             data_path = f'/home/alex/flex/BLIP2_DATASET/REPLACE/{task}_{obj}_new'
    #             generate_data(num_data, objs=objs, data_path=data_path, command=task, target_idx=target_idx, pscale=0.0)
    run_extraction()
                    
