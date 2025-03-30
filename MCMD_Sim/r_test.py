import os
from simulator import MCMDSimSampler
from simulator.utils import generate_closed_loop_1drone_2ball_env_init
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random

def generate_data(num_data, objs=None, data_path=None, command=None, target_idx=None):
    data_path_full = data_path
    for i in range(num_data):
        subdir = 'eval' if i % 10 == 4 else 'train'
        if data_path is not None: data_path_full = data_path.replace('REPLACE', subdir)
        init_cond = generate_closed_loop_1drone_2ball_env_init(
                        env_name='samurai', 
                        data_path=data_path_full,
                        command=command,
                        target_idx=target_idx,
                        objs=objs
                    )
        sim = MCMDSimSampler(init_cond, 3)
        sim.run_simulation_to_completion(random_walk=True)


if __name__ == '__main__':
    all_inst = ''
    for task in ['left', 'right']:
        for objs in [['blue ball', 'red ball'], ['red ball', 'blue ball']]:
            for target_idx in [0, 1]:
                obj = objs[target_idx].split(' ')[0]
                data_path = f'/home/alex/flex/BLIP2_DATASET/REPLACE/{task}_{obj}'
                generate_data(75, objs=objs, data_path=data_path, command=task, target_idx=target_idx)
                    
