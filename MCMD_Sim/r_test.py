import os
from simulator import MCMDSimSampler
from simulator.utils import generate_closed_loop_1drone_2ball_env_init
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random

if __name__ == '__main__':
    all_inst = ''
    for _ in tqdm(range(75)):
        data_path = f'/home/alex/flex/BLIP2_DATASET/{"train" if random.random() < 0.9 else "eval"}/left_blue'
        init_cond = generate_closed_loop_1drone_2ball_env_init(
            env_name='samurai', 
            data_path=data_path,
            command='left',
            target_idx=1,
            objs= ['red ball', 'blue ball']
        )
        # print(init_cond, 'init_cond')

        sim = MCMDSimSampler(init_cond, 3)
        inst = sim.run_simulation_to_completion(random_walk=True)
        # all_inst += inst

        # im_path = os.path.join(log_path, "pybullet_pics")

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(os.path.join(log_path, f'out_video_{log_path[-13:]}.mp4'), fourcc, 20, (224, 224))
        # images = [f for f in os.listdir(im_path) if f.endswith('.png')]
        # images.sort()
        # for i, image in enumerate(images):
        #     if i < 8000:
        #         image = os.path.join(im_path, image)
        #         img = Image.open(image)
        #         img = img.convert('RGB')
        #         img = img.resize((224, 224))
        #         img = np.array(img)
        #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #         out.write(img)

        # out.release()

    # with open('r_test_inst.txt', 'w') as f:
    #     f.write(all_inst)


