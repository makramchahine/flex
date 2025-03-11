import os
from simulator import MCMDSimulator
from simulator.utils import generate_closed_loop_1drone_2ball_env_init
import numpy as np
import cv2
from PIL import Image

if __name__ == '__main__':
    init_cond = generate_closed_loop_1drone_2ball_env_init(command='to')
    print(init_cond, 'init_cond')

    sim = MCMDSimulator(init_cond, 3)
    log_path = sim.run_simulation_to_completion()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(log_path, 'out_video.mp4'), fourcc, 20, (224, 224))

    imdir = os.path.join(log_path, "pybullet_pics")
    images = os.listdir(imdir)
    images.sort()
    for i, image in enumerate(images):
        if i < 8000:
            image = os.path.join(imdir, image)
            img = Image.open(image)
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)

    out.release()


