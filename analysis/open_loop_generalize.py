import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import hydra
from lightning import LightningModule
from omegaconf import OmegaConf
import io

from assets.instructions import texts_dict

# Configurations and texts
cfgs = [
    # "/home/makramchahine/repos/flex/local/train_flight/simplevit_8x8/config",
    "/home/makramchahine/repos/flex/local/train_flight/simplevit_4x4_1/config"
    # "/home/makramchahine/repos/flex/local/train_flight/simplevit_2x2/config",
    # "/home/makramchahine/repos/flex/local/train_flight/linear/config"
    # "/home/makramchahine/repos/flex/local/train_flight/conv/config",
    # "/home/makramchahine/repos/flex/local/train_flight/linear/config",
    # "/home/makramchahine/repos/flex/local/train_flight/full_img_first_dim_trans/config",
    # "/home/makramchahine/repos/flex/local/train_flight/full_img_all_dim_trans/config",
    # "/home/makramchahine/repos/flex/local/train_flight/simplevit/config"
]

# Directories structure
root_dir = "/home/makramchahine/repos/fm_flight/analysis/activations_50_fix/run_0"
dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
# get rid of the directory that is called generated_paths
dirs = [d for d in dirs if "generated_paths" not in d]
# sort the directories by name alphabetically
dirs = sorted(dirs)

# Initialize output structure with keys corresponding to configuration parent directories
c_name = [os.path.basename(os.path.dirname(c)) for c in cfgs]
outputs = {cfg : {os.path.basename(d): {} for d in dirs} for cfg in c_name}

# Iterate over each configuration
for i, cfg_path in enumerate(cfgs):
    # Load model configuration
    cfg = OmegaConf.load(cfg_path)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    if cfg.ckpt_path:
        cpath = str(cfg.ckpt_path[0])
        if os.path.isfile(cpath):
            with open(cpath, 'rb') as f:
                buffer = io.BytesIO(f.read())
                ckpt = torch.load(buffer, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {cpath}")
        for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
            ckpt["state_dict"].pop(dropped_key, None)
        model.load_state_dict(ckpt["state_dict"])

    model = model.to(device)

    # Process each directory
    for directory in tqdm(dirs):
        dir_name = os.path.basename(directory)

        # Process images in pybullet_pics0
        images = sorted([img for img in os.listdir(os.path.join(directory, 'pybullet_pics0')) if img.endswith(".png")])
        # remove text before the first underscore
        didi = dir_name.split("_")[1:][0]

        outputs[c_name[i]][dir_name] = np.zeros((len(texts_dict[didi]), len(os.listdir(os.path.join(directory, 'pybullet_pics0'))), 4))

        # Iterate over each text prompt
        for j, text in enumerate(texts_dict[didi]):
            print(f"Processing {dir_name} with text: {text}")
            for k, image in tqdm(enumerate(images[:150])):
                img = Image.open(os.path.join(directory, 'pybullet_pics0', image)).convert('RGB')
                img = img.resize((224, 224))
                img = transforms.ToTensor()(img).to(device)

                preds = model.forward({"image": img, "text": text})
                preds = np.array([
                    preds["vx"].detach().cpu().numpy(),
                    preds["vy"].detach().cpu().numpy(),
                    preds["vz"].detach().cpu().numpy(),
                    preds["yaw"].detach().cpu().numpy()
                ]).reshape(-1)

                outputs[c_name[i]][dir_name][j, k] = preds

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    if not os.path.exists("results"):
        os.makedirs("results")

    # save the outputs
    np.save(f"results/{c_name[i]}.npy", outputs[c_name[i]])

print("Processing complete.")
