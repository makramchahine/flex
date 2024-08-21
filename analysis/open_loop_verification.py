from typing import List, Optional, Tuple

import hydra
from lightning import LightningModule
from omegaconf import OmegaConf
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# make a dict conf called CFG
cfgs = ["/home/makramchahine/repos/flex/local/train_flight/simplevit/config",
        "/home/makramchahine/repos/flex/local/train_flight/conv/config",
        "/home/makramchahine/repos/flex/local/train_flight/linear/config",
        "/home/makramchahine/repos/flex/local/train_flight/full_img_first_dim_trans/config",
        "/home/makramchahine/repos/flex/local/train_flight/full_img_all_dim_trans/config"]

texts = ["fly to the blue object",
         "fly to the blue target",
         "reach the blue goal",
         "atteins la cible bleue"]

images_path = "/home/makramchahine/repos/fm_flight/BLIP2_DATASET/eval/save-flight-02.13.2024_21.41.47.286898"

# list images in the images_path
images = os.listdir(images_path)

outputs = np.zeros((len(cfgs), len(texts), len(images),4))


for i, cfg in enumerate(cfgs):
    cfg = OmegaConf.load(cfg)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if cfg.ckpt_path:
        # if its a list load the last one
        cpath = cfg.ckpt_path[0]
        print(f"Loading checkpoint: {cpath}")
        ckpt = torch.load(cpath, map_location=device)
        for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
            if dropped_key in ckpt["state_dict"].keys():
                ckpt["state_dict"].pop(dropped_key) # HACK: remove param used for determining device
        model.load_state_dict(ckpt["state_dict"])

    # make a copy of the model
    modello = model.to(device)

    print(f"\n checkpoint {i} loaded successfully \n")

    for j, text in enumerate(texts):
        print(f"Running model {i} with text {texts[j]}")
        for k, image in tqdm(enumerate(images)):
            # load the image
            img = Image.open(os.path.join(images_path, image))
            # make the image have 3 channels
            img = img.convert('RGB')

            # resize the image to 224x224
            img = img.resize((224, 224))

            # convert the image to a tensor
            img = transforms.ToTensor()(img).to(device)

            # run inference
            preds = model.forward({"image": img, "text": text})
            # convert the preds dict to a numpy array of shape (4,)
            print(preds)
            exit()
            preds = np.array([preds["x"], preds["y"], preds["z"], preds["yaw"]])
            outputs[i, j, k] = preds

# save the outputs
np.save("open_loop_outputs.npy", outputs)