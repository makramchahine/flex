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
import io
import gc

# Configuration paths
cfgs = [
    "/home/makramchahine/repos/flex/local/train_flight/simplevit/config",
    "/home/makramchahine/repos/flex/local/train_flight/conv/config",
    "/home/makramchahine/repos/flex/local/train_flight/linear/config",
    "/home/makramchahine/repos/flex/local/train_flight/full_img_first_dim_trans/config",
    "/home/makramchahine/repos/flex/local/train_flight/full_img_all_dim_trans/config"
]

# Corresponding texts for inference
texts = [
    "fly to the blue object",
    "fly to the blue target",
    "reach the blue goal",
    "atteins la cible bleue"
]

# Path to the images directory
images_path = "/home/makramchahine/repos/fm_flight/BLIP2_DATASET/eval/save-flight-02.13.2024_21.41.47.286898"

# List all images in the images directory
images = os.listdir(images_path)
images = sorted([img for img in images if img.endswith(".png")])

# Initialize the output array
outputs = np.zeros((len(cfgs), len(texts), len(images), 4))

# Iterate over each configuration
for i, cfg_path in enumerate(cfgs):
    # Load the configuration
    cfg = OmegaConf.load(cfg_path)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load the checkpoint if provided
    if cfg.ckpt_path:
        # Ensure the first checkpoint path is valid
        cpath = str(cfg.ckpt_path[0])
        print(f"Loading checkpoint: {cpath}")

        # Check if the checkpoint file exists
        if os.path.isfile(cpath):
            # Load the checkpoint into a buffer and then into the model
            with open(cpath, 'rb') as f:
                buffer = io.BytesIO(f.read())
                ckpt = torch.load(buffer, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {cpath}")

        # Remove unnecessary keys
        for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
            ckpt["state_dict"].pop(dropped_key, None)

        # Load state dict into model
        model.load_state_dict(ckpt["state_dict"])

    # Move model to the appropriate device
    model = model.to(device)
    print(f"\nCheckpoint {i} loaded successfully\n")

    # Iterate over each text prompt
    for j, text in enumerate(texts):
        print(f"Running model {i} with text: '{texts[j]}'")
        # Iterate over each image
        for k, image in tqdm(enumerate(images)):
            # Load and process the image
            img = Image.open(os.path.join(images_path, image)).convert('RGB')
            img = img.resize((224, 224))
            img = transforms.ToTensor()(img).to(device)

            # Run the model inference
            preds = model.forward({"image": img, "text": text})

            # Convert predictions to a numpy array and reshape
            preds = np.array([
                preds["vx"].detach().cpu().numpy(),
                preds["vy"].detach().cpu().numpy(),
                preds["vz"].detach().cpu().numpy(),
                preds["yaw"].detach().cpu().numpy()
            ]).reshape(-1)

            # Store the predictions in the outputs array
            outputs[i, j, k] = preds

    # After processing with the model:
    del model  # Delete the model
    torch.cuda.empty_cache()  # Release cached memory
    gc.collect()  # Force garbage collection
    torch.cuda.reset_max_memory_allocated()  # Reset CUDA memory allocator state
    print(f"CUDA memory cleaned up after model {i}")

# Save the output predictions
np.save("open_loop_outputs.npy", outputs)
