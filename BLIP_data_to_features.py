import os
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from learning.src.models.components.extractors.blip import BLIPExtractor
from torchvision import transforms

# ----------------- SEED FUNCTION -----------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# ----------------- DATA to BLIP FUNCTION -----------------
def data2BLIP(model, data_path, target_path, device):
    runs = sorted(os.listdir(data_path))
    for i, run in enumerate(runs):
        target_path_run = os.path.join(target_path, run)
        os.makedirs(target_path_run, exist_ok=True)

        run = os.path.join(data_path, run)
        with open(os.path.join(run,  'label.txt'), 'r') as f:
            text = f.read()

        image_names = sorted([f for f in os.listdir(run) if f.endswith('.png')])
        run_loop = tqdm(enumerate(image_names), total=len(image_names), desc=f"Run {i+1}/{len(runs)}")
        for j, im_name in run_loop:
            img = Image.open(os.path.join(run, im_name))
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img = transforms.ToTensor()(img)
            data_out = model({"image": img.to(device), "text":text})
            # print("sanity_check", data_out.shape)
            torch.save(data_out, os.path.join(target_path_run, im_name.replace('.png', '.pth')))
            run_loop.set_postfix({'Image': im_name})


# ----------------- MAIN EXECUTION -----------------
if __name__ == '__main__':
    data_path:str = '/home/alex/flex/BLIP2_DATASET/'
    target_dir:str = "BLIP_Features/"
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
            patch_size=2,
            all_q_dims= False,
            use_masked_patch_wise_feature= True,
            use_visual_encoder_only= False
        )
    
    model.to(device)
    model.eval()
    # data2BLIP(model, data_path+'train/', target_dir+'train/', device)
    data2BLIP(model, data_path+'eval/', target_dir+'eval/', device)

