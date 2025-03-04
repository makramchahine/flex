from collections import OrderedDict
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig
import random
import torch, torchvision
from torch.utils.data import IterableDataset
import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import cv2



class FlightILDataset(IterableDataset):
    def __init__(
            self,
            data_path: List[str],
            load_features_directly: Optional[bool] = False,
            train: Optional[bool] = False,
            shuffle: Optional[bool] = True,
            snippet_size: Optional[int] = 100,
            use_standardize: Optional[bool] = True,
            use_clip_preprocess: Optional[bool] = False,
            use_lavis_preprocess: Optional[bool] = False,
            lavis_preprocess_cfg: Optional[DictConfig] = None,
            multi_instruction: Optional[bool] = False,
            seq_length = 32,
            stride = 5,
            **kwargs,
    ):
        self.data_path = data_path[0]
        self._use_standardize = use_standardize
        self._use_clip_preprocess = use_clip_preprocess
        self._use_lavis_preprocess = use_lavis_preprocess
        self._lavis_preprocess_cfg = lavis_preprocess_cfg
        self._shuffle = False#shuffle
        self._multi_instructions = multi_instruction
        self._seq_length = seq_length
        self._stride = stride
        self._load_features_directly = load_features_directly

        if self._use_lavis_preprocess:
            if self._lavis_preprocess_cfg.name == "BlipImageEvalProcessor":
                from lavis.processors import BlipImageEvalProcessor
                pp_cls = BlipImageEvalProcessor(
                    image_size=224,
                    # TODO: check if the mean and std are correct, are these dataset specific?
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )
                n_px = 224
                assert pp_cls.transform.transforms[0].size[0] == n_px
                pp_cls.transform.transforms[0].size = n_px # only resize the smaller side
                pp_cls.transform.transforms[0].antialias = True
                pp_cls.transform = torchvision.transforms.Compose([
                    pp_cls.transform.transforms[0], # resize smaller side to 224
                    torchvision.transforms.CenterCrop(n_px), # center crop
                    pp_cls.transform.transforms[2], # normalize
                ])
            else:
                raise ValueError(f"Unrecognized lavis preprocessor {self._lavis_preprocess_cfg.name}")
            self._lavis_preprocessor = pp_cls
        else:
            self._lavis_preprocessor = None

        self._transform_rgb = partial(
            transform_rgb,
            use_standardize=self._use_standardize,
            use_clip_preprocess=self._use_clip_preprocess,
            lavis_preprocessor=self._lavis_preprocessor,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        self._rng = random.Random(worker_id)

        while True:
            if not self._shuffle:
                # for each run folder in the data path
                runs = os.listdir(self.data_path)
                runs.sort()
                for run in runs:
                    run = os.path.join(self.data_path, run)
                    if self._load_features_directly:
                        run_feat = run.replace('DATASET', 'Features')
                        image_names = [f for f in os.listdir(run_feat) if f.endswith('.pth')]
                    else:
                        image_names = [f for f in os.listdir(run) if f.endswith('.png')]
                    image_names.sort()

                    # print(f"Run: {run} has {len(image_names)} images", self.data_path)

                    # load the text input instruction as a string
                    with open(os.path.join(run,  'label.txt'), 'r') as f:
                        texts = f.read()
                    texts = texts.split('\n')
                    
                    batch_imgs = []
                    #looping over texts in the case of multiple/sequence of text instructions
                    for j, text in enumerate(texts):
                        # load the labels from the data_out.csv file with pandas, ignore the header
                        labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'))
                        reached_last = False
                        stride_start = 1 # skip first image to account for data mismatch
                        while not reached_last:
                            for i in range(self._seq_length):
                                data_id = i + stride_start
                                if data_id > 0.5 * len(labels):
                                    data_id = len(labels)
                                    reached_last = True

                                # load the image and preprocess to make it 224x224 tensor
                                # print('image_path', os.path.join(run, image_names[data_id]))
                                if self._load_features_directly:
                                    img = torch.load(os.path.join(run_feat, image_names[data_id]), map_location=torch.device('cpu'), weights_only=False)
                                    img = img[0]
                                else:
                                    img = Image.open(os.path.join(run, image_names[data_id]))
                                    img = img.convert('RGB')
                                    img = img.resize((224, 224))
                                    img = transforms.ToTensor()(img)

                                if self._multi_instructions:
                                    batch_imgs.append(img)
                                else:
                                    # label is the 4 first elements of the i-th row of the labels dataframe
                                    label = labels.iloc[data_id - 1, :4].values
                                    label = np.array(label).astype(np.float32)
                                    # print('label path', label)
                                    # yield the image, text and the label
                                    yield {"image": img, "text":text, "is_last":reached_last}, OrderedDict({"vx": label[0], "vy": label[1], "vz": label[2], "yaw": label[3]})
                            if self._multi_instructions:
                                label = labels.iloc[stride_start: stride_start + self._seq_length, :4].values
                                if label.shape[0] < self._seq_length:
                                    last_row = label[-1] if label.shape[0] > 0 else np.zeros(4)  # Handle empty case
                                    padding = np.tile(last_row, (self._seq_length - label.shape[0], 1))
                                    label = np.vstack([label, padding])
                                label = np.array(label).astype(np.float32)
                                yield {"image": torch.stack(batch_imgs), "text":text}, OrderedDict({"vx": label[:, 0], "vy": label[:, 1], "vz": label[:, 2], "yaw": label[:, 3]})
                            stride_start += self._stride
            else:
                # pick a random run folder in the data path
                run = self._rng.choice(os.listdir(self.data_path))
                run = os.path.join(self.data_path, run)
                if self._load_features_directly:
                    run_feat = run.replace('DATASET', 'Features')
                    image_names = [f for f in os.listdir(run_feat) if f.endswith('.pth')]
                    image_names.sort()
                    i = self._rng.choice(range(1, len(image_names)))
                    img = torch.load(os.path.join(run_feat, image_names[i]), map_location=torch.device('cpu'))
                    img = img[0]
                    text = ''
                else:
                    # sort the list of image names
                    image_names = [f for f in os.listdir(run) if f.endswith('.png')]
                    image_names.sort()
                    # pick a random non zero index in len(image_names)
                    i = self._rng.choice(range(1, len(image_names)))

                    # load the image
                    img = Image.open(os.path.join(run, image_names[i]))
                    # convert to RGB
                    img = img.convert('RGB')
                    # resize the image to 224x224
                    img = img.resize((224, 224))
                    # convert the image to a tensor
                    img = transforms.ToTensor()(img)

                    # load the text input instruction as a string
                    with open(os.path.join(run,  'label.txt'), 'r') as f:
                        text = f.read()

                # load the labels from the data_out.csv file with pandas, ignore the header
                labels = pd.read_csv(os.path.join(run, 'data_out.csv'))
                # label is the 4 first elements of the i-th row of the labels dataframe
                label = labels.iloc[i-1, :4].values
                label = np.array(label).astype(np.float32)

                # yield the image, text and the label
                yield {"image": img, "text":text}, OrderedDict({"vx": label[0], "vy": label[1], "vz": label[2], "yaw": label[3]})




def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()

def standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    mean, stddev = x.mean(), x.std()
    adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.prod(x.shape)))
    return (x - mean) / adjusted_stddev

def transform_rgb(img: np.ndarray,
                train: bool,
                label: float = None,
                use_standardize: bool = True,
                use_clip_preprocess: bool = False,
                lavis_preprocessor: str = None):
    # need copy here probably since img is not contiguous
    img = img.copy()
    img = TF.to_tensor(img)
    if train:  # perform color jitter
        gamma_range = [0.5, 1.5]
        brightness_range = [0.5, 1.5]
        contrast_range = [0.3, 1.7]
        saturation_range = [0.5, 1.5]

        img = TF.adjust_gamma(img, np.random.uniform(*gamma_range))
        img = TF.adjust_brightness(img, np.random.uniform(*brightness_range))
        img = TF.adjust_contrast(img, np.random.uniform(*contrast_range))
        img = TF.adjust_saturation(img, np.random.uniform(*saturation_range))
    if use_standardize:
        img = standardize(img)
    if use_clip_preprocess:
        # follow clip._transform
        n_px = 224
        img = TF.resize(img, n_px, interpolation=InterpolationMode.BICUBIC, antialias=True) # NOTE: resize needs to come after color jitter otherwise will cause nan
        img = TF.center_crop(img, n_px)

        img = TF.normalize(img, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if lavis_preprocessor:
        img = lavis_preprocessor(img)
    return img if label is None else (img, label)

# if main plot data for sequence length 10 and stride of 9
if __name__ == '__main__':
    # plot the 100 first yielded images on a single plot
    import matplotlib.pyplot as plt

    dataset = FlightILDataset(data_path=['/home/alex/flex/BLIP2_DATASET/train'], seq_length=10, stride=9, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, worker_init_fn=worker_init_fn)

    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    for i, data in enumerate(dataloader):
        img = data[0]["image"]
        text = data[0]["text"]
        label = data[1]
        axs[i // 10, i % 10].imshow(img[0].permute(1, 2, 0))
        axs[i // 10, i % 10].set_title(text)
        if i == 99:
            break
    plt.show()