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
from src.data.components.utils import transform_rgb
import cv2



class FlightILDataset(IterableDataset):
    def __init__(
            self,
            data_path: List[str],
            train: Optional[bool] = False,
            shuffle: Optional[bool] = True,
            snippet_size: Optional[int] = 100,
            use_standardize: Optional[bool] = True,
            use_clip_preprocess: Optional[bool] = False,
            use_lavis_preprocess: Optional[bool] = False,
            lavis_preprocess_cfg: Optional[DictConfig] = None,
            **kwargs,
    ):
        self.data_path = data_path[0]
        self._use_standardize = use_standardize
        self._use_clip_preprocess = use_clip_preprocess
        self._use_lavis_preprocess = use_lavis_preprocess
        self._lavis_preprocess_cfg = lavis_preprocess_cfg
        self._shuffle = shuffle

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
                for run in os.listdir(self.data_path):
                    run = os.path.join(self.data_path, run)
                    # list all png images in the run folder
                    image_names = [f for f in os.listdir(run) if f.endswith('.png')]
                    # sort the list of image names
                    image_names.sort()

                    # print(f"Run: {run} has {len(image_names)} images")


                    # load the text input instruction as a string
                    with open(os.path.join(run,  'label.txt'), 'r') as f:
                        text = f.read()

                    # load the labels from the data_out.csv file with pandas, ignore the header
                    labels = pd.read_csv(os.path.join(run, 'data_out.csv'))

                    # print(f"Run: {run} has {len(labels)} labels")

                    # assert that there are as many rows of numerical labels as there are images
                    assert len(image_names) == len(labels)+1

                    # for each image in the run
                    for i, image_name in enumerate(image_names):
                        # skip first image to account for data mismatch
                        if i == 0:
                            continue

                        # load the image
                        img = Image.open(os.path.join(run, image_name))
                        # convert to RGB
                        img = img.convert('RGB')
                        # resize the image to 224x224
                        img = img.resize((224, 224))
                        # convert the image to a tensor
                        img = transforms.ToTensor()(img)

                        # label is the 4 first elements of the i-th row of the labels dataframe
                        label = labels.iloc[i-1, :4].values
                        label = np.array(label).astype(np.float32)

                        # yield the image, text and the label
                        yield {"image": img, "text":text}, OrderedDict({"vx": label[0], "vy": label[1], "vz": label[2], "yaw": label[3]})

            else:
                # pick a random run folder in the data path
                run = self._rng.choice(os.listdir(self.data_path))
                run = os.path.join(self.data_path, run)
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