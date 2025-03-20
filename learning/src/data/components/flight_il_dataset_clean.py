from collections import OrderedDict
from typing import List, Dict, Any, Optional
import random
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from PIL import Image
import os
import numpy as np
from torchvision import transforms # type: ignore
import json

class FlightILDataset(IterableDataset):
    """
    FlightILDataset is an IterableDataset that yields a sequence of images and labels from the BLIP2 dataset.
    args:
        data_path: the path to the dataset
        shuffle: whether to shuffle the dataset
        seq_length: the length of the sequence of images
        stride: the stride between sequences
    """
    def __init__(
            self,
            data_path: List[str],
            shuffle: Optional[bool] = True,
            seq_length: Optional[int] = 32,
            stride:int = 1,
            load_features_directly: Optional[bool] = False,
            flag_last_num: Optional[int] = 5,
            **kwargs,
    ):
        self.data_path = data_path[0]
        self._shuffle = shuffle
        self._seq_length = seq_length
        self._stride = stride
        self._load_features_directly = load_features_directly
        self._num_last = flag_last_num

        #Logging
        self.iter_num = 0
        self.runs_count = {}

    def _load_data(self, run: str):
        """
        Load the data from the run folder.
        args:
            run: the path to the run folder
        """
        if self._load_features_directly:
            self.run_feat = run.replace('DATASET', 'Features')
            self.cur_image_names = sorted([f for f in os.listdir(self.run_feat) if f.endswith('.pt')])
            self.cur_texts = ['']
        else:
            self.cur_image_names = sorted([f for f in os.listdir(run) if f.endswith('.png')])
            with open(os.path.join(run,  'label.txt'), 'r') as f:
                texts = f.read()
            self.cur_texts = texts.split('\n')
    
    def _get_image_label(self, run:str, index:int, im_shift=0):
        """
        Get the image and label at the given index.
        args:
            run: the path to the run folder
            index: the index of the image in the sequence
            im_shift: the shift in the index of the image
        """
        if self._load_features_directly:
            img = torch.load(os.path.join(self.run_feat, self.cur_image_names[index + 1 + im_shift]),
                             map_location=torch.device('cpu'), weights_only=False)
            img = img[0]
        else:
            img = Image.open(os.path.join(run, self.cur_image_names[index + 1 + im_shift]))
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img = transforms.ToTensor()(img)
        # label is the 4 first elements of the i-th row of the labels dataframe
        label = self.cur_labels.iloc[index, :4].values
        label = np.array(label).astype(np.float32)
        is_last = 1.0 if self.num_label - index <= self._num_last else 0.0
        label = OrderedDict({"vx": label[0], "vy": label[1], "vz": label[2], "yaw": label[3], "stop": is_last})
        return img, label
    
    def __iter__(self):
        """
        Iterate over the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        self._rng = random.Random(worker_id)
        # for each run folder in the data path
        # runs = sorted(os.listdir(self.data_path))
        runs = sorted([f for f in os.listdir(self.data_path) if f.endswith('right')])
        while True:
            if self._shuffle:
                # pick a random run folder in the data path
                run = self._rng.choice(runs)
                self.runs_count[run] = self.runs_count.get(run, 0) + 1
                if (self.iter_num % 500 == 0):
                    with open('/home/alex/flex/local/runs_count.json', 'w') as f:
                        json.dump(self.runs_count, f)
                        self.iter_num = 0
                self.iter_num += 1

                is_old_run = run.startswith('save')
                run = os.path.join(self.data_path, run)
                self._load_data(run)

                j = self._rng.choice(range(len(self.cur_texts)))
                text = self.cur_texts[j]
                self.cur_labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'))
                self.num_label = len(self.cur_labels)
                if is_old_run:
                    self.num_label = int(self.num_label * 0.85)

                im_shift = 0
                if j > 0:
                    # use a map in future to record length so as to avoid time cost here
                    for k in range(j-1):
                        im_shift += len(pd.read_csv(os.path.join(run, f'data_out{k if k>0 else ""}.csv'))) + 1
                
                if self._seq_length is not None:
                    thresh = self.num_label - self._seq_length
                    i = self._rng.choice(range(thresh + 5)) #Adding 5 for more encounter to stop labels
                    if i > thresh: i = thresh
                    for k in range(i, i + self._seq_length):
                        img, label = self._get_image_label(run, k, im_shift)
                        yield {"image": img, "text":text}, label
                else:
                    i = self._rng.choice(range(self.num_label))
                    img, label = self._get_image_label(run, i, im_shift)
                    yield {"image": img, "text":text}, label

            else:
                for run in runs:
                    is_old_run = run.startswith('save')
                    run = os.path.join(self.data_path, run)
                    self._load_data(run)

                    for j, text in enumerate(self.cur_texts):
                        # load the labels from the data_out.csv file with pandas, ignore the header
                        self.cur_labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'))
                        self.num_label = len(self.cur_labels)
                        if is_old_run:
                            self.num_label = int(self.num_label * 0.85)
                        reached_last = False
                        stride_start = 1 # skip first image to account for data mismatch
                        while not reached_last:
                            for i in range(self._seq_length):
                                data_id = i + stride_start
                                if data_id > self.num_label:
                                    data_id = self.num_label
                                    reached_last = True

                                img, label = self._get_image_label(run, data_id - 1)
                                yield {"image": img, "text":text}, label
                            stride_start += self._stride
                

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
