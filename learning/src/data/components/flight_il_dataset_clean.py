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
import datetime 


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
            features_folder: Optional[str] = None,
            flag_last_num: Optional[int] = 5,
            **kwargs,
    ):
        self.data_path = data_path[0]
        self.run_types = [
            'left_blue', 'left_red', 
            'right_blue', 'right_red', 
            'above_blue', 'above_red', 
            'below_blue', 'below_red', 
            'save_flight'
        ]
        self.run_types_wts = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]

        self._shuffle = shuffle
        self._seq_length = seq_length
        self._stride = stride
        self._features_folder = features_folder
        self._num_last = flag_last_num

        #Logging
        self.iter_num = 0
        self.runs_count = {}
        self.runs_name = f'runs_count_{datetime.datetime.now().strftime("%m_%d_%H_%M")}.json'
        self.cur_run = None

    def _load_data(self, run: str):
        """
        Load the data from the run folder.
        args:
            run: the path to the run folder
        """
        if self._features_folder is not None:
            self.run_feat = run.replace('BLIP2_DATASET', self._features_folder)
            self.cur_image_names = sorted([f for f in os.listdir(self.run_feat) if f.endswith('.pt')])
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
        if self._features_folder is not None:
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
        label = OrderedDict({
            "vx": label[0], "vy": label[1], "vz": label[2], 
            "yaw": label[3], "stop": is_last, 'step_remain': np.clip(1.0 - index/(self.num_label - self._num_last), 0.0, 1.0).astype(np.float32)
        })
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
        # runs = sorted([f for f in os.listdir(self.data_path) if f.startswith('save')])
        while True:
            if self._shuffle:
                # pick a random run folder in the data path
                run_type = self._rng.choices(self.run_types, weights=self.run_types_wts)[0]
                runs = os.listdir(os.path.join(self.data_path, run_type))
                run = self._rng.choice(runs)
#--------------------------------------------------Logging--------------------------------------------------
                self.cur_run = run
                # self.runs_count[run] = self.runs_count.get(run, 0) + 1
                # if (self.iter_num % 500 == 0):
                #     with open(f'/home/alex/flex/local/{self.runs_name}', 'w') as f:
                #         json.dump(self.runs_count, f)
                #         self.iter_num = 0
                # self.iter_num += 1
#--------------------------------------------------Logging--------------------------------------------------

                is_old_run = run.startswith('save')
                run = os.path.join(self.data_path, run_type, run)
                self._load_data(run)

                j = self._rng.choice(range(len(self.cur_texts)))
                text = self.cur_texts[j]
                if is_old_run:
                    # self.num_label = int(self.num_label * 0.85)
                    self.cur_labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'))
                    self._num_last = 14
                else:
                    self.cur_labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'), skiprows=1)
                    self._num_last = 5
                self.num_label = len(self.cur_labels)

                im_shift = 0
                if j > 0:
                    # use a map in future to record length so as to avoid time cost here
                    for k in range(j-1):
                        im_shift += len(pd.read_csv(os.path.join(run, f'data_out{k if k>0 else ""}.csv'))) + 1
                
                if self._seq_length is not None:
                    # thresh = self.num_label - self._seq_length
                    i = self._rng.choice(range(self.num_label)) #TODO check
                    i = max(0, min(self.num_label - self._seq_length, i))
                    # if i > thresh: i = thresh
                    for k in range(i, i + self._seq_length):
                        kl = k if k < self.num_label else self.num_label - 1
                        img, label = self._get_image_label(run, kl, im_shift)
                        yield {"image": img, "text":text}, label
                else:
                    i = self._rng.choice(range(self.num_label))
                    img, label = self._get_image_label(run, i, im_shift)
                    yield {"image": img, "text":text, "meta":run}, label

            else:
                for run_type in self.run_types:
                    runs = sorted(os.listdir(os.path.join(self.data_path, run_type)))
                    for run in runs:
                        self.cur_run = run
                        is_old_run = run.startswith('save')
                        run = os.path.join(self.data_path, run_type, run)
                        self._load_data(run)

                        for j, text in enumerate(self.cur_texts):
                            # load the labels from the data_out.csv file with pandas, ignore the header
                            if is_old_run:
                                # self.num_label = int(self.num_label * 0.85)
                                self.cur_labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'))
                                self._num_last = 14
                            else:
                                self.cur_labels = pd.read_csv(os.path.join(run, f'data_out{j if j>0 else ""}.csv'), skiprows=1)
                                self._num_last = 5
                            self.num_label = len(self.cur_labels)
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

# if main plot data for sequence length 10 and stride of 9
if __name__ == '__main__':
    # plot the 100 first yielded images on a single plot
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

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
