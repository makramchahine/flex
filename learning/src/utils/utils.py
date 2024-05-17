import warnings
from importlib.util import find_spec
from typing import Callable
import pickle

from omegaconf import DictConfig
import h5py
import numpy as np
import pandas as pd
from pandas import HDFStore
from torch.utils.data import Dataset

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


class FeatureSaver:
    def __init__(self, filepath, max_len=100, mode='hdfstore', write_freq=1):
        self.filepath = filepath
        self.max_len = max_len
        self.mode = mode
        self.write_freq = write_freq

        self.cache = dict()
        if self.mode == 'h5file':
            self.dset = dict()
        elif self.mode == 'hdfstore':
            self.hdf = HDFStore(self.filepath, mode='w')
            self.data_shape = dict()
        else:
            raise ValueError(f'Unrecognized mode {self.mode}')
        
        self.write_counter = 0

    def write(self, data):
        if self.write_counter % self.write_freq == 0:
            if self.mode == 'h5file' and len(self.dset.keys()) == 0:
                with h5py.File(self.filepath, 'w') as hf:
                    for k, v in data.items():
                        v = np.array(v)
                        self.dset[k] = hf.create_dataset(k, data=v[None], compression="gzip", chunks=True, 
                                                        maxshape=(None,)+tuple(v.shape))
            else:
                if len(self.cache.keys()) == 0:
                    for k in data.keys():
                        self.cache[k] = []

                for k, v in data.items():
                    self.cache[k].append(v)

                k0 = list(data.keys())[0]
                if len(self.cache[k0]) >= self.max_len:
                    self.flush()
                
        self.write_counter += 1

    def flush(self):
        if self.mode == 'cache':
            with h5py.File(self.filepath, 'a') as hf:
                for k, v in self.cache.items():
                    v = np.array(v)
                    hf[k].resize(hf[k].shape[0] + v.shape[0], axis=0)
                    hf[k][-v.shape[0]:] = v
        elif self.mode == 'hdfstore':
            for k, v in self.cache.items():
                v = np.array(v)

                if k not in self.data_shape.keys():
                    self.data_shape[k] = list(v.shape)
                else:
                    assert np.all([self.data_shape[k][_i] == v.shape[_i] for _i in range(1, len(self.data_shape[k]))])
                    self.data_shape[k][0] += v.shape[0]

                df_v = pd.concat([pd.DataFrame(vv) for vv in v])
                if self.hdf.is_open:
                    self.hdf.append(f'{k}', df_v)
                else:
                    self.hdf.open()
                    self.hdf.put(f'{k}', df_v, format='table')
        self.cache = dict()

    def close(self):
        if self.mode == 'hdfstore' and self.hdf.is_open:
            self.flush()
            for k, v in self.data_shape.items():
                self.hdf.put(f'{k}_shape', pd.Series(v))
            self.hdf.close()


class H5Dataset(Dataset):
    def __init__(self, filepath, key, **kwargs):
        self.data_shape = pd.read_hdf(filepath, key=f'{key}_shape')
        self.data = dict()
        self.key = key
        self.filepath = filepath

    def __getitem__(self, index):
        if index not in self.data.keys():
            start = index * self.data_shape[1]
            stop = (index + 1) * self.data_shape[1]
            data_i = pd.read_hdf(self.filepath, key=self.key, start=start, stop=stop)
            data_i = data_i.to_numpy()
            self.data[index] = data_i
        else:
            data_i = self.data[index]

        return data_i

    def __len__(self):
        return self.data_shape[0]


class H5DatasetReadOnce(Dataset):
    def __init__(self, filepath, key, indexing_dim=0, verbose=True):
        self.data_shape = pd.read_hdf(filepath, key=f'{key}_shape')
        self.key = key
        self.filepath = filepath
        self.indexing_dim = indexing_dim

        if verbose:
            print('Reading data at once, probably take some time...')
        self.data = pd.read_hdf(self.filepath, key=self.key)
        self.data = self.data.to_numpy().reshape([-1,] + self.data_shape[1:].to_list())
        if verbose:
            print('Data loading done!')

    def __getitem__(self, index):
        if self.indexing_dim == 0:
            data_i = self.data[index]
        elif self.indexing_dim == 1:
            data_i = self.data[:, index]
        elif self.indexing_dim == 2:
            data_i = self.data[:, :, index]
        else:
            raise NotImplementedError

        return data_i

    def __len__(self):
        if self.indexing_dim == 0:
            return self.data_shape[0]
        elif self.indexing_dim == 1:
            return self.data_shape[1]
        elif self.indexing_dim == 2:
            return self.data_shape[2]
        else:
            raise NotImplementedError


def read_h5(filepath, key='feats', mode='dataset', verbose=False, **kwargs):
    """ Read h5 at once. Inefficient if the file size is large. """
    if mode == 'dataset':
        dset = H5Dataset(filepath, key, **kwargs)
    elif mode == 'dataset_read_once':
        dset = H5DatasetReadOnce(filepath, key, verbose=verbose, **kwargs)
    else:
        with h5py.File(filepath, 'r') as hf:
            a_group_key = list(hf.keys())[0]
            if verbose:
                print("Keys: %s" % hf.keys())
                print(type(hf[a_group_key]))

            data = list(hf[a_group_key])

            data = list(hf[a_group_key])

            ds_obj = hf[a_group_key]
            ds_arr = hf[a_group_key][()]

        dset = ds_arr

    return dset


class InfoSaver:
    def __init__(self, out_path, write_freq):
        self.out_path = out_path
        self.write_freq = write_freq
        self.cache = []
        self.counter = 0
        
    def set_new_ep(self):
        self.cache.append([])
        
    def write(self, info):
        self.cache[-1].append(info)
        self.counter += 1
        if self.counter % self.write_freq == 0:
            self.save()
        
    def save(self):
        with open(self.out_path, "wb") as f:
            pickle.dump(self.cache, f)
            
    def close(self):
        self.save()
