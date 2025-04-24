from typing import Any, Dict, Optional, List
import os
from lightning import LightningDataModule # type: ignore
from torch.utils.data import DataLoader, Dataset
from .components.flight_il_dataset_clean import FlightILDataset, worker_init_fn


class FlightDataModule(LightningDataModule):
    def __init__(
            self,
            train_data_path: List[str],
            eval_data_path: List[str],
            load_features_directly: Optional[bool] = False,
            shuffle: Optional[bool] = True,
            flag_last_num: Optional[int] = 5,
            seq_length: Optional[int] = 32,
            stride:Optional[int] = 1,
            batch_size: int = 64,
            num_workers: Optional[str] = 0,
            pin_memory: Optional[bool] = False,
            persistent_workers: Optional[bool] = False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_data_path = train_data_path[0]
        self.eval_data_path = eval_data_path[0]

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        assert os.path.isdir(self.train_data_path), f"{self.train_data_path} should be a directory"
        assert os.path.isdir(self.eval_data_path), f"{self.eval_data_path} should be a directory"


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.data_train, self.worker_init_fn_train = self._instantiate_dataset(
                self.hparams.train_data_path, self.hparams.mode, train=True)

            self.data_val, self.worker_init_fn_val = self._instantiate_dataset(
                self.hparams.eval_data_path, self.hparams.mode)

        if stage == "test" or stage is None:
            self.data_test, self.worker_init_fn_test = self._instantiate_dataset(
                self.hparams.eval_data_path, self.hparams.mode)

    def _instantiate_dataset(self, data_path, **kwargs):
        dataset = FlightILDataset(
            data_path=data_path,
            shuffle=self.hparams.shuffle,
            seq_length=self.hparams.seq_length,
            stride=self.hparams.stride,
            load_features_directly=self.hparams.load_features_directly,
            flag_last_num=self.hparams.flag_last_num
        )
        return dataset, worker_init_fn
    
    def _create_dataloader(self, dataset, worker_init_fn, mode="train"):
        is_mp = self.hparams.num_workers > 0
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers if mode == "train" else int(is_mp),
            # we don't quite use passive eval but we need to make it the same as train loader in terms of multi-processing
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.hparams.persistent_workers and is_mp,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.data_train, self.worker_init_fn_train, mode="train")

    def val_dataloader(self):
        return self._create_dataloader(self.data_val, self.worker_init_fn_val, mode="val")
            
    def test_dataloader(self):
        return self._create_dataloader(self.data_test, self.worker_init_fn_test, mode="test")

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass