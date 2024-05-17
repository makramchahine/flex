from typing import Any, Dict, Optional, List
from omegaconf import DictConfig
import os
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class FlightDataModule(LightningDataModule):
    def __init__(
            self,
            train_data_path: List[str],
            eval_data_path: List[str],
            mode: Optional[str] = "IL",
            buffer_size: Optional[int] = 1,
            snippet_size: Optional[int] = 100,
            batch_size: int = 64,
            num_workers: Optional[str] = 0,
            pin_memory: Optional[bool] = False,
            persistent_workers: Optional[bool] = False,
            use_roi: Optional[bool] = True,
            use_standardize: Optional[bool] = True,
            use_clip_preprocess: Optional[bool] = False,
            use_lavis_preprocess: Optional[bool] = False,
            lavis_preprocess_cfg: Optional[DictConfig] = None,
            use_rejection_sampling: Optional[bool] = True,
            task_config: Optional[Dict[str, Any]] = None,
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
                self.hparams.train_data_path, self.hparams.mode, train=True, shuffle=True)

            self.data_val, self.worker_init_fn_val = self._instantiate_dataset(
                self.hparams.eval_data_path, self.hparams.mode)

        if stage == "test" or stage is None:
            self.data_test, self.worker_init_fn_test = self._instantiate_dataset(
                self.hparams.eval_data_path, self.hparams.mode)

    def _instantiate_dataset(self, data_path, mode="IL", train=False, shuffle=True):
        from src.data.components.flight_il_dataset import FlightILDataset, worker_init_fn
        dataset_cls = FlightILDataset
        dataset = dataset_cls(
            data_path=data_path,
            train=train,
            snippet_size=self.hparams.snippet_size,
            shuffle=shuffle,
            use_roi=self.hparams.use_roi,
            use_standardize=self.hparams.use_standardize,
            use_clip_preprocess=self.hparams.use_clip_preprocess,
            use_lavis_preprocess=self.hparams.use_lavis_preprocess,
            lavis_preprocess_cfg=self.hparams.lavis_preprocess_cfg,
        )

        return dataset, worker_init_fn

    def train_dataloader(self):
        is_mp = self.hparams.num_workers > 0
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self.worker_init_fn_train,
            # shuffle=True, # no shuffle setting in iterable dataset
            persistent_workers=self.hparams.persistent_workers and is_mp,
        )

    def val_dataloader(self):
        is_mp = self.hparams.num_workers > 0
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=int(is_mp),
            # we don't quite use passive eval but we need to make it the same as train loader in terms of multi-processing
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self.worker_init_fn_val,
            persistent_workers=self.hparams.persistent_workers and is_mp,
        )

    def test_dataloader(self):
        is_mp = self.hparams.num_workers > 0
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=int(is_mp),
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self.worker_init_fn_test,
            persistent_workers=self.hparams.persistent_workers and is_mp,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

# if __name__ == "__main__":
#     train_data_p=["/home/makramchahine/repos/fm_flight/BLIP2_DATASET/train/"]
#     eval_data_p=["/home/makramchahine/repos/fm_flight/BLIP2_DATASET/eval/"]
#
#     flight_data_module = FlightDataModule(
#         train_data_path=train_data_p,
#         eval_data_path=eval_data_p
#     )
#     flight_data_module.prepare_data()
#     flight_data_module.setup()
#     train_data_loader = flight_data_module.train_dataloader()
#     eval_data_loader = flight_data_module.val_dataloader()
#
#     i = 0
#     for sample in train_data_loader:
#         i+=1
#     print(i)


# data_loader = DataLoader(dataset, batch_size=1)
#
# for sample in data_loader:
#     text = sample[0]['text']
#     # get the image from the sample
#     img = sample[0]['image']
#     # convert the torch.Size([1, 3, 224, 224]) tensor to an image numpy array
#     img = img.squeeze().permute(1, 2, 0).numpy()
#     # colors are inverted so we need to invert them back
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # use cv to show the image until space is pressed
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     # break if key is escape
#     if cv2.waitKey(0) == 27:
#         break
#
#     print(text)