import warnings
from copy import deepcopy
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from rul_datasets.cmapss import CMAPSSDataModule, PairedCMAPSS
from rul_datasets.loader import CMAPSSLoader


class BaselineDataModule(pl.LightningDataModule):
    def __init__(self, data_module: CMAPSSDataModule):
        super().__init__()

        self.data_module = data_module
        hparams = self.data_module.hparams
        self.save_hyperparameters(hparams)

        self.cmapss = {}
        for fd in range(1, 5):
            self.cmapss[fd] = self._get_cmapss(fd)

    def _get_cmapss(self, fd):
        if fd == self.hparams["fd"]:
            cmapss = self.data_module
        else:
            loader = deepcopy(self.data_module.loader)
            loader.fd = fd
            loader.percent_fail_runs = None
            loader.percent_broken = None
            cmapss = CMAPSSDataModule(loader, self.data_module.batch_size)

        return cmapss

    def prepare_data(self, *args, **kwargs):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.data_module.train_dataloader()

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.data_module.val_dataloader()

    def test_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        test_dataloaders = []
        for fd_target in range(1, 5):
            target_dl = self.cmapss[fd_target].test_dataloader()
            test_dataloaders.append(target_dl)

        return test_dataloaders


class PretrainingBaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        failed_data_module: CMAPSSDataModule,
        unfailed_data_module: CMAPSSDataModule,
        num_samples: int,
        min_distance: int = 1,
        distance_mode: str = "linear",
    ):
        super().__init__()

        self.failed_loader = failed_data_module.loader
        self.unfailed_loader = unfailed_data_module.loader
        self.num_samples = num_samples
        self.batch_size = failed_data_module.batch_size
        self.min_distance = min_distance
        self.distance_mode = distance_mode
        self.window_size = self.unfailed_loader.window_size
        self.source = unfailed_data_module

        self._check_loaders()

        self.save_hyperparameters(
            {
                "fd_source": self.unfailed_loader.fd,
                "num_samples": self.num_samples,
                "batch_size": self.batch_size,
                "window_size": self.window_size,
                "max_rul": self.unfailed_loader.max_rul,
                "min_distance": self.min_distance,
                "percent_broken": self.unfailed_loader.percent_broken,
                "percent_fail_runs": self.failed_loader.percent_fail_runs,
                "truncate_val": self.unfailed_loader.truncate_val,
                "distance_mode": self.distance_mode,
            }
        )

    def _check_loaders(self):
        self.failed_loader.check_compatibility(self.unfailed_loader)
        if not self.failed_loader.fd == self.unfailed_loader.fd:
            raise ValueError("Failed and unfailed data need to come from the same FD.")
        if self.failed_loader.percent_fail_runs is None or isinstance(
            self.failed_loader.percent_fail_runs, float
        ):
            raise ValueError(
                "Failed data needs list of failed runs "
                "for pre-training but uses a float or is None."
            )
        if self.unfailed_loader.percent_fail_runs is None or isinstance(
            self.unfailed_loader.percent_fail_runs, float
        ):
            raise ValueError(
                "Unfailed data needs list of failed runs "
                "for pre-training but uses a float or is None."
            )
        if set(self.failed_loader.percent_fail_runs).intersection(
            self.unfailed_loader.percent_fail_runs
        ):
            raise ValueError(
                "Runs of failed and unfailed data overlap. "
                "Please use mututally exclusive sets of runs."
            )
        if (
            self.unfailed_loader.percent_broken is None
            or self.unfailed_loader.percent_broken == 1.0
        ):
            raise ValueError(
                "Unfailed data needs a percent_broken smaller than 1 for pre-training."
            )
        if (
            self.failed_loader.percent_broken is not None
            and self.failed_loader.percent_broken < 1.0
        ):
            raise ValueError(
                "Failed data cannot have a percent_broken smaller than 1, "
                "otherwise it would not be failed data."
            )
        if not self.unfailed_loader.truncate_val:
            warnings.warn(
                "Validation data of unfailed runs is not truncated. "
                "The validation metrics will not be valid."
            )

    def _get_unbroken_runs(self, fail_runs):
        if fail_runs is None or isinstance(fail_runs, float):
            unfailed_runs = None
        else:
            run_idx = range(CMAPSSLoader.NUM_TRAIN_RUNS[self.fd_source])
            unfailed_runs = list(set(run_idx).difference(fail_runs))

        return unfailed_runs

    def prepare_data(self, *args, **kwargs):
        self.unfailed_loader.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._get_paired_dataset("dev"), batch_size=self.batch_size, pin_memory=True
        )

    def val_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        combined_loader = DataLoader(
            self._get_paired_dataset("val"), batch_size=self.batch_size, pin_memory=True
        )
        source_loader = self.source.val_dataloader()

        return [combined_loader, source_loader]

    def _get_paired_dataset(self, split: str) -> PairedCMAPSS:
        deterministic = split == "val"
        min_distance = 1 if split == "val" else self.min_distance
        num_samples = 25000 if split == "val" else self.num_samples
        paired = PairedCMAPSS(
            [self.unfailed_loader, self.failed_loader],
            split,
            num_samples,
            min_distance,
            deterministic,
            mode=self.distance_mode,
        )

        return paired
