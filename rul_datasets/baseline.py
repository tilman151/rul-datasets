from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.cmapss import CMAPSSDataModule, PairedCMAPSS
from datasets.loader import CMAPSSLoader


class BaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source: int,
        batch_size: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_fail_runs: float = None,
        feature_select: List[int] = None,
    ):
        super().__init__()

        self.fd_source = fd_source
        self.batch_size = batch_size
        self.max_rul = max_rul
        self.window_size = window_size or CMAPSSLoader.WINDOW_SIZES[self.fd_source]
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select

        self.hparams = {
            "fd_source": self.fd_source,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "percent_fail_runs": self.percent_fail_runs,
            "max_rul": self.max_rul,
        }

        self.cmapss = {}
        for fd in range(1, 5):
            self.cmapss[fd] = self._get_cmapss(fd)

    def _get_cmapss(self, fd):
        fail_runs = self.percent_fail_runs if fd == self.fd_source else None
        cmapss = CMAPSSDataModule(
            fd,
            self.batch_size,
            self.window_size,
            self.max_rul,
            None,
            fail_runs,
            self.feature_select,
        )

        return cmapss

    def prepare_data(self, *args, **kwargs):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        for cmapss_fd in self.cmapss.values():
            cmapss_fd.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.cmapss[self.fd_source].train_dataloader()

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.cmapss[self.fd_source].val_dataloader()

    def test_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        test_dataloaders = []
        for fd_target in range(1, 5):
            target_dl = self.cmapss[fd_target].test_dataloader()
            test_dataloaders.append(target_dl)

        return test_dataloaders


class PretrainingBaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source: int,
        num_samples: int,
        batch_size: int,
        window_size: int = None,
        max_rul: int = 125,
        min_distance: int = 1,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        feature_select: List[int] = None,
        truncate_val: bool = False,
        distance_mode: str = "linear",
    ):
        super().__init__()

        self.fd_source = fd_source
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.min_distance = min_distance
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_val = truncate_val
        self.distance_mode = distance_mode

        self.fails_source_loader = CMAPSSLoader(
            self.fd_source,
            window_size,
            self.max_rul,
            None,
            self.percent_fail_runs,
            self.feature_select,
            self.truncate_val,
        )
        unfailed_runs = self._get_unbroken_runs(self.percent_fail_runs)
        self.broken_source_loader = CMAPSSLoader(
            self.fd_source,
            window_size,
            self.max_rul,
            self.percent_broken,
            unfailed_runs,
            self.feature_select,
            self.truncate_val,
        )

        self.window_size = self.broken_source_loader.window_size

        self.source = CMAPSSDataModule.from_loader(
            self.broken_source_loader, self.batch_size
        )

        self.hparams = {
            "fd_source": self.fd_source,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "min_distance": self.min_distance,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_val": self.truncate_val,
            "distance_mode": self.distance_mode,
        }

    def _get_unbroken_runs(self, fail_runs):
        if fail_runs is None or isinstance(fail_runs, float):
            unfailed_runs = None
        else:
            run_idx = range(CMAPSSLoader.NUM_TRAIN_RUNS[self.fd_source])
            unfailed_runs = list(set(run_idx).difference(fail_runs))

        return unfailed_runs

    def prepare_data(self, *args, **kwargs):
        self.broken_source_loader.prepare_data()

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
            [self.broken_source_loader, self.fails_source_loader],
            split,
            num_samples,
            min_distance,
            deterministic,
            mode=self.distance_mode,
        )

        return paired
