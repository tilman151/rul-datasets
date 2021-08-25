from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from rul_datasets.cmapss import CMAPSSDataModule, PairedCMAPSS
from rul_datasets.loader import CMAPSSLoader


class AdaptionDataset(Dataset):
    def __init__(self, source, target, deterministic=False):
        self.source = source
        self.target = target
        self.deterministic = deterministic
        self._target_len = len(target)

        self._rng = np.random.default_rng(seed=42)
        if self.deterministic:
            self._get_target_idx = self._get_deterministic_target_idx
            self._target_idx = [
                self._get_random_target_idx(_) for _ in range(len(self))
            ]
        else:
            self._get_target_idx = self._get_random_target_idx
            self._target_idx = None

    def _get_random_target_idx(self, _):
        return self._rng.integers(0, self._target_len)

    def _get_deterministic_target_idx(self, idx):
        return self._target_idx[idx]

    def __getitem__(self, idx):
        target_idx = self._get_target_idx(idx)
        source, source_label = self.source[idx]
        target, _ = self.target[target_idx]

        return source, source_label, target

    def __len__(self):
        return len(self.source)


class DomainAdaptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source: int,
        fd_target: int,
        batch_size: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_fail_runs: float = None,
        percent_broken: float = None,
        feature_select: float = None,
    ):
        super().__init__()

        self.fd_source = fd_source
        self.fd_target = fd_target
        self.batch_size = batch_size
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select

        self.target = CMAPSSDataModule(
            self.fd_target,
            self.batch_size,
            window_size,
            self.max_rul,
            self.percent_broken,
            self.percent_fail_runs,
            self.feature_select,
        )
        self.window_size = self.target.window_size

        self.source = CMAPSSDataModule(
            self.fd_source,
            self.batch_size,
            self.window_size,
            self.max_rul,
            None,
            None,
            self.feature_select,
        )
        self.target_truncated = CMAPSSLoader(
            self.fd_target,
            self.window_size,
            self.max_rul,
            self.percent_broken,
            self.percent_fail_runs,
            self.feature_select,
            truncate_val=True,
        )

        self.hparams = {
            "fd_source": self.fd_source,
            "fd_target": self.fd_target,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
        }

    def prepare_data(self, *args, **kwargs):
        self.source.prepare_data(*args, **kwargs)
        self.target.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)
        self.target.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._to_dataset("dev"),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        return [
            self.source.val_dataloader(*args, **kwargs),
            self.target.val_dataloader(*args, **kwargs),
            DataLoader(
                self._get_paired_dataset(), batch_size=self.batch_size, pin_memory=True
            ),
        ]

    def test_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        return [
            self.source.test_dataloader(*args, **kwargs),
            self.target.test_dataloader(*args, **kwargs),
        ]

    def _to_dataset(self, split: str) -> AdaptionDataset:
        source = self.source.to_dataset(split)
        target = self.target.to_dataset(split)
        dataset = AdaptionDataset(source, target)

        return dataset

    def _get_paired_dataset(self) -> PairedCMAPSS:
        paired = PairedCMAPSS(
            [self.target_truncated],
            "val",
            num_samples=25000,
            min_distance=1,
            deterministic=True,
        )

        return paired


class PretrainingAdaptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd_source: int,
        fd_target: int,
        num_samples: int,
        batch_size: int,
        window_size: int = None,
        max_rul: int = 125,
        min_distance: int = 1,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        feature_select: List[int] = None,
        truncate_target_val: bool = False,
        distance_mode: str = "linear",
    ):
        super().__init__()

        self.fd_source = fd_source
        self.fd_target = fd_target
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.min_distance = min_distance
        self.max_rul = max_rul
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs
        self.feature_select = feature_select
        self.truncate_target_val = truncate_target_val
        self.distance_mode = distance_mode

        self.target_loader = CMAPSSLoader(
            self.fd_target,
            window_size,
            self.max_rul,
            self.percent_broken,
            self.percent_fail_runs,
            self.feature_select,
            self.truncate_target_val,
        )
        self.window_size = self.target_loader.window_size

        self.source_loader = CMAPSSLoader(
            self.fd_source,
            self.window_size,
            self.max_rul,
            None,
            None,
            self.feature_select,
        )

        self.source = CMAPSSDataModule.from_loader(self.source_loader, self.batch_size)
        self.target = CMAPSSDataModule.from_loader(self.target_loader, self.batch_size)

        self.hparams = {
            "fd_source": self.fd_source,
            "fd_target": self.fd_target,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "min_distance": self.min_distance,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_target_val": self.truncate_target_val,
            "distance_mode": self.distance_mode,
        }

    def prepare_data(self, *args, **kwargs):
        self.source_loader.prepare_data()
        self.target_loader.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.source.setup(stage)
        self.target.setup(stage)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self._get_paired_dataset("dev"), batch_size=self.batch_size, pin_memory=True
        )

    def val_dataloader(self, *args, **kwargs) -> List[DataLoader]:
        combined_loader = DataLoader(
            self._get_paired_dataset("val"), batch_size=self.batch_size, pin_memory=True
        )
        source_loader = self.source.val_dataloader()
        target_loader = self.target.val_dataloader()

        return [combined_loader, source_loader, target_loader]

    def _get_paired_dataset(self, split: str) -> PairedCMAPSS:
        deterministic = split == "val"
        min_distance = 1 if split == "val" else self.min_distance
        num_samples = 50000 if split == "val" else self.num_samples
        paired = PairedCMAPSS(
            [self.source_loader, self.target_loader],
            split,
            num_samples,
            min_distance,
            deterministic,
            mode=self.distance_mode,
        )

        return paired
