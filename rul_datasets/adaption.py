import warnings
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from rul_datasets.cmapss import CMAPSSDataModule, PairedCMAPSS


class DomainAdaptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        source: CMAPSSDataModule,
        target: CMAPSSDataModule,
    ):
        super().__init__()

        self.source = source
        self.target = target
        self.batch_size = source.batch_size

        self.target_truncated = deepcopy(self.target.loader)
        self.target_truncated.truncate_val = True

        self._check_compatibility()

        self.save_hyperparameters(
            {
                "fd_source": self.source.loader.fd,
                "fd_target": self.target.loader.fd,
                "batch_size": self.batch_size,
                "window_size": self.source.loader.window_size,
                "max_rul": self.source.loader.max_rul,
                "percent_broken": self.target.loader.percent_broken,
                "percent_fail_runs": self.target.loader.percent_fail_runs,
            }
        )

    def _check_compatibility(self):
        self.source.check_compatibility(self.target)
        self.target.loader.check_compatibility(self.target_truncated)
        if self.source.loader.fd == self.target.loader.fd:
            raise ValueError(
                f"FD of source and target has to be different for "
                f"domain adaption, but is {self.source.loader.fd} bot times."
            )

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

    def _to_dataset(self, split: str) -> "AdaptionDataset":
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


class PretrainingAdaptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        source: CMAPSSDataModule,
        target: CMAPSSDataModule,
        num_samples: int,
        min_distance: int = 1,
        distance_mode: str = "linear",
    ):
        super().__init__()

        self.source = source
        self.target = target
        self.num_samples = num_samples
        self.batch_size = source.batch_size
        self.min_distance = min_distance
        self.distance_mode = distance_mode

        self.target_loader = self.target.loader
        self.source_loader = self.source.loader

        self._check_compatibility()

        self.save_hyperparameters(
            {
                "fd_source": self.source_loader.fd,
                "fd_target": self.target_loader.fd,
                "num_samples": self.num_samples,
                "batch_size": self.batch_size,
                "window_size": self.source_loader.window_size,
                "max_rul": self.source_loader.max_rul,
                "min_distance": self.min_distance,
                "percent_broken": self.target_loader.percent_broken,
                "percent_fail_runs": self.target_loader.percent_fail_runs,
                "truncate_target_val": self.target_loader.truncate_val,
                "distance_mode": self.distance_mode,
            }
        )

    def _check_compatibility(self):
        self.source.check_compatibility(self.target)
        if self.source_loader.fd == self.target_loader.fd:
            raise ValueError(
                f"FD of source and target has to be different for "
                f"domain adaption, but is {self.source_loader.fd} bot times."
            )
        if (
            self.target_loader.percent_broken is None
            or self.target_loader.percent_broken == 1.0
        ):
            raise ValueError(
                "Target data needs a percent_broken smaller than 1 for pre-training."
            )
        if (
            self.source_loader.percent_broken is not None
            and self.source_loader.percent_broken < 1.0
        ):
            raise ValueError(
                "Source data cannot have a percent_broken smaller than 1, "
                "otherwise it would not be failed, labeled data."
            )
        if not self.target_loader.truncate_val:
            warnings.warn(
                "Validation data of unfailed runs is not truncated. "
                "The validation metrics will not be valid."
            )

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
