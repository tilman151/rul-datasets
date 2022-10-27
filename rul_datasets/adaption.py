"""Higher-order data modules to run unsupervised domain adaption experiments."""

import warnings
from copy import deepcopy
from typing import List, Optional, Any

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from rul_datasets.core import PairedRulDataset, RulDataModule


class DomainAdaptionDataModule(pl.LightningDataModule):
    """
    A higher-order [data module][pytorch_lightning.core.LightningDataModule] used for
    unsupervised domain adaption of a labeled source to an unlabeled target domain.
    The training data of both domains is wrapped in a [AdaptionDataset]
    [rul_datasets.adaption.AdaptionDataset] which provides a random sample of the
    target domain with each sample of the source domain. It provides the validation and
    test splits of both domains, and a [paired dataset]
    [rul_datasets.core.PairedRulDataset] for both.

    Examples:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.CmapssLoader(fd=1, window_size=20)
        >>> fd2 = rul_datasets.CmapssLoader(fd=2, percent_broken=0.8)
        >>> source = rul_datasets.RulDataModule(fd1, 32)
        >>> target = rul_datasets.RulDataModule(fd2, 32)
        >>> dm = rul_datasets.DomainAdaptionDataModule(source, target)
        >>> train_1_2 = dm.train_dataloader()
        >>> val_1, val_2, paired_val_1_2 = dm.val_dataloader()
        >>> test_1, test_2, paired_test_1_2 = dm.test_dataloader()
    """

    def __init__(self, source: RulDataModule, target: RulDataModule) -> None:
        """
        Create a new domain adaption data module from a source and target
        [RulDataModule][rul_datasets.RulDataModule]. The source domain is considered
        labeled and the target domain unlabeled.

        The source and target data modules are checked for compatability (see
        [RulDataModule][rul_datasets.core.RulDataModule.check_compatibility]). These
        checks include that the `fd` differs between them, as they come from the same
        domain otherwise.

        Args:
            source: The data module of the labeled source domain.
            target: The data module of the unlabeled target domain.
        """
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
                f"domain adaption, but is {self.source.loader.fd} both times."
            )

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Download and pre-process the underlying data.

        This calls the `prepare_data` function for source and target domain. All
        previously completed preparation steps are skipped. It is called
        automatically by `pytorch_lightning` and executed on the first GPU in
        distributed mode.

        Args:
            *args: Passed down to each data module's `prepare_data` function.
            **kwargs: Passed down to each data module's `prepare_data` function..
        """
        self.source.prepare_data(*args, **kwargs)
        self.target.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load source and target domain into memory.

        Args:
            stage: Passed down to each data module's `setup` function.
        """
        self.source.setup(stage)
        self.target.setup(stage)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a data loader of an [AdaptionDataset]
        [rul_datasets.adaption.AdaptionDataset] using source and target domain.

        The data loader is configured to shuffle the data. The `pin_memory` option is
        activated to achieve maximum transfer speed to the GPU.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        Returns:
            The training data loader
        """
        return DataLoader(
            self._to_dataset("dev"),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> List[DataLoader]:
        """
        Create a data loader of the source, target and paired validation data.

        The first two data loaders are the return values of `source.val_dataloader`
        and `target.val_dataloader`. The third is a data loader of a
        [PairedRulDataset][rul_datasets.core.PairedRulDataset] using both source and
        target.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        Returns:
            The source, target and paired validation data loader.
        """
        return [
            self.source.val_dataloader(*args, **kwargs),
            self.target.val_dataloader(*args, **kwargs),
            DataLoader(
                self._get_paired_dataset(), batch_size=self.batch_size, pin_memory=True
            ),
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> List[DataLoader]:
        """
        Create a data loader of the source and target test data.

        The data loaders are the return values of `source.test_dataloader`
        and `target.test_dataloader`.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        Returns:
            The source and target test data loader.
        """
        return [
            self.source.test_dataloader(*args, **kwargs),
            self.target.test_dataloader(*args, **kwargs),
        ]

    def _to_dataset(self, split: str) -> "AdaptionDataset":
        source = self.source.to_dataset(split)
        target = self.target.to_dataset(split)
        dataset = AdaptionDataset(source, target)

        return dataset

    def _get_paired_dataset(self) -> PairedRulDataset:
        paired = PairedRulDataset(
            [self.target_truncated],
            "val",
            num_samples=25000,
            min_distance=1,
            deterministic=True,
        )

        return paired


class AdaptionDataset(Dataset):
    """A torch [dataset][torch.utils.data.Dataset] for unsupervised domain adaption."""

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
        source: RulDataModule,
        target: RulDataModule,
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

    def _get_paired_dataset(self, split: str) -> PairedRulDataset:
        deterministic = split == "val"
        min_distance = 1 if split == "val" else self.min_distance
        num_samples = 50000 if split == "val" else self.num_samples
        paired = PairedRulDataset(
            [self.source_loader, self.target_loader],
            split,
            num_samples,
            min_distance,
            deterministic,
            mode=self.distance_mode,
        )

        return paired
