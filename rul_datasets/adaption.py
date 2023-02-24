"""Higher-order data modules to run unsupervised domain adaption experiments."""

import warnings
from copy import deepcopy
from typing import List, Optional, Any, Tuple, Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co

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
        >>> fd1 = rul_datasets.CmapssReader(fd=1, window_size=20)
        >>> fd2 = rul_datasets.CmapssReader(fd=2, percent_broken=0.8)
        >>> source = rul_datasets.RulDataModule(fd1, 32)
        >>> target = rul_datasets.RulDataModule(fd2, 32)
        >>> dm = rul_datasets.DomainAdaptionDataModule(source, target)
        >>> train_1_2 = dm.train_dataloader()
        >>> val_1, val_2, paired_val_1_2 = dm.val_dataloader()
        >>> test_1, test_2, paired_test_1_2 = dm.test_dataloader()
    """

    def __init__(
        self, source: RulDataModule, target: RulDataModule, paired_val: bool = False
    ) -> None:
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
            paired_val: Whether to include paired data in validation.
        """
        super().__init__()

        self.source = source
        self.target = target
        self.paired_val = paired_val
        self.batch_size = source.batch_size

        self.target_truncated = deepcopy(self.target.reader)
        self.target_truncated.truncate_val = True

        self._check_compatibility()

        self.save_hyperparameters(
            {
                "fd_source": self.source.reader.fd,
                "fd_target": self.target.reader.fd,
                "batch_size": self.batch_size,
                "window_size": self.source.reader.window_size,
                "max_rul": self.source.reader.max_rul,
                "percent_broken": self.target.reader.percent_broken,
                "percent_fail_runs": self.target.reader.percent_fail_runs,
            }
        )

    def _check_compatibility(self):
        self.source.check_compatibility(self.target)
        self.target.reader.check_compatibility(self.target_truncated)
        if self.source.reader.fd == self.target.reader.fd:
            raise ValueError(
                f"FD of source and target has to be different for "
                f"domain adaption, but is {self.source.reader.fd} both times."
            )
        if self.target.reader.percent_broken is None:
            warnings.warn(
                "The target domain is not truncated by 'percent_broken'."
                "This may lead to unrealistically good results."
                "If this was intentional, please set `percent_broken` "
                "to 1.0 to silence this warning."
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

        By default, two data loaders are returned, which correspond to the source
        and the target validation data loader. An optional third is a data loader of a
        [PairedRulDataset][rul_datasets.core.PairedRulDataset] using both source and
        target is returned if `paired_val` was set to `True` in the constructor.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        Returns:
            The source, target and an optional paired validation data loader.
        """
        loaders = [
            self.source.val_dataloader(*args, **kwargs),
            self.target.val_dataloader(*args, **kwargs),
        ]
        if self.paired_val:
            loaders.append(
                DataLoader(
                    self._get_paired_dataset(),
                    batch_size=self.batch_size,
                    pin_memory=True,
                )
            )

        return loaders

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
    """
    A torch [dataset][torch.utils.data.Dataset] for unsupervised domain adaption. The
    dataset takes a labeled source and one or multiple unlabeled target [dataset][
    torch.utils.data.Dataset] and combines them.

    For each label/features pair from the source dataset, a random sample of features
    is drawn from each target dataset. The datasets are supposed to provide a sample
    as a tuple of tensors. The target datasets' labels are assumed to be the last
    element of the tuple and are omitted. The datasets length is determined by the
    source dataset. This setup can be used to train with common unsupervised domain
    adaption methods like DAN, DANN or JAN.

    Examples:
        >>> import torch
        >>> import rul_datasets
        >>> source = torch.utils.data.TensorDataset(torch.randn(10), torch.randn(10))
        >>> target = torch.utils.data.TensorDataset(torch.randn(10), torch.randn(10))
        >>> dataset = rul_datasets.adaption.AdaptionDataset(source, target)
        >>> source_features, source_label, target_features = dataset[0]
    """

    _unlabeled_idx: np.ndarray
    _get_unlabeled_idx: Callable

    def __init__(
        self, labeled: Dataset, *unlabeled: Dataset, deterministic: bool = False
    ) -> None:
        """
        Create a new adaption data set from a labeled source and one or multiple
        unlabeled target dataset.

        By default, a random sample is drawn from each target dataset when a source
        sample is accessed. This is the recommended setting for training. To
        deactivate this behavior and fix the pairing of source and target samples,
        set `deterministic` to `True`. This is the recommended setting for evaluation.

        Args:
            labeled: The dataset from the labeled domain.
            unlabeled: The dataset(s) from the unlabeled domain(s).
            deterministic: Return the same target sample for each source sample.
        """
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.deterministic = deterministic
        self._unlabeled_len = [len(ul) for ul in self.unlabeled]

        if self.deterministic:
            self._rng = np.random.default_rng(seed=42)
            size = (len(self), len(self._unlabeled_len))
            self._unlabeled_idx = self._rng.integers(0, self._unlabeled_len, size)
            self._get_unlabeled_idx = self._get_deterministic_unlabeled_idx
        else:
            self._rng = np.random.default_rng()
            self._get_unlabeled_idx = self._get_random_unlabeled_idx

    def _get_random_unlabeled_idx(self, _: int) -> np.ndarray:
        return self._rng.integers(0, self._unlabeled_len)

    def _get_deterministic_unlabeled_idx(self, idx: int) -> np.ndarray:
        return self._unlabeled_idx[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        item = self.labeled[idx]
        for unlabeled, ul_idx in zip(self.unlabeled, self._get_unlabeled_idx(idx)):
            item += unlabeled[ul_idx][:-1]  # drop label tensor in last position

        return item

    def __len__(self) -> int:
        return len(self.labeled)  # type: ignore


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

        self.target_loader = self.target.reader
        self.source_loader = self.source.reader

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
