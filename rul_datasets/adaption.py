"""Higher-order data modules to run unsupervised domain adaption experiments."""

import warnings
from copy import deepcopy
from typing import List, Optional, Any, Tuple, Callable, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset, TensorDataset

from rul_datasets import utils
from rul_datasets.core import PairedRulDataset, RulDataModule


class DomainAdaptionDataModule(pl.LightningDataModule):
    """
    A higher-order [data module][pytorch_lightning.core.LightningDataModule] used for
    unsupervised domain adaption of a labeled source to an unlabeled target domain.
    The training data of both domains is wrapped in a [AdaptionDataset]
    [rul_datasets.adaption.AdaptionDataset] which provides a random sample of the
    target domain with each sample of the source domain. It provides the validation and
    test splits of both domains, and optionally a [paired dataset]
    [rul_datasets.core.PairedRulDataset] for both.

    Examples:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.CmapssReader(fd=1, window_size=20)
        >>> fd2 = rul_datasets.CmapssReader(fd=2, percent_broken=0.8)
        >>> source = rul_datasets.RulDataModule(fd1, 32)
        >>> target = rul_datasets.RulDataModule(fd2, 32)
        >>> dm = rul_datasets.DomainAdaptionDataModule(source, target)
        >>> train_1_2 = dm.train_dataloader()
        >>> val_1, val_2 = dm.val_dataloader()
        >>> test_1, test_2 = dm.test_dataloader()
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


class LatentAlignDataModule(DomainAdaptionDataModule):
    """
    A higher-order [data module][pytorch_lightning.core.LightningDataModule] based on
    [DomainAdaptionDataModule][rul_datasets.adaption.DomainAdaptionDataModule].

    It is specifically made to work with the latent space alignment approach by Zhang
    et al. The training data of both domains is wrapped in a [AdaptionDataset]
    [rul_datasets.adaption.AdaptionDataset] which splits the data into healthy and
    degrading. For each sample of degrading source data, a random sample of degrading
    target data and healthy sample of either source or target data is drawn. The
    number of steps in degradation are supplied for each degrading sample, as well.
    The data module also provides the validation and test splits of both domains, and
    optionally a [paired dataset][rul_datasets.core.PairedRulDataset] for both.

    Examples:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.CmapssReader(fd=1, window_size=20)
        >>> fd2 = rul_datasets.CmapssReader(fd=2, percent_broken=0.8)
        >>> source = rul_datasets.RulDataModule(fd1, 32)
        >>> target = rul_datasets.RulDataModule(fd2, 32)
        >>> dm = rul_datasets.LatentAlignDataModule(source, target)
        >>> train_1_2 = dm.train_dataloader()
        >>> val_1, val_2 = dm.val_dataloader()
        >>> test_1, test_2 = dm.test_dataloader()
    """

    def __init__(
        self,
        source: RulDataModule,
        target: RulDataModule,
        paired_val: bool = False,
        split_by_max_rul: bool = False,
        split_by_steps: Optional[int] = None,
    ) -> None:
        """
        Create a new latent align data module from a source and target
        [RulDataModule][rul_datasets.RulDataModule]. The source domain is considered
        labeled and the target domain unlabeled.

        The source and target data modules are checked for compatability (see
        [RulDataModule][rul_datasets.core.RulDataModule.check_compatibility]). These
        checks include that the `fd` differs between them, as they come from the same
        domain otherwise.

        The healthy and degrading data can be split by either maximum RUL value or
        the number of time steps. See [split_healthy]
        [rul_datasets.adaption.split_healthy] for more information.

        Args:
            source: The data module of the labeled source domain.
            target: The data module of the unlabeled target domain.
            paired_val: Whether to include paired data in validation.
            split_by_max_rul: Whether to split healthy and degrading by max RUL value.
            split_by_steps: Split the healthy and degrading data after this number of
                            time steps.
        """
        super().__init__(source, target, paired_val)

        if not split_by_max_rul and (split_by_steps is None):
            raise ValueError(
                "Either 'split_by_max_rul' or 'split_by_steps' need to be set."
            )

        self.split_by_max_rul = split_by_max_rul
        self.split_by_steps = split_by_steps

    def _to_dataset(self, split: str) -> "AdaptionDataset":
        source_healthy, source_degraded = split_healthy(
            *self.source.reader.load_split(split), by_max_rul=True
        )
        target_healthy, target_degraded = split_healthy(
            *self.target.reader.load_split(split),
            self.split_by_max_rul,
            self.split_by_steps,
        )
        healthy: Dataset = ConcatDataset([source_healthy, target_healthy])
        dataset = AdaptionDataset(source_degraded, target_degraded, healthy)

        return dataset


def split_healthy(
    features: List[np.ndarray],
    targets: List[np.ndarray],
    by_max_rul: bool = False,
    by_steps: Optional[int] = None,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Split the feature and target time series into healthy and degrading parts and
    return a dataset of each.

    If `by_max_rul` is set to `True` the time steps with the maximum RUL value in
    each time series is considered healthy. This option is intended for labeled data
    with piece-wise linear RUL functions. If `by_steps` is set to an integer,
    the first `by_steps` time steps of each series are considered healthy. This
    option is intended for unlabeled data or data with a linear RUL function.

    One option has to be set and both are mutually exclusive.

    Args:
        features: List of feature time series.
        targets: List of target time series.
        by_max_rul: Whether to split healthy and degrading data by max RUL value.
        by_steps: Split healthy and degrading data after this number of time steps.
    Returns:
        healthy: Dataset of healthy data.
        degrading: Dataset of degrading data.
    """
    if not by_max_rul and (by_steps is None):
        raise ValueError("Either 'by_max_rul' or 'by_steps' need to be set.")

    healthy = []
    degraded = []
    for feature, target in zip(features, targets):
        # get index of last max RUL or use step
        split_idx = [np.argmax(target[::-1]) if by_max_rul else by_steps]
        healthy_feature, degraded_feature = np.split(feature, split_idx)  # type: ignore
        healthy_target, degraded_target = np.split(target, split_idx)  # type: ignore
        degradation_steps = np.arange(len(degraded_target))
        healthy.append((healthy_feature, healthy_target))
        degraded.append((degraded_feature, degradation_steps, degraded_target))

    healthy_dataset = _to_dataset(healthy)
    degraded_dataset = _to_dataset(degraded)

    return healthy_dataset, degraded_dataset


def _to_dataset(data: Sequence[Tuple[np.ndarray, ...]]) -> TensorDataset:
    tensor_data = [torch.cat(h) for h in utils.to_tensor(*zip(*data))]
    dataset = TensorDataset(*tensor_data)

    return dataset


class AdaptionDataset(Dataset):
    """
    A torch [dataset][torch.utils.data.Dataset] for unsupervised domain adaption. The
    dataset takes a labeled source and one or multiple unlabeled target [dataset]
    [torch.utils.data.Dataset] and combines them.

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
            *unlabeled: The dataset(s) from the unlabeled domain(s).
            deterministic: Return the same target sample for each source sample.
        """
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.deterministic = deterministic
        self._unlabeled_len = [len(ul) for ul in self.unlabeled]  # type: ignore

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
