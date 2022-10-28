"""Higher-order data modules to establish a baseline for transfer learning and domain
adaption experiments. """

import warnings
from copy import deepcopy
from typing import List, Optional, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from rul_datasets.core import PairedRulDataset, RulDataModule


class BaselineDataModule(pl.LightningDataModule):
    """
    A higher-order [data module][pytorch_lightning.core.LightningDataModule] that
    takes a [RulDataModule][rul_datasets.core.RulDataModule]. It provides the
    training and validation splits of the sub-dataset selected in the underlying data
    module but provides the test splits of all available subsets of the dataset. This
    makes it easy to evaluate the generalization of a supervised model on all
    sub-datasets.

    Examples:
        >>> import rul_datasets
        >>> cmapss = rul_datasets.reader.CmapssReader(fd=1)
        >>> dm = rul_datasets.RulDataModule(cmapss, batch_size=32)
        >>> baseline_dm = rul_datasets.BaselineDataModule(dm)
        >>> train_fd1 = baseline_dm.train_dataloader()
        >>> val_fd1 = baseline_dm.val_dataloader()
        >>> test_fd1, test_fd2, test_fd3, test_fd4 = baseline_dm.test_dataloader()

    """

    def __init__(self, data_module: RulDataModule) -> None:
        """
        Create a new baseline data module from a [RulDataModule]
        [rul_datasets.RulDataModule].

        It will provide a data loader of the underlying data module's training and
        validation splits. Additionally, it provides a data loader of the test split
        of all sub-datasets.

        The data module keeps the configuration made in the underlying data module.
        The same configuration is then passed on to create RulDataModules for all
        sub-datasets, beside `percent_fail_runs` and `percent_broken`.

        Args:
            data_module: the underlying RulDataModule
        """
        super().__init__()

        self.data_module = data_module
        hparams = self.data_module.hparams
        self.save_hyperparameters(hparams)

        self.subsets = {}
        for fd in self.data_module.fds:
            self.subsets[fd] = self._get_fd(fd)

    def _get_fd(self, fd):
        if fd == self.hparams["fd"]:
            dm = self.data_module
        else:
            loader = deepcopy(self.data_module.reader)
            loader.fd = fd
            loader.percent_fail_runs = None
            loader.percent_broken = None
            dm = RulDataModule(loader, self.data_module.batch_size)

        return dm

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Download and pre-process the underlying data.

        This calls the `prepare_data` function for all sub-datasets. All
        previously completed preparation steps are skipped. It is called
        automatically by `pytorch_lightning` and executed on the first GPU in
        distributed mode.

        Args:
            *args: Passed down to each data module's `prepare_data` function.
            **kwargs: Passed down to each data module's `prepare_data` function..
        """
        for dm in self.subsets.values():
            dm.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        """
        Load all splits as tensors into memory.

        Args:
            stage: Passed down to each data module's `setup` function.
        """
        for dm in self.subsets.values():
            dm.setup(stage)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """See [rul_datasets.core.RulDataModule.train_dataloader][]."""
        return self.data_module.train_dataloader(*args, **kwargs)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """See [rul_datasets.core.RulDataModule.val_dataloader][]."""
        return self.data_module.val_dataloader(*args, **kwargs)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> List[DataLoader]:
        """
        Return data loaders for all sub-datasets.

        Args:
            *args: Passed down to each data module.
            **kwargs: Passed down to each data module.
        Returns:
            The test dataloaders of all sub-datasets.
        """
        test_dataloaders = []
        for fd_target in self.data_module.fds:
            target_dl = self.subsets[fd_target].test_dataloader(*args, **kwargs)
            test_dataloaders.append(target_dl)

        return test_dataloaders


class PretrainingBaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        failed_data_module: RulDataModule,
        unfailed_data_module: RulDataModule,
        num_samples: int,
        min_distance: int = 1,
        distance_mode: str = "linear",
    ):
        super().__init__()

        self.failed_loader = failed_data_module.reader
        self.unfailed_loader = unfailed_data_module.reader
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

    def _get_paired_dataset(self, split: str) -> PairedRulDataset:
        deterministic = split == "val"
        min_distance = 1 if split == "val" else self.min_distance
        num_samples = 25000 if split == "val" else self.num_samples
        paired = PairedRulDataset(
            [self.unfailed_loader, self.failed_loader],
            split,
            num_samples,
            min_distance,
            deterministic,
            mode=self.distance_mode,
        )

        return paired
