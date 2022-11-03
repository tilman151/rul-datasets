"""A module with higher-order data modules for semi-supervised learning."""
import warnings
from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from rul_datasets.adaption import AdaptionDataset
from rul_datasets.core import RulDataModule


class SemiSupervisedDataModule(pl.LightningDataModule):
    """
    A higher-order [data module][pytorch_lightning.core.LightningDataModule] used for
    semi-supervised learning with a labeled data module and an unlabeled one. It
    makes sure that both data modules come from the same sub-dataset.

    Examples:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.CmapssReader(fd=1, window_size=20, percent_fail_runs=0.5)
        >>> fd1_complement = fd1.get_complement(percent_broken=0.8)
        >>> labeled = rul_datasets.RulDataModule(fd1, 32)
        >>> unlabeled = rul_datasets.RulDataModule(fd1_complement, 32)
        >>> dm = rul_datasets.SemiSupervisedDataModule(labeled, unlabeled)
        >>> train_ssl = dm.train_dataloader()
        >>> val = dm.val_dataloader()
        >>> test = dm.test_dataloader()
    """

    def __init__(self, labeled: RulDataModule, unlabeled: RulDataModule) -> None:
        """
        Create a new semi-supervised data module from a labeled and unlabeled
        [RulDataModule][rul_datasets.RulDataModule].

        The both data modules are checked for compatability (see[RulDataModule]
        [rul_datasets.core.RulDataModule.check_compatibility]). These
        checks include that the `fd` match between them.

        Args:
            labeled: The data module of the labeled dataset.
            unlabeled: The data module of the unlabeled dataset.
        """
        super().__init__()

        self.labeled = labeled
        self.unlabeled = unlabeled
        self.batch_size = labeled.batch_size

        self._check_compatibility()

        self.save_hyperparameters(
            {
                "fd": self.labeled.reader.fd,
                "batch_size": self.batch_size,
                "window_size": self.labeled.reader.window_size,
                "max_rul": self.labeled.reader.max_rul,
                "percent_broken_unlabeled": self.unlabeled.reader.percent_broken,
                "percent_fail_runs_labeled": self.labeled.reader.percent_fail_runs,
            }
        )

    def _check_compatibility(self) -> None:
        self.labeled.check_compatibility(self.unlabeled)
        if not self.labeled.reader.fd == self.unlabeled.reader.fd:
            raise ValueError(
                "FD of source and target has to be the same for "
                "semi-supervised learning, but they are "
                f"{self.labeled.reader.fd} and {self.unlabeled.reader.fd}."
            )
        if self.unlabeled.reader.percent_broken is None:
            warnings.warn(
                "The unlabeled data is not truncated by 'percent_broken'."
                "This may lead to unrealistically good results."
                "If this was intentional, please set `percent_broken` "
                "to 1.0 to silence this warning."
            )
        if not self.labeled.is_mutually_exclusive(self.unlabeled):
            warnings.warn(
                "The data modules are not mutually exclusive. "
                "This means there is an overlap between labeled and "
                "unlabeled data, which should not be that case for "
                "semi-supervised learning. You can check this by calling "
                "'is_mutually_exclusive' on a reader or RulDataModule."
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
        self.labeled.prepare_data(*args, **kwargs)
        self.unlabeled.prepare_data(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load labeled and unlabeled data into memory.

        Args:
            stage: Passed down to each data module's `setup` function.
        """
        self.labeled.setup(stage)
        self.unlabeled.setup(stage)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a data loader of an [AdaptionDataset]
        [rul_datasets.adaption.AdaptionDataset] using labeled and unlabeled.

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

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a data loader of the labeled validation data.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        Returns:
            The labeled validation data loader.
        """
        return self.labeled.val_dataloader()

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a data loader of the labeled test data.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        Returns:
            The labeled test data loader.
        """
        return self.labeled.test_dataloader()

    def _to_dataset(self, split: str) -> "AdaptionDataset":
        labeled = self.labeled.to_dataset(split)
        unlabeled = self.unlabeled.to_dataset(split)
        dataset = AdaptionDataset(labeled, unlabeled)

        return dataset
