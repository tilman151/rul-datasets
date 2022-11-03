"""A module with higher-order data modules for semi-supervised learning."""
import warnings

from rul_datasets.core import RulDataModule
from rul_datasets.adaption import DomainAdaptionDataModule


class SemiSupervisedDataModule(DomainAdaptionDataModule):
    """
    A higher-order [data module][pytorch_lightning.core.LightningDataModule] used for
    semi-supervised learning with a labeled data module and an unlabeled one. It
    behaves exactly the same as [DomainAdaptionDataModule]
    [rul_datasets.adaption.DomainAdaptionDataModule] but makes sure that both data
    modules come from the same sub-dataset.

    Examples:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.CmapssReader(fd=1, window_size=20, percent_fail_runs=0.5)
        >>> fd1_complement = fd1.get_complement(percent_broken=0.8)
        >>> labeled = rul_datasets.RulDataModule(fd1, 32)
        >>> unlabeled = rul_datasets.RulDataModule(fd1_complement, 32)
        >>> dm = rul_datasets.SemiSupervisedDataModule(labeled, unlabeled)
        >>> train_1_2 = dm.train_dataloader()
        >>> val_1, val_2, paired_val_1_2 = dm.val_dataloader()
        >>> test_1, test_2, paired_test_1_2 = dm.test_dataloader()
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
        super().__init__(labeled, unlabeled)

    def _check_compatibility(self) -> None:
        self.source.check_compatibility(self.target)
        self.target.reader.check_compatibility(self.target_truncated)
        if not self.source.reader.fd == self.target.reader.fd:
            raise ValueError(
                "FD of source and target has to be the same for "
                "semi-supervised learning, but they are "
                f"{self.source.reader.fd} and {self.target.reader.fd}."
            )
        if self.target.reader.percent_broken is None:
            warnings.warn(
                "The unlabeled data is not truncated by 'percent_broken'."
                "This may lead to unrealistically good results."
                "If this was intentional, please set `percent_broken` "
                "to 1.0 to silence this warning."
            )
        if not self.source.is_mutually_exclusive(self.target):
            warnings.warn(
                "The data modules are not mutually exclusive. "
                "This means there is an overlap between labeled and "
                "unlabeled data, which should not be that case for "
                "semi-supervised learning. You can check this by calling "
                "'is_mutually_exclusive' on a reader or RulDataModule."
            )
