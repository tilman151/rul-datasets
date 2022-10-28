"""This module contains the base class for all readers. It is only relevant to people
that want to extend this package with their own dataset. """
import abc
import os
from copy import deepcopy
from typing import Optional, Union, List, Dict, Any, Iterable, Tuple

import numpy as np
import torch

from rul_datasets.reader import truncating
from rul_datasets import utils

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


class AbstractReader(metaclass=abc.ABCMeta):
    """
    This reader is the abstract base class of all readers.

    In case you want to extend this library with a dataset of your own, you should
    create a subclass of `AbstractReader`. It defines the public interface that all
    data modules in this library use. Just inherit from this class implement the
    abstract functions, and you should be good to go.

    Please consider contributing your work afterwards to help the community.

    Examples:
        >>> import rul_datasets
        >>> class MyReader(rul_datasets.reader.AbstractReader):
        ...     def fds(self):
        ...         return [1]
        ...
        ...     def prepare_data(self):
        ...         pass
        ...
        ...     def default_window_size(self, fd):
        ...         return 30
        ...
        ...     def load_complete_split(self, split):
        ...         features = [np.random.randn(100, 2, 30) for _ in range(10)]
        ...         targets = [np.arange(100, 0, -1) for _ in range(10)]
        ...
        ...         return features, targets
        ...
        >>> my_reader = MyReader(fd=1)
        >>> features, targets = my_reader.load_split("dev")
        >>> features[0].shape
        torch.Size([100, 2, 30])
    """

    fd: int
    window_size: int
    max_rul: Optional[int]
    percent_broken: Optional[float]
    percent_fail_runs: Optional[Union[float, List[int], None]]
    truncate_val: bool

    _NUM_TRAIN_RUNS: Dict[int, int]

    def __init__(
        self,
        fd: int,
        window_size: Optional[int] = None,
        max_rul: Optional[int] = None,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Optional[Union[float, List[int]]] = None,
        truncate_val: bool = False,
    ) -> None:
        """
        Create a new reader. If your reader needs additional input arguments,
        create your own `__init__` function and call this one from within as `super(
        ).__init__(...)`.

        For more information about using readers refer to the [reader]
        [rul_datasets.reader] module page.

        Args:
            fd: Index of the selected sub-dataset
            window_size: Size of the sliding window. Defaults to 2560.
            max_rul: Maximum RUL value of targets.
            percent_broken: The maximum relative degradation per time series.
            percent_fail_runs: The percentage or index list of available time series.
            truncate_val: Truncate the validation data with `percent_broken`, too.
        """
        self.fd = fd
        self.window_size = window_size or self.default_window_size(self.fd)
        self.max_rul = max_rul
        self.truncate_val = truncate_val
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs

    @property
    def hparams(self) -> Dict[str, Any]:
        """A dictionary containing all input arguments of the constructor. This
        information is used by the data modules to log their hyperparameters in
        PyTorch Lightning."""
        return {
            "fd": self.fd,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_val": self.truncate_val,
        }

    @property
    @abc.abstractmethod
    def fds(self) -> List[int]:
        """The indices of available sub-datasets."""
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """Prepare the data. This function should take care of things that need to be
        done once, before the data can be used. This may include downloading,
        extracting or transforming the data, as well as fitting scalers. It is best
        practice to check if a preparation step was completed before to avoid
        repeating it unnecessarily."""
        raise NotImplementedError

    @abc.abstractmethod
    def default_window_size(self, fd: int) -> int:
        """
        The default window size of the data set. This may vary from sub-dataset to
        sub-dataset.

        Args:
            fd: The index of a sub-dataset.
        Returns:
            The default window size for the sub-dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load a complete split without truncation.

        This function should return the features and targets of the desired split.
        Both should be contained in a list of numpy arrays. Each of the arrays
        contains one time series. The features should have a shape of `[num_windows,
        num_channels, window_size]` and the targets a shape of `[num_windows]`.
        The features should be scaled as desired. The targets should be capped by
        `max_rul`.

        This function is used internally in [load_split]
        [rul_datasets.reader.abstract.AbstractReader.load_split] which takes care of
        truncation and conversion to tensors.

        Args:
            split: The name of the split to load.
        Returns:
            features: The complete, scaled features of the desired split.
            targets: The capped target values corresponding to the features.
        """
        raise NotImplementedError

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load a split as tensors and apply truncation to it.

        This function loads the scaled features and the targets of a split into
        memory. Afterwards, truncation is applied if the `split` is set to `dev`. The
        validation set is also truncated with `percent_broken` if `truncate_val` is
        set to `True`. At last, the data is transformed into [tensors][torch.Tensor].

        Args:
            split: The desired split to load.
        Returns:
            features: The scaled, truncated features of the desired split.
            targets: The truncated targets of the desired split.
        """
        features, targets = self.load_complete_split(split)
        if split == "dev":
            features, targets = truncating.truncate_runs(
                features, targets, self.percent_broken, self.percent_fail_runs
            )
        elif split == "val" and self.truncate_val:
            features, targets = truncating.truncate_runs(
                features, targets, self.percent_broken
            )
        tensor_feats, tensor_targets = utils.to_tensor(features, targets)

        return tensor_feats, tensor_targets

    def get_compatible(
        self,
        fd: Optional[int] = None,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Union[float, List[int], None] = None,
        truncate_val: Optional[bool] = None,
    ) -> "AbstractReader":
        """
        Create a new reader of the desired sub-dataset that is compatible to this one
        (see [check_compatibility]
        [rul_datasets.reader.abstract.AbstractReader.check_compatibility]). Useful for
        domain adaption.

        The values for `percent_broken`, `percent_fail_runs` and `truncate_val` of
        the new reader can be overridden.

        Args:
            fd: The index of the sub-dataset for the new reader.
            percent_broken: Override this value in the new reader.
            percent_fail_runs: Override this value in the new reader.
            truncate_val: Override this value in the new reader.
        Returns:
            A compatible reader with optional overrides.
        """
        other = deepcopy(self)
        if percent_broken is not None:
            other.percent_broken = percent_broken
        if percent_fail_runs is not None:
            other.percent_fail_runs = percent_fail_runs
        if truncate_val is not None:
            other.truncate_val = truncate_val

        if fd is not None:
            other.fd = fd
            window_size = min(self.window_size, self.default_window_size(other.fd))
        else:
            window_size = self.window_size
        self.window_size = window_size
        other.window_size = window_size

        return other

    def get_complement(
        self,
        percent_broken: Optional[float] = None,
        truncate_val: Optional[bool] = None,
    ) -> "AbstractReader":
        """
        Get a compatible reader that contains all development runs that are not in
        this reader (see [check_compatibility]
        [rul_datasets.reader.abstract.AbstractReader.check_compatibility]). Useful for
        semi-supervised learning.

        The new reader will contain the development runs that were discarded in this
        reader due to truncation through `percent_fail_runs`. If `percent_fail_runs`
        was not set or this reader contains all development runs, it returns a reader
        with an empty development set.

        The values for `percent_broken`, and `truncate_val` of the new reader can be
        overridden.

        Args:
            percent_broken: Override this value in the new reader.
            truncate_val: Override this value in the new reader.
        Returns:
            A compatible reader with all development runs missing in this one.
        """
        complement_idx = self._get_complement_idx()
        other = self.get_compatible(
            percent_broken=percent_broken,
            percent_fail_runs=complement_idx,
            truncate_val=truncate_val,
        )

        return other

    def _get_complement_idx(self) -> List[int]:
        num_runs = self._NUM_TRAIN_RUNS[self.fd]
        run_idx = list(range(num_runs))
        if isinstance(self.percent_fail_runs, float):
            split_idx = int(self.percent_fail_runs * num_runs)
            complement_idx = run_idx[split_idx:]
        elif isinstance(self.percent_fail_runs, Iterable):
            complement_idx = list(set(run_idx).difference(self.percent_fail_runs))
        else:
            complement_idx = []

        return complement_idx

    def check_compatibility(self, other: "AbstractReader") -> None:
        """
        Check if the other reader is compatible with this one.

        Compatibility of two readers ensures that training with both will probably
        succeed and produce valid results. Two readers are considered compatible, if
        they:

        * are both children of [AbstractReader]
        [rul_datasets.reader.abstract.AbstractReader]

        * have the same `window size`

        * have the same `max_rul`

        If any of these conditions is not met, the readers are considered
        misconfigured and a `ValueError` is thrown.

        Args:
            other: Another reader object.
        """
        if not isinstance(other, type(self)):
            raise ValueError(
                f"The other loader is not of class {type(self)} but {type(other)}."
            )
        if not self.window_size == other.window_size:
            raise ValueError(
                f"Window sizes are not compatible "
                f"{self.window_size} vs. {other.window_size}"
            )
        if not self.max_rul == other.max_rul:
            raise ValueError(
                f"Max RULs are not compatible " f"{self.max_rul} vs. {other.max_rul}"
            )
