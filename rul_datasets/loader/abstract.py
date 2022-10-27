import os
from copy import deepcopy
from typing import Optional, Union, List, Dict, Any, Iterable, Tuple

import numpy as np
import torch

from rul_datasets.loader import truncating

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


class AbstractLoader:
    """The base class of all loaders."""

    fd: int
    window_size: int
    max_rul: int
    percent_broken: Optional[float]
    percent_fail_runs: Union[float, List[int], None]
    truncate_val: bool

    _NUM_TRAIN_RUNS: Dict[int, int]

    def __init__(
        self,
        fd: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        truncate_val: bool = False,
    ) -> None:
        self.fd = fd
        self.window_size = window_size or self._default_window_size(self.fd)
        self.max_rul = max_rul
        self.truncate_val = truncate_val
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "fd": self.fd,
            "window_size": self.window_size,
            "max_rul": self.max_rul,
            "percent_broken": self.percent_broken,
            "percent_fail_runs": self.percent_fail_runs,
            "truncate_val": self.truncate_val,
        }

    @property
    def fds(self) -> List[int]:
        raise NotImplementedError

    def prepare_data(self) -> None:
        raise NotImplementedError

    def _default_window_size(self, fd: int) -> int:
        raise NotImplementedError

    def _load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        raise NotImplementedError

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        features, targets = self._load_complete_split(split)
        if split == "dev" or (split == "val" and self.truncate_val):
            features, targets = truncating.truncate_runs(
                features, targets, self.percent_broken, self.percent_fail_runs
            )
        tensor_feats, tensor_targets = self._to_tensor(features, targets)

        return tensor_feats, tensor_targets

    def get_compatible(
        self,
        fd: Optional[int] = None,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Union[float, List[int], None] = None,
        truncate_val: Optional[bool] = None,
    ) -> "AbstractLoader":
        other = deepcopy(self)
        if percent_broken is not None:
            other.percent_broken = percent_broken
        if percent_fail_runs is not None:
            other.percent_fail_runs = percent_fail_runs
        if truncate_val is not None:
            other.truncate_val = truncate_val

        if fd is not None:
            other.fd = fd
            window_size = min(self.window_size, self._default_window_size(other.fd))
        else:
            window_size = self.window_size
        self.window_size = window_size
        other.window_size = window_size

        return other

    def get_complement(
        self,
        percent_broken: Optional[float] = None,
        truncate_val: Optional[bool] = None,
    ) -> "AbstractLoader":
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

    def check_compatibility(self, other: "AbstractLoader") -> None:
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

    def _to_tensor(
        self, features: List[np.ndarray], targets: List[np.ndarray]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        dtype = torch.float32
        tensor_feats = [torch.tensor(f, dtype=dtype).permute(0, 2, 1) for f in features]
        tensor_targets = [torch.tensor(t, dtype=dtype) for t in targets]

        return tensor_feats, tensor_targets
