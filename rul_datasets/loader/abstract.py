import os
from copy import deepcopy
from typing import Optional, Union, List, Dict, Any, Iterable, Tuple

import numpy as np
import torch

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


class AbstractLoader:
    fd: int
    window_size: int
    max_rul: int
    percent_broken: Optional[float]
    percent_fail_runs: Union[float, List[int], None]
    truncate_val: bool

    _NUM_TRAIN_RUNS: Dict[int, int]

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

    def _default_window_size(self, fd: int) -> int:
        raise NotImplementedError

    def prepare_data(self) -> None:
        raise NotImplementedError

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        raise NotImplementedError

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

    def _truncate_runs(
        self,
        features: List[np.ndarray],
        targets: List[np.ndarray],
        percent_broken: float = None,
        included_runs: Union[float, Iterable[int]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Truncate the number of runs
        if included_runs is not None:
            features, targets = self._truncate_included(
                features, targets, included_runs
            )

        # Truncate the number of samples per run, starting at failure
        if percent_broken is not None and percent_broken < 1:
            features, targets = self._truncate_broken(features, targets, percent_broken)

        return features, targets

    def _truncate_included(
        self,
        features: List[np.ndarray],
        targets: List[np.ndarray],
        included_runs: Union[float, Iterable[int]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if isinstance(included_runs, float):
            features, targets = self._truncate_included_by_percentage(
                features, targets, included_runs
            )
        elif isinstance(included_runs, Iterable):
            features, targets = self._truncate_included_by_index(
                features, targets, included_runs
            )

        return features, targets

    def _truncate_included_by_percentage(
        self,
        features: List[np.ndarray],
        targets: List[np.ndarray],
        percent_included: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        num_runs = int(percent_included * len(features))
        features = features[:num_runs]
        targets = targets[:num_runs]

        return features, targets

    def _truncate_included_by_index(
        self,
        features: List[np.ndarray],
        targets: List[np.ndarray],
        included_idx: Iterable[int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features = [features[i] for i in included_idx]
        targets = [targets[i] for i in included_idx]

        return features, targets

    def _truncate_broken(
        self,
        features: List[np.ndarray],
        targets: List[np.ndarray],
        percent_broken: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        for i, run in enumerate(features):
            num_cycles = int(percent_broken * len(run))
            features[i] = run[:num_cycles]
            targets[i] = targets[i][:num_cycles]

        return features, targets

    def _to_tensor(
        self, features: List[np.ndarray], targets: List[np.ndarray]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        dtype = torch.float32
        tensor_feats = [torch.tensor(f, dtype=dtype).permute(0, 2, 1) for f in features]
        tensor_targets = [torch.tensor(t, dtype=dtype) for t in targets]

        return tensor_feats, tensor_targets
