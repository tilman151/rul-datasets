from typing import Tuple, List, Optional, Union

import numpy as np
from sklearn import preprocessing

from rul_datasets import utils
from rul_datasets.reader import AbstractReader, scaling


class DummyReader(AbstractReader):
    _FDS = [1, 2]
    _DEFAULT_WINDOW_SIZE = 10
    _NOISE_FACTOR = {1: 0.01, 2: 0.02}
    _OFFSET = {1: 0.5, 2: 0.75}
    _SPLIT_SEED = {"dev": 42, "val": 1337, "test": 101}

    def __init__(
        self,
        fd: int,
        window_size: Optional[int] = None,
        max_rul: Optional[int] = 50,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Optional[Union[float, List[int]]] = None,
        truncate_val: bool = False,
    ):
        super(DummyReader, self).__init__(
            fd,
            window_size,
            max_rul,
            percent_broken,
            percent_fail_runs,
            truncate_val,
        )

        features, _ = self._generate_split("dev")
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaling.fit_scaler(features, scaler)

    @property
    def fds(self) -> List[int]:
        return self._FDS

    def prepare_data(self) -> None:
        pass

    def default_window_size(self, fd: int) -> int:
        return self._DEFAULT_WINDOW_SIZE

    def load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, targets = self._generate_split(split)
        features = scaling.scale_features(features, self.scaler)

        return features, targets

    def _generate_split(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features = []
        targets = []
        rng = np.random.default_rng(self._SPLIT_SEED[split])
        for i in range(10):
            t = self._generate_targets(rng)
            f = self._generate_features(rng, t)
            f = utils.extract_windows(f, self.window_size)
            t = t[: -(self.window_size - 1)]
            features.append(f)
            targets.append(t)
        if split == "test":
            features, targets = self._truncate_test_split(rng, features, targets)

        return features, targets

    def _generate_targets(self, rng):
        length = rng.integers(90, 110)
        t = np.clip(np.arange(length, 1, -1), a_min=0, a_max=self.max_rul)
        t = t.astype(np.float)

        return t[:, None]

    def _generate_features(self, rng, targets):
        steady = -0.05 * targets + self._OFFSET[self.fd] + rng.normal() * 0.01
        noise = rng.normal(size=targets.shape) * self._NOISE_FACTOR[self.fd]
        f = np.exp(steady) + noise

        return f

    def _truncate_test_split(self, rng, features, targets):
        """Extract a single window from a random position of the time series."""
        for i in range(len(features)):
            run_len = len(features[i])
            cutoff = rng.integers(run_len // 2, run_len - 1)
            features[i] = features[i][None, cutoff]
            targets[i] = targets[i][None, cutoff]

        return features, targets
