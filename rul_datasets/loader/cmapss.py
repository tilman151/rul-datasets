import os
import warnings
from typing import Union, List, Tuple, Dict

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore

from rul_datasets.loader import scaling
from rul_datasets.loader.abstract import AbstractLoader, DATA_ROOT
from rul_datasets import utils


class CmapssLoader(AbstractLoader):
    _FMT: str = (
        "%d %d %.4f %.4f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
        "%.2f %.2f %.2f %.2f %.2f %.2f %.4f %.2f %d %d %.2f %.2f %.4f"
    )
    _TRAIN_PERCENTAGE: float = 0.8
    _WINDOW_SIZES: Dict[int, int] = {1: 30, 2: 20, 3: 30, 4: 15}
    _DEFAULT_CHANNELS: List[int] = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]
    _NUM_TRAIN_RUNS: Dict[int, int] = {1: 80, 2: 208, 3: 80, 4: 199}
    _CMAPSS_ROOT: str = os.path.join(DATA_ROOT, "CMAPSS")

    def __init__(
        self,
        fd: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        feature_select: List[int] = None,
        truncate_val: bool = False,
    ) -> None:
        super().__init__(
            fd, window_size, max_rul, percent_broken, percent_fail_runs, truncate_val
        )
        # Select features according to https://doi.org/10.1016/j.ress.2017.11.021
        if feature_select is None:
            feature_select = self._DEFAULT_CHANNELS
        self.feature_select = feature_select

    def _default_window_size(self, fd: int) -> int:
        return self._WINDOW_SIZES[fd]

    def prepare_data(self) -> None:
        # Check if training data was already split
        if not os.path.exists(self._get_feature_path("dev")):
            warnings.warn(
                f"Training data for FD{self.fd:03d} not "
                f"yet split into dev and val. Splitting now."
            )
            self._split_fd_train(self._get_feature_path("train"))
        if not os.path.exists(self._get_scaler_path()):
            self._prepare_scaler()

    def _prepare_scaler(self) -> None:
        dev_features = self._load_features(self._get_feature_path("dev"))
        dev_features, _ = self._split_time_steps_from_features(dev_features)
        scaler = scalers.MinMaxScaler(feature_range=(-1, 1))
        scaler = scaling.fit_scaler(dev_features, scaler)
        scaling.save_scaler(scaler, self._get_scaler_path())

    def _split_fd_train(self, train_path: str) -> None:
        train_data = np.loadtxt(train_path)

        # Split into runs
        _, samples_per_run = np.unique(train_data[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        train_data = np.split(train_data, split_idx, axis=0)

        split_idx = int(len(train_data) * self._TRAIN_PERCENTAGE)
        dev_data = np.concatenate(train_data[:split_idx])
        val_data = np.concatenate(train_data[split_idx:])

        data_root, train_file = os.path.split(train_path)
        dev_file = train_file.replace("train_", "dev_")
        dev_file = os.path.join(data_root, dev_file)
        np.savetxt(dev_file, dev_data, fmt=self._FMT)
        val_file = train_file.replace("train_", "val_")
        val_file = os.path.join(data_root, val_file)
        np.savetxt(val_file, val_data, fmt=self._FMT)

    def _get_scaler_path(self) -> str:
        return os.path.join(self._CMAPSS_ROOT, self._get_scaler_name())

    def _get_scaler_name(self) -> str:
        return f"FD{self.fd:03d}_scaler_{self.feature_select}.pkl"

    def _get_feature_path(self, split: str) -> str:
        return os.path.join(self._CMAPSS_ROOT, self._get_feature_name(split))

    def _get_feature_name(self, split: str) -> str:
        return f"{split}_FD{self.fd:03d}.txt"

    def _load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        file_path = self._get_feature_path(split)
        features = self._load_features(file_path)
        features, time_steps = self._split_time_steps_from_features(features)
        features = self._scale_features(features)

        if split in ["dev", "val"]:
            targets = self._generate_targets(time_steps)
            features, targets = self._window_data(features, targets)
        elif split == "test":
            targets = self._load_targets()
            features = self._crop_data(features)
        else:
            raise ValueError(f"Unknown split {split}.")

        return features, targets

    def _load_features(self, file_path: str) -> List[np.ndarray]:
        features = np.loadtxt(file_path)

        feature_idx = [0, 1] + [idx + 2 for idx in self.feature_select]
        features = features[:, feature_idx]

        # Split into runs
        _, samples_per_run = np.unique(features[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        features = np.split(features, split_idx, axis=0)

        return features

    def _scale_features(self, features: List[np.ndarray]) -> List[np.ndarray]:
        scaler_path = self._get_scaler_path()
        if not os.path.exists(scaler_path):
            raise RuntimeError(
                f"Scaler for FD{self.fd:03d} with features "
                f"{self.feature_select} does not exist. "
                f"Did you call prepare_data yet?"
            )
        scaler = scaling.load_scaler(scaler_path)
        features = scaling.scale_features(features, scaler)

        return features

    @staticmethod
    def _split_time_steps_from_features(
        features: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        time_steps = []
        for i, seq in enumerate(features):
            time_steps.append(seq[:, 1])
            seq = seq[:, 2:]
            features[i] = seq

        return features, time_steps

    def _generate_targets(self, time_steps: List[np.ndarray]) -> List[np.ndarray]:
        """Generate RUL targets from time steps."""
        return [np.minimum(self.max_rul, steps)[::-1].copy() for steps in time_steps]

    def _load_targets(self) -> List[np.ndarray]:
        """Load target file."""
        file_name = f"RUL_FD{self.fd:03d}.txt"
        file_path = os.path.join(self._CMAPSS_ROOT, file_name)
        targets = np.loadtxt(file_path)

        targets = np.minimum(self.max_rul, targets)
        targets = np.split(targets, len(targets))

        return targets

    def _window_data(
        self, features: List[np.ndarray], targets: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Window features with specified window size."""
        new_features = []
        new_targets = []
        for seq, target in zip(features, targets):
            windows = utils.extract_windows(seq, self.window_size)
            target = target[self.window_size - 1 :]
            new_features.append(windows)
            new_targets.append(target)

        return new_features, new_targets

    def _crop_data(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """Crop length of features to specified window size."""
        cropped_features = []
        for seq in features:
            if seq.shape[0] < self.window_size:
                pad = (self.window_size - seq.shape[0], seq.shape[1])
                seq = np.concatenate([np.zeros(pad), seq])
            else:
                seq = seq[-self.window_size :]
            cropped_features.append(np.expand_dims(seq, axis=0))

        return cropped_features