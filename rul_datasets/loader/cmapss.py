import os
import warnings
from typing import Union, List, Tuple

import numpy as np
import torch
from sklearn import preprocessing as scalers

from rul_datasets.loader.abstract import AbstractLoader, DATA_ROOT


class CmapssLoader(AbstractLoader):
    _FMT = (
        "%d %d %.4f %.4f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
        "%.2f %.2f %.2f %.2f %.2f %.2f %.4f %.2f %d %d %.2f %.2f %.4f"
    )
    _TRAIN_PERCENTAGE = 0.8
    _WINDOW_SIZES = {1: 30, 2: 20, 3: 30, 4: 15}
    _DEFAULT_CHANNELS = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]
    _NUM_TRAIN_RUNS = {1: 80, 2: 208, 3: 80, 4: 199}
    _CMAPSS_ROOT = os.path.join(DATA_ROOT, "CMAPSS")

    def __init__(
        self,
        fd: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        feature_select: List[int] = None,
        truncate_val: bool = False,
    ):
        # Select features according to https://doi.org/10.1016/j.ress.2017.11.021
        if feature_select is None:
            feature_select = self._DEFAULT_CHANNELS

        self.fd = fd
        self.window_size = window_size or self._default_window_size(self.fd)
        self.max_rul = max_rul
        self.feature_select = feature_select
        self.truncate_val = truncate_val
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs

    def _default_window_size(self, fd: int) -> int:
        return self._WINDOW_SIZES[fd]

    def prepare_data(self):
        # Check if training data was already split
        dev_path = self._file_path("dev")
        if not os.path.exists(dev_path):
            warnings.warn(
                f"Training data for FD{self.fd:03d} not "
                f"yet split into dev and val. Splitting now."
            )
            self._split_fd_train(self._file_path("train"))

    def _split_fd_train(self, train_path: str):
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

    def _file_path(self, split: str) -> str:
        return os.path.join(self._CMAPSS_ROOT, self._file_name(split))

    def _file_name(self, split: str) -> str:
        return f"{split}_FD{self.fd:03d}.txt"

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        file_path = self._file_path(split)

        features = self._load_features(file_path)
        features = self._normalize(features)
        features, time_steps = self._remove_time_steps_from_features(features)

        if split in ["dev", "val"]:
            features, targets = self._process_dev_or_val_split(
                split, features, time_steps
            )
        elif split == "test":
            features, targets = self._process_test_split(features)
        else:
            raise ValueError(f"Unknown split {split}.")

        tensor_feats, tensor_targets = self._to_tensor(features, targets)

        return tensor_feats, tensor_targets

    def _process_dev_or_val_split(self, split, features, time_steps):
        # Build targets from time steps on training
        targets = self._generate_targets(time_steps)
        # Window data to get uniform sequence lengths
        features, targets = self._window_data(features, targets)
        if split == "dev":
            features, targets = self._truncate_runs(
                features, targets, self.percent_broken, self.percent_fail_runs
            )
        elif split == "val" and self.truncate_val:
            features, targets = self._truncate_runs(
                features, targets, self.percent_broken
            )

        return features, targets

    def _process_test_split(self, features):
        # Load targets from file on test
        targets = self._load_targets()
        # Crop data to get uniform sequence lengths
        features = self._crop_data(features)

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

    def _normalize(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize features with sklearn transform."""
        # Fit scaler on corresponding training split
        train_file = self._file_path("dev")
        train_features = self._load_features(train_file)
        full_features = np.concatenate(train_features, axis=0)
        scaler = scalers.MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(full_features[:, 2:])

        # Normalize features
        for i, run in enumerate(features):
            features[i][:, 2:] = scaler.transform(run[:, 2:])

        return features

    @staticmethod
    def _remove_time_steps_from_features(
        features: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract and return time steps from feature array."""
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
            windows = self.extract_windows(seq, self.window_size)
            target = target[self.window_size - 1 :]
            new_features.append(windows)
            new_targets.append(target)

        return new_features, new_targets

    def extract_windows(self, seq, window_size):
        num_frames = seq.shape[0] - window_size + 1
        window_idx = np.expand_dims(np.arange(window_size), 0)
        window_idx = window_idx + np.expand_dims(np.arange(num_frames), 0).T
        windows = seq[window_idx]

        return windows

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
