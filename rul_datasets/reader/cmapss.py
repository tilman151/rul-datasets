"""The NASA CMAPSS Turbofan Degradation dataset is a collection of simulated
degradation experiments on jet engines. It contains four sub-datasets named FD1, FD2,
FD3 and FD4 which differ in operation conditions and possible failure types."""

import os
import tempfile
import warnings
import zipfile
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore

from rul_datasets.reader import scaling
from rul_datasets.reader.data_root import get_data_root
from rul_datasets.reader.abstract import AbstractReader
from rul_datasets import utils


CMAPSS_URL = "https://kr0k0tsch.de/rul-datasets/CMAPSSData.zip"


class CmapssReader(AbstractReader):
    """
    This reader represents the NASA CMAPSS Turbofan Degradation dataset. Each of its
    four sub-datasets contain a training and a test split. Upon first usage,
    the training split will be further divided into a development and a validation
    split. 20% of the original training split are reserved for validation.

    The features are provided as sliding windows over each time series in the
    dataset. The label of a window is the label of its last time step. The RUL labels
    are capped by a maximum value. The original data contains 24 channels per time
    step. Following the literature, we omit the constant channels and operation
    condition channels by default. Therefore, the default channel indices are 4, 5,
    6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22 and 23.

    The features are min-max scaled between -1 and 1. The scaler is fitted on the
    development data only.

    Examples:
        Default channels
        >>> import rul_datasets
        >>> fd1 = rul_datasets.reader.CmapssReader(fd=1, window_size=30)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> features[0].shape
        (163, 30, 14)

        Custom channels
        >>> import rul_datasets
        >>> fd1 = rul_datasets.reader.CmapssReader(fd=1, feature_select=[1, 2, 3])
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> features[0].shape
        (163, 30, 3)
    """

    _FMT: str = (
        "%d %d %.4f %.4f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
        "%.2f %.2f %.2f %.2f %.2f %.2f %.4f %.2f %d %d %.2f %.2f %.4f"
    )
    _TRAIN_PERCENTAGE: float = 0.8
    _WINDOW_SIZES: Dict[int, int] = {1: 30, 2: 20, 3: 30, 4: 15}
    _DEFAULT_CHANNELS: List[int] = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]
    _NUM_TRAIN_RUNS: Dict[int, int] = {1: 80, 2: 208, 3: 80, 4: 199}
    _CONDITION_BOUNDARIES: List[Tuple[float, float]] = [
        (-0.009, 0.009),  # Different from paper to include FD001 and FD003
        (9.998, 10.008),
        (19.998, 20.008),
        (24.998, 25.008),
        (34.998, 35.008),
        (41.998, 42.008),
    ]
    _CONDITION_COLUMN: int = 0
    _CMAPSS_ROOT: str = os.path.join(get_data_root(), "CMAPSS")

    def __init__(
        self,
        fd: int,
        window_size: Optional[int] = None,
        max_rul: Optional[int] = 125,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Optional[Union[float, List[int]]] = None,
        feature_select: List[int] = None,
        truncate_val: bool = False,
        operation_condition_aware_scaling: bool = False,
    ) -> None:
        """
        Create a new CMAPSS reader for one of the sub-datasets. The maximum RUL value
        is set to 125 by default. The 14 feature channels selected by default can be
        overridden by passing a list of channel indices to `feature_select`. The
        default window size is defined per sub-dataset as the minimum time series
        length in the test set.

        The data can be scaled separately for each operation condition, as done by
        Ragab et al. This only affects FD002 and FD004 due to them having multiple
        operation conditions.

        For more information about using readers refer to the [reader]
        [rul_datasets.reader] module page.

        Args:
            fd: Index of the selected sub-dataset
            window_size: Size of the sliding window. Default defined per sub-dataset.
            max_rul: Maximum RUL value of targets.
            percent_broken: The maximum relative degradation per time series.
            percent_fail_runs: The percentage or index list of available time series.
            feature_select: The index list of selected feature channels.
            truncate_val: Truncate the validation data with `percent_broken`, too.
            operation_condition_aware_scaling: Scale data separatly for each
                                               operation condition.
        """
        super().__init__(
            fd, window_size, max_rul, percent_broken, percent_fail_runs, truncate_val
        )
        # Select features according to https://doi.org/10.1016/j.ress.2017.11.021
        if feature_select is None:
            feature_select = self._DEFAULT_CHANNELS
        self.feature_select = feature_select
        self.operation_condition_aware_scaling = operation_condition_aware_scaling

    @property
    def fds(self) -> List[int]:
        """Indices of available sub-datasets."""
        return list(self._WINDOW_SIZES)

    def default_window_size(self, fd: int) -> int:
        return self._WINDOW_SIZES[fd]

    def prepare_data(self) -> None:
        """
        Prepare the CMAPSS dataset. This function needs to be called before using the
        dataset for the first time.

        The dataset is downloaded from a custom mirror and extracted into the data
        root directory. The training data is then split into development and
        validation set. Afterwards, a scaler is fit on the development features.
        Previously completed steps are skipped.
        """
        if not os.path.exists(self._CMAPSS_ROOT):
            _download_cmapss(get_data_root())
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
        dev_features, ops_cond = self._load_features(self._get_feature_path("dev"))
        dev_features, _ = self._split_time_steps_from_features(dev_features)
        scaler = self._fit_scaler(dev_features, ops_cond)
        scaling.save_scaler(scaler, self._get_scaler_path())

    def _fit_scaler(self, features, operation_conditions):
        scaler = scalers.MinMaxScaler(feature_range=(-1, 1))
        if self.operation_condition_aware_scaling:
            scaler = scaling.OperationConditionAwareScaler(
                scaler, self._CONDITION_BOUNDARIES
            )
            operation_conditions = [
                c[:, self._CONDITION_COLUMN] for c in operation_conditions
            ]
            scaler = scaling.fit_scaler(features, scaler, operation_conditions)
        else:
            scaler = scaling.fit_scaler(features, scaler)

        return scaler

    def _split_fd_train(self, train_path: str) -> None:
        train_data = np.loadtxt(train_path)

        # Split into runs
        _, samples_per_run = np.unique(train_data[:, 0], return_counts=True)
        split_indices = np.cumsum(samples_per_run)[:-1]
        split_train_data = np.split(train_data, split_indices, axis=0)

        split_idx = int(len(split_train_data) * self._TRAIN_PERCENTAGE)
        dev_data = np.concatenate(split_train_data[:split_idx])
        val_data = np.concatenate(split_train_data[split_idx:])

        data_root, train_file = os.path.split(train_path)
        dev_file = train_file.replace("train_", "dev_")
        dev_file = os.path.join(data_root, dev_file)
        np.savetxt(dev_file, dev_data, fmt=self._FMT)  # type: ignore
        val_file = train_file.replace("train_", "val_")
        val_file = os.path.join(data_root, val_file)
        np.savetxt(val_file, val_data, fmt=self._FMT)  # type: ignore

    def _get_scaler_path(self) -> str:
        return os.path.join(self._CMAPSS_ROOT, self._get_scaler_name())

    def _get_scaler_name(self) -> str:
        ops_aware = "_ops_aware" if self.operation_condition_aware_scaling else ""
        name = f"FD{self.fd:03d}_scaler_{self.feature_select}{ops_aware}.pkl"

        return name

    def _get_feature_path(self, split: str) -> str:
        return os.path.join(self._CMAPSS_ROOT, self._get_feature_name(split))

    def _get_feature_name(self, split: str) -> str:
        return f"{split}_FD{self.fd:03d}.txt"

    def load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        file_path = self._get_feature_path(split)
        features, operation_conditions = self._load_features(file_path)
        features, time_steps = self._split_time_steps_from_features(features)
        features = self._scale_features(features, operation_conditions)

        if split in ["dev", "val"]:
            targets = self._generate_targets(time_steps)
            features, targets = self._window_data(features, targets)
        elif split == "test":
            targets = self._load_targets()
            features = self._crop_data(features)
        else:
            raise ValueError(f"Unknown split {split}.")

        return features, targets

    def _load_features(
        self, file_path: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        raw_features = np.loadtxt(file_path)

        feature_idx = [0, 1] + [idx + 2 for idx in self.feature_select]
        operation_conditions = raw_features[:, [2, 3, 4]]
        raw_features = raw_features[:, feature_idx]

        # Split into runs
        _, samples_per_run = np.unique(raw_features[:, 0], return_counts=True)
        split_idx = np.cumsum(samples_per_run)[:-1]
        features = np.split(raw_features, split_idx, axis=0)
        cond_per_run = np.split(operation_conditions, split_idx, axis=0)

        return features, cond_per_run

    def _scale_features(
        self, features: List[np.ndarray], operation_conditions: List[np.ndarray]
    ) -> List[np.ndarray]:
        scaler_path = self._get_scaler_path()
        if not os.path.exists(scaler_path):
            raise RuntimeError(
                f"Scaler for FD{self.fd:03d} with features "
                f"{self.feature_select} does not exist. "
                f"Did you call prepare_data yet?"
            )
        scaler = scaling.load_scaler(scaler_path)
        if self.operation_condition_aware_scaling:
            operation_conditions = [
                c[:, self._CONDITION_COLUMN] for c in operation_conditions
            ]
            features = scaling.scale_features(features, scaler, operation_conditions)
        else:
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
        max_rul = self.max_rul or np.inf  # no capping if max_rul is None
        targets = [np.minimum(max_rul, steps)[::-1].copy() for steps in time_steps]

        return targets

    def _load_targets(self) -> List[np.ndarray]:
        """Load target file."""
        file_name = f"RUL_FD{self.fd:03d}.txt"
        file_path = os.path.join(self._CMAPSS_ROOT, file_name)
        raw_targets = np.loadtxt(file_path)

        max_rul = self.max_rul or np.inf
        raw_targets = np.minimum(max_rul, raw_targets)
        targets = np.split(raw_targets, len(raw_targets))

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


def _download_cmapss(data_root: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_path:
        print("Download CMAPSS dataset")
        download_path = os.path.join(tmp_path, "CMAPSSData.zip")
        utils.download_file(CMAPSS_URL, download_path)
        print("Extract CMAPSS dataset")
        with zipfile.ZipFile(download_path, mode="r") as f:
            f.extractall(data_root)
