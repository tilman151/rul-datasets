import os
import pickle
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import sklearn.preprocessing as scalers  # type: ignore
import torch
from tqdm import tqdm  # type: ignore

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class AbstractLoader:
    fd: int
    window_size: int
    max_rul: int
    percent_broken: Optional[float]
    percent_fail_runs: Union[float, List[int], None]
    truncate_val: bool

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
        self.window_size = window_size or self._WINDOW_SIZES[self.fd]
        self.max_rul = max_rul
        self.feature_select = feature_select
        self.truncate_val = truncate_val
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs

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


class FemtoLoader(AbstractLoader):
    _FEMTO_ROOT = os.path.join(DATA_ROOT, "FEMTOBearingDataSet")

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
        self.fd = fd
        self.window_size = window_size or FemtoPreparator.DEFAULT_WINDOW_SIZE
        self.max_rul = max_rul
        self.feature_select = feature_select
        self.truncate_val = truncate_val
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs

        self._preparator = FemtoPreparator(self.fd, self._FEMTO_ROOT)

    def prepare_data(self):
        self._preparator.prepare_split("dev")
        self._preparator.prepare_split("test")

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        features, targets = self._load_runs(split)
        if split == "dev":
            features, targets = self._truncate_runs(
                features, targets, self.percent_broken, self.percent_fail_runs
            )
        features = self._scale_features(features)
        features, targets = self._to_tensor(features, targets)

        return features, targets

    def _load_runs(self, split: str):
        features, targets = self._preparator.load_runs(split)
        features = [f[:, -self.window_size :, :] for f in features]

        return features, targets

    def _scale_features(self, runs: List[np.ndarray]) -> List[np.ndarray]:
        scaler = self._preparator.load_scaler()
        for i, run in enumerate(runs):
            run = run.reshape(-1, 2)
            run = scaler.transform(run)
            runs[i] = run.reshape(-1, self.window_size, 2)

        return runs


class FemtoPreparator:
    DEFAULT_WINDOW_SIZE = 2560
    SPLIT_FOLDERS = {"dev": "Learning_set", "test": "Full_Test_Set"}

    def __init__(self, fd, data_root):
        self.fd = fd
        self._data_root = data_root

    def prepare_split(self, split: str):
        if not os.path.exists(self._get_run_file_path(split)):
            print(f"Prepare FEMTO {split} data of condition {self.fd}...")
            features, targets = self._load_raw_runs(split)
            self._save_efficient(split, features, targets)
        if split == "dev" and not os.path.exists(self._get_scaler_path()):
            features, _ = self.load_runs(split)
            scaler = self._fit_scaler(features)
            self._save_scaler(scaler)

    def load_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if split == "val":
            raise ValueError("FEMTO does not define a validation set.")

        save_path = self._get_run_file_path(split)
        with open(save_path, mode="rb") as f:
            features, targets = pickle.load(f)

        return features, targets

    def _load_raw_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        file_paths = self._get_csv_file_paths(split)
        features = self._load_raw_features(file_paths)
        targets = self._targets_from_file_paths(file_paths)

        return features, targets

    def _load_raw_features(self, file_paths: List[List[str]]) -> List[np.ndarray]:
        runs = []
        for run_files in tqdm(file_paths, desc="Runs"):
            run_features = np.empty((len(run_files), self.DEFAULT_WINDOW_SIZE, 2))
            for i, file_path in enumerate(tqdm(run_files, desc="Files")):
                run_features[i] = self._load_feature_file(file_path)
            runs.append(run_features)

        return runs

    def _get_csv_file_paths(self, split: str) -> List[List[str]]:
        split_path = self._get_split_folder(split)
        run_folders = self._get_run_folders(split_path)
        file_paths = []
        for run_folder in run_folders:
            run_path = os.path.join(split_path, run_folder)
            feature_files = self._get_csv_files_in_path(run_path)
            file_paths.append(feature_files)

        return file_paths

    def _get_split_folder(self, split: str) -> str:
        return os.path.join(self._data_root, self.SPLIT_FOLDERS[split])

    def _get_run_folders(self, split_path):
        folder_pattern = self._get_run_folder_pattern()
        all_folders = os.listdir(split_path)
        run_folders = [f for f in all_folders if folder_pattern.match(f) is not None]

        return run_folders

    def _get_run_folder_pattern(self) -> re.Pattern:
        return re.compile(fr"Bearing{self.fd}_\d")

    def _get_csv_files_in_path(self, run_path):
        feature_files = [f for f in os.listdir(run_path) if f.startswith("acc")]
        feature_files = sorted(os.path.join(run_path, f) for f in feature_files)

        return feature_files

    def _load_feature_file(self, file_path: str) -> np.ndarray:
        try:
            features = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            self._replace_delimiters(file_path)
            features = np.loadtxt(file_path, delimiter=",")
        features = features[:, [4, 5]]

        return features

    def _replace_delimiters(self, file_path: str):
        with open(file_path, mode="r+t") as f:
            content = f.read()
            f.seek(0)
            content = content.replace(";", ",")
            f.write(content)
            f.truncate()

    def _targets_from_file_paths(self, file_paths: List[List[str]]) -> List[np.ndarray]:
        targets = []
        for run_files in file_paths:
            run_targets = np.empty(len(run_files))
            for i, file_path in enumerate(run_files):
                run_targets[i] = self._timestep_from_file_path(file_path)
            run_targets = run_targets[::-1].copy()
            targets.append(run_targets)

        return targets

    def _timestep_from_file_path(self, file_path: str) -> int:
        file_name = os.path.basename(file_path)
        time_step = int(file_name[4:9])

        return time_step

    def _fit_scaler(self, features):
        scaler = scalers.StandardScaler()
        for run in features:
            run = run.reshape(-1, run.shape[-1])
            scaler.partial_fit(run)

        return scaler

    def _save_scaler(self, scaler):
        save_path = self._get_scaler_path()
        with open(save_path, mode="wb") as f:
            pickle.dump(scaler, f)

    def load_scaler(self) -> scalers.StandardScaler:
        save_path = self._get_scaler_path()
        with open(save_path, mode="rb") as f:
            scaler = pickle.load(f)

        return scaler

    def _get_scaler_path(self):
        return os.path.join(self._get_split_folder("dev"), f"scaler_{self.fd}.pkl")

    def _save_efficient(
        self, split: str, features: List[np.ndarray], targets: List[np.ndarray]
    ):
        with open(self._get_run_file_path(split), mode="wb") as f:
            pickle.dump((features, targets), f)

    def _get_run_file_path(self, split: str) -> str:
        split_folder = self._get_split_folder(split)
        run_file_path = os.path.join(split_folder, f"runs_{self.fd}.pkl")

        return run_file_path
