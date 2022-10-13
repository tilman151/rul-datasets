import os
import pickle
import re
from typing import List, Tuple, Union

import numpy as np
import sklearn.preprocessing as scalers  # type: ignore
import torch
from tqdm import tqdm  # type: ignore

from rul_datasets.loader.abstract import AbstractLoader, DATA_ROOT


class FemtoLoader(AbstractLoader):
    _FEMTO_ROOT = os.path.join(DATA_ROOT, "FEMTOBearingDataSet")
    _NUM_TRAIN_RUNS = {1: 2, 2: 2, 3: 2}

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
        self.window_size = window_size or self._default_window_size(self.fd)
        self.max_rul = max_rul
        self.feature_select = feature_select
        self.truncate_val = truncate_val
        self.percent_broken = percent_broken
        self.percent_fail_runs = percent_fail_runs

        self._preparator = FemtoPreparator(self.fd, self._FEMTO_ROOT)

    def _default_window_size(self, fd: int) -> int:
        return FemtoPreparator.DEFAULT_WINDOW_SIZE

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
    SPLIT_FOLDERS = {"dev": "Learning_set", "test": "Test_set"}

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
        return re.compile(rf"Bearing{self.fd}_\d")

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
