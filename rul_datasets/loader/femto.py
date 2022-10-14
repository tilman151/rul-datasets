import os
import pickle
import re
from typing import List, Tuple, Union

import numpy as np
import sklearn.preprocessing as scalers  # type: ignore
import torch
from tqdm import tqdm  # type: ignore

from rul_datasets import utils
from rul_datasets.loader.abstract import AbstractLoader, DATA_ROOT
from rul_datasets.loader import scaling


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

    def prepare_data(self):
        self._preparator.prepare_split("dev")
        self._preparator.prepare_split("test")

    def load_split(self, split: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        features, targets = self._load_runs(split)
        if split == "dev":
            features, targets = self._truncate_runs(
                features, targets, self.percent_broken, self.percent_fail_runs
            )
        features = scaling.scale_features(features, self._preparator.load_scaler())
        features, targets = self._to_tensor(features, targets)

        return features, targets

    def _load_runs(self, split: str):
        features, targets = self._preparator.load_runs(split)
        features = [f[:, -self.window_size :, :] for f in features]

        return features, targets

    def _default_window_size(self, fd: int) -> int:
        return FemtoPreparator.DEFAULT_WINDOW_SIZE


class FemtoPreparator:
    DEFAULT_WINDOW_SIZE = 2560
    SPLIT_FOLDERS = {"dev": "Learning_set", "test": "Test_set", "val": None}

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
            scaler = scaling.fit_scaler(features)
            scaling.save_scaler(scaler, self._get_scaler_path())

    def load_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self._validate_split(split)
        with open(self._get_run_file_path(split), mode="rb") as f:
            features, targets = pickle.load(f)

        return features, targets

    def _load_raw_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        file_paths = self._get_csv_file_paths(split)
        features = self._load_raw_features(file_paths)
        targets = utils.get_targets_from_file_paths(
            file_paths, self._timestep_from_file_path
        )

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
            feature_files = utils.get_csv_files_in_path(
                run_path, lambda f: f.startswith("acc")
            )
            file_paths.append(feature_files)

        return file_paths

    def _validate_split(self, split: str) -> None:
        if split not in self.SPLIT_FOLDERS:
            raise ValueError(f"Unsupported split '{split}' supplied.")
        if split == "val":
            raise ValueError("FEMTO does not define a validation set.")

    def _get_run_folders(self, split_path):
        folder_pattern = self._get_run_folder_pattern()
        all_folders = os.listdir(split_path)
        run_folders = [f for f in all_folders if folder_pattern.match(f) is not None]

        return run_folders

    def _get_run_folder_pattern(self) -> re.Pattern:
        return re.compile(rf"Bearing{self.fd}_\d")

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

    @staticmethod
    def _timestep_from_file_path(file_path: str) -> int:
        file_name = os.path.basename(file_path)
        time_step = int(file_name[4:9])

        return time_step

    def load_scaler(self) -> scalers.StandardScaler:
        return scaling.load_scaler(self._get_scaler_path())

    def _save_efficient(
        self, split: str, features: List[np.ndarray], targets: List[np.ndarray]
    ):
        with open(self._get_run_file_path(split), mode="wb") as f:
            pickle.dump((features, targets), f)

    def _get_scaler_path(self):
        return os.path.join(self._get_split_folder("dev"), f"scaler_{self.fd}.pkl")

    def _get_run_file_path(self, split: str) -> str:
        return os.path.join(self._get_split_folder(split), f"runs_{self.fd}.pkl")

    def _get_split_folder(self, split: str) -> str:
        return os.path.join(self._data_root, self.SPLIT_FOLDERS[split])
