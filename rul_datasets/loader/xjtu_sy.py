import os.path
import pickle
from typing import Tuple, List, Union, Dict, Optional

import numpy as np
from tqdm import tqdm  # type: ignore
from sklearn import preprocessing as scalers  # type: ignore

from rul_datasets import utils
from rul_datasets.loader import AbstractLoader, DATA_ROOT
from rul_datasets.loader import scaling


class XjtuSyLoader(AbstractLoader):
    _XJTU_SY_ROOT: str = os.path.join(DATA_ROOT, "XJTU-SY")
    _NUM_TRAIN_RUNS: Dict[int, int] = {1: 5, 2: 5, 3: 5}

    def __init__(
        self,
        fd: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        truncate_val: bool = False,
        run_split_dist: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        super().__init__(
            fd, window_size, max_rul, percent_broken, percent_fail_runs, truncate_val
        )
        self._preparator = XjtuSyPreparator(self.fd, self._XJTU_SY_ROOT, run_split_dist)

    @property
    def fds(self) -> List[int]:
        return list(self._NUM_TRAIN_RUNS)

    def prepare_data(self) -> None:
        self._preparator.prepare_split("dev")

    def _load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, targets = self._preparator.load_runs(split)
        features = [f[:, -self.window_size :, :] for f in features]  # crop to window
        features = scaling.scale_features(features, self._preparator.load_scaler())

        return features, targets

    def _default_window_size(self, fd: int) -> int:
        return XjtuSyPreparator.DEFAULT_WINDOW_SIZE


class XjtuSyPreparator:
    DEFAULT_WINDOW_SIZE: int = 32768
    FD_FOLDERS: Dict[int, str] = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
    _DEFAULT_RUN_SPLIT_DIST: Dict[str, List[int]] = {
        "dev": [0, 1],
        "val": [2],
        "test": [3, 4],
    }

    def __init__(
        self,
        fd: int,
        data_root: str,
        run_split_dist: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.fd = fd
        self.data_root = data_root
        self.run_split_dist = run_split_dist or self._DEFAULT_RUN_SPLIT_DIST

    def prepare_split(self, split: str) -> None:
        self._validate_split(split)
        run_file_path = self._get_run_file_path()
        if not os.path.exists(run_file_path):
            features, targets = self._load_raw_runs(split)
            features, targets = self._sort_runs(features, targets)
            self._save_efficient(features, targets)
        if not os.path.exists(self._get_scaler_path()):
            features, _ = self.load_runs(split)
            scaler = scaling.fit_scaler(features)
            scaling.save_scaler(scaler, self._get_scaler_path())

    def load_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self._validate_split(split)
        with open(self._get_run_file_path(), mode="rb") as f:
            features, targets = pickle.load(f)
        features = [features[i] for i in self.run_split_dist[split]]
        targets = [targets[i] for i in self.run_split_dist[split]]

        return features, targets

    def load_scaler(self) -> scalers.StandardScaler:
        return scaling.load_scaler(self._get_scaler_path())

    def _load_raw_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        file_paths = self._get_csv_file_paths(split)
        runs = self._load_raw_features(file_paths)
        targets = utils.get_targets_from_file_paths(
            file_paths, self._timestep_from_file_path
        )

        return runs, targets

    def _sort_runs(
        self, features: List[np.ndarray], targets: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        sort_idx = [np.argsort(t)[::-1] for t in targets]
        features = [f[i] for f, i in zip(features, sort_idx)]
        targets = [t[i] for t, i in zip(targets, sort_idx)]

        return features, targets

    def _load_raw_features(self, file_paths: List[List[str]]) -> List[np.ndarray]:
        runs = []
        for run_files in tqdm(file_paths, desc="Runs"):
            run_features = np.empty((len(run_files), self.DEFAULT_WINDOW_SIZE, 2))
            for i, file_path in enumerate(tqdm(run_files, desc="Files")):
                run_features[i] = self._load_feature_file(file_path)
            runs.append(run_features)
        return runs

    def _load_feature_file(self, file_path: str) -> np.ndarray:
        return np.loadtxt(file_path, skiprows=1, delimiter=",")

    def _get_csv_file_paths(self, split: str) -> List[List[str]]:
        fd_folder_path = self._get_fd_folder_path()
        file_paths = []
        run_folders = self._get_run_folders(fd_folder_path)
        for run_folder in run_folders:
            run_path = os.path.join(fd_folder_path, run_folder)
            feature_files = utils.get_files_in_path(run_path)
            file_paths.append(feature_files)

        return file_paths

    def _get_run_folders(self, split_path: str) -> List[str]:
        all_folders = sorted(os.listdir(split_path))
        run_folders = [
            f for f in all_folders if os.path.isdir(os.path.join(split_path, f))
        ]

        return run_folders

    @staticmethod
    def _timestep_from_file_path(file_path: str) -> int:
        file_name = os.path.basename(file_path)
        time_step = int(file_name.replace(".csv", ""))

        return time_step

    def _save_efficient(
        self, features: List[np.ndarray], targets: List[np.ndarray]
    ) -> None:
        with open(self._get_run_file_path(), mode="wb") as f:
            pickle.dump((features, targets), f)

    def _validate_split(self, split: str) -> None:
        if split not in ["dev", "val", "test"]:
            raise ValueError(
                "XJTU-SY provides a dev, val and test split, "
                f"but you provided '{split}'."
            )

    def _get_scaler_path(self) -> str:
        return os.path.join(self._get_fd_folder_path(), "scaler.pkl")

    def _get_run_file_path(self) -> str:
        return os.path.join(self._get_fd_folder_path(), "runs.pkl")

    def _get_fd_folder_path(self) -> str:
        return os.path.join(self.data_root, self.FD_FOLDERS[self.fd])
