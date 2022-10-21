import os.path
from typing import Tuple, List, Union, Dict, Optional

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore
from tqdm import tqdm  # type: ignore

from rul_datasets import utils
from rul_datasets.loader import AbstractLoader, DATA_ROOT, saving, scaling


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
    _FD_FOLDERS: Dict[int, str] = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
    _DEFAULT_RUN_SPLIT_DIST: Dict[str, List[int]] = {
        "dev": [1, 2],
        "val": [3],
        "test": [4, 5],
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
        run_file_path = self._get_run_file_path(1)
        if not saving.exists(run_file_path):
            runs = self._load_raw_runs()
            runs = self._sort_runs(runs)
            self._save_efficient(runs)
        if not os.path.exists(self._get_scaler_path()):
            features, _ = self.load_runs(split)
            scaler = scaling.fit_scaler(features)
            scaling.save_scaler(scaler, self._get_scaler_path())

    def load_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self._validate_split(split)
        run_idx = self.run_split_dist[split]
        run_paths = [self._get_run_file_path(idx) for idx in run_idx]
        features, targets = saving.load_multiple(run_paths)

        return features, targets

    def load_scaler(self) -> scalers.StandardScaler:
        return scaling.load_scaler(self._get_scaler_path())

    def _load_raw_runs(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        file_paths = self._get_csv_file_paths()
        features = saving.load_raw(
            file_paths, self.DEFAULT_WINDOW_SIZE, columns=[0, 1], skip_rows=1
        )
        targets = utils.get_targets_from_file_paths(
            file_paths, self._timestep_from_file_path
        )
        runs = {idx: (features[idx], targets[idx]) for idx in features}

        return runs

    def _sort_runs(
        self, runs: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        sort_idx = {run_idx: np.argsort(t)[::-1] for run_idx, (_, t) in runs.items()}
        runs = {
            run_idx: (f[sort_idx[run_idx]], t[sort_idx[run_idx]])
            for run_idx, (f, t) in runs.items()
        }

        return runs

    def _get_csv_file_paths(self) -> Dict[int, List[str]]:
        fd_folder_path = self._get_fd_folder_path()
        file_paths = {}
        run_folders = self._get_run_folders(fd_folder_path)
        for run_idx, run_folder in run_folders.items():
            run_path = os.path.join(fd_folder_path, run_folder)
            feature_files = utils.get_files_in_path(run_path)
            file_paths[run_idx] = feature_files

        return file_paths

    def _get_run_folders(self, split_path: str) -> Dict[int, str]:
        all_folders = sorted(os.listdir(split_path))
        run_folders = {
            int(f[-1]): f
            for f in all_folders
            if os.path.isdir(os.path.join(split_path, f))
        }

        return run_folders

    @staticmethod
    def _timestep_from_file_path(file_path: str) -> int:
        file_name = os.path.basename(file_path)
        time_step = int(file_name.replace(".csv", ""))

        return time_step

    def _save_efficient(self, runs) -> None:
        for run_idx, (features, targets) in runs.items():
            saving.save(self._get_run_file_path(run_idx), features, targets)

    def _validate_split(self, split: str) -> None:
        if split not in ["dev", "val", "test"]:
            raise ValueError(
                "XJTU-SY provides a dev, val and test split, "
                f"but you provided '{split}'."
            )

    def _get_scaler_path(self) -> str:
        return os.path.join(self._get_fd_folder_path(), "scaler.pkl")

    def _get_run_file_path(self, run_idx: int) -> str:
        return os.path.join(self._get_fd_folder_path(), f"run_{run_idx}.npy")

    def _get_fd_folder_path(self) -> str:
        return os.path.join(self.data_root, self._FD_FOLDERS[self.fd])
