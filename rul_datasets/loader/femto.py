import os
import re
import tempfile
import warnings
import zipfile
from typing import List, Tuple, Union, Dict, Optional

import numpy as np
import sklearn.preprocessing as scalers  # type: ignore

from rul_datasets import utils
from rul_datasets.loader import scaling, saving
from rul_datasets.loader.abstract import AbstractLoader, DATA_ROOT

FEMTO_URL = "https://kr0k0tsch.de/rul-datasets/FEMTOBearingDataSet.zip"


class FemtoLoader(AbstractLoader):
    _FEMTO_ROOT: str = os.path.join(DATA_ROOT, "FEMTOBearingDataSet")
    _NUM_TRAIN_RUNS: Dict[int, int] = {1: 2, 2: 2, 3: 2}

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
        self._preparator = FemtoPreparator(self.fd, self._FEMTO_ROOT, run_split_dist)

    @property
    def fds(self) -> List[int]:
        return list(self._NUM_TRAIN_RUNS)

    def prepare_data(self) -> None:
        if not os.path.exists(self._FEMTO_ROOT):
            _download_femto(DATA_ROOT)
        self._preparator.prepare_split("dev")
        self._preparator.prepare_split("val")
        self._preparator.prepare_split("test")

    def _load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, targets = self._preparator.load_runs(split)
        features = [f[:, -self.window_size :, :] for f in features]  # crop to window
        features = scaling.scale_features(features, self._preparator.load_scaler())

        return features, targets

    def _default_window_size(self, fd: int) -> int:
        return FemtoPreparator.DEFAULT_WINDOW_SIZE


class FemtoPreparator:
    DEFAULT_WINDOW_SIZE = 2560
    _SPLIT_FOLDERS = {"dev": "Learning_set", "val": "Full_Test_Set", "test": "Test_set"}
    _DEFAULT_RUN_SPLIT_DIST = {
        1: {"dev": [1, 2], "val": [3], "test": [4, 5, 6, 7]},
        2: {"dev": [1, 2], "val": [3], "test": [4, 5, 6, 7]},
        3: {"dev": [1], "val": [2], "test": [3]},
    }

    def __init__(
        self,
        fd: int,
        data_root: str,
        run_split_dist: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.fd = fd
        self._data_root = data_root
        self.run_split_dist = run_split_dist or self._DEFAULT_RUN_SPLIT_DIST[self.fd]

    def prepare_split(self, split: str) -> None:
        if not self._split_already_prepared(split):
            warnings.warn(f"First time use. Pre-process {split} split of FD{self.fd}.")
            runs = self._load_raw_runs(split)
            self._save_efficient(split, runs)
        if split == "dev" and not os.path.exists(self._get_scaler_path()):
            features, _ = self.load_runs(split)
            scaler = scaling.fit_scaler(features)
            scaling.save_scaler(scaler, self._get_scaler_path())

    def _split_already_prepared(self, split: str) -> bool:
        run_idx_in_split = self._DEFAULT_RUN_SPLIT_DIST[self.fd][split][0]
        run_file_path = self._get_run_file_path(split, run_idx_in_split)
        already_prepared = saving.exists(run_file_path)

        return already_prepared

    def load_runs(self, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self._validate_split(split)
        run_idx = self.run_split_dist[split]
        run_paths = [self._get_run_file_path(split, idx) for idx in run_idx]
        features, targets = saving.load_multiple(run_paths)

        return features, targets

    def load_scaler(self) -> scalers.StandardScaler:
        return scaling.load_scaler(self._get_scaler_path())

    def _load_raw_runs(self, split: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        file_paths = self._get_csv_file_paths(split)
        features = saving.load_raw(file_paths, self.DEFAULT_WINDOW_SIZE, columns=[4, 5])
        targets = utils.get_targets_from_file_paths(
            file_paths, self._timestep_from_file_path
        )
        runs = {idx: (features[idx], targets[idx]) for idx in features}

        return runs

    def _get_csv_file_paths(self, split: str) -> Dict[int, List[str]]:
        split_path = self._get_split_folder(split)
        run_folders = self._get_run_folders(split_path)
        file_paths = {}
        for run_idx, run_folder in run_folders.items():
            run_path = os.path.join(split_path, run_folder)
            feature_files = utils.get_files_in_path(
                run_path, lambda f: f.startswith("acc")
            )
            file_paths[run_idx] = feature_files

        return file_paths

    @staticmethod
    def _timestep_from_file_path(file_path: str) -> int:
        file_name = os.path.basename(file_path)
        time_step = int(file_name[4:9])

        return time_step

    def _save_efficient(
        self, split: str, runs: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        for run_idx, (features, targets) in runs.items():
            saving.save(self._get_run_file_path(split, run_idx), features, targets)

    def _validate_split(self, split: str) -> None:
        if split not in self._SPLIT_FOLDERS:
            raise ValueError(f"Unsupported split '{split}' supplied.")

    def _get_run_folders(self, split_path: str) -> Dict[int, str]:
        pattern = self._get_run_folder_pattern()
        content = sorted(os.listdir(split_path))
        run_folders = {int(f[-1]): f for f in content if pattern.match(f) is not None}

        return run_folders

    def _get_run_folder_pattern(self) -> re.Pattern:
        return re.compile(rf"Bearing{self.fd}_\d")

    def _get_scaler_path(self) -> str:
        return os.path.join(self._get_split_folder("dev"), f"scaler_{self.fd}.pkl")

    def _get_run_file_path(self, split: str, run_idx: int) -> str:
        return os.path.join(
            self._get_split_folder(split), f"run_{self.fd}_{run_idx}.npy"
        )

    def _get_split_folder(self, split: str) -> str:
        return os.path.join(self._data_root, self._SPLIT_FOLDERS[split])


def _download_femto(data_root: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_path:
        print("Download FEMTO dataset")
        download_path = os.path.join(tmp_path, "FEMTO.zip")
        utils.download_file(FEMTO_URL, download_path)
        print("Extract FEMTO dataset")
        with zipfile.ZipFile(download_path, mode="r") as f:
            f.extractall(data_root)
