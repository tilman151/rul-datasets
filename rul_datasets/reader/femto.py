"""The FEMTO (PRONOSTIA) Bearing dataset is a collection of run-to-failure
experiments on bearings. Three different operation conditions were used, resulting in
three sub-datasets. Sub-dataset 1 and 2 contain two training runs and five test runs,
while sub-dataset 3 contains only one test run. It was part of the 2012 IEEE
Prognostics Challenge. """

import os
import re
import tempfile
import warnings
import zipfile
from typing import List, Tuple, Union, Dict, Optional

import numpy as np
import sklearn.preprocessing as scalers  # type: ignore

from rul_datasets import utils
from rul_datasets.reader import scaling, saving
from rul_datasets.reader.data_root import get_data_root
from rul_datasets.reader.abstract import AbstractReader

FEMTO_URL = "https://kr0k0tsch.de/rul-datasets/FEMTOBearingDataSet.zip"


class FemtoReader(AbstractReader):
    """
    This reader represents the FEMTO (PRONOSTIA) Bearing dataset. Each of its three
    sub-datasets contain a training and a test split. By default, the reader
    constructs a validation split for sub-datasets 1 and 2 each by taking the first
    run of the test split. For sub-dataset 3 the second training run is used for
    validation because only one test run is available. The remaining training data is
    denoted as the development split. This run to split assignment can be overridden
    by setting `run_split_dist`.

    The features contain windows with three channels. Only the two acceleration
    channels are used because the test runs are missing the temperature channel.
    These features are standardized to zero mean and one standard deviation. The
    scaler is fitted on the development data.

    Examples:
        Default splits:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.reader.FemtoReader(fd=1)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> features[0].shape
        (2803, 2560, 2)

        Custom splits:
        >>> import rul_datasets
        >>> splits = {"dev": [5], "val": [4], "test": [3]}
        >>> fd1 = rul_datasets.reader.FemtoReader(fd=1, run_split_dist=splits)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> features[0].shape
        (2463, 2560, 2)

        Set first-time-to-predict:
        >>> import rul_datasets
        >>> fttp = [10, 20, 30, 40, 50]
        >>> fd1 = rul_datasets.reader.FemtoReader(fd=1, first_time_to_predict=fttp)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> labels[0][:15]
        array([2793., 2793., 2793., 2793., 2793., 2793., 2793., 2793., 2793.,
               2793., 2793., 2792., 2791., 2790., 2789.])
    """

    _FEMTO_ROOT: str = os.path.join(get_data_root(), "FEMTOBearingDataSet")
    _NUM_TRAIN_RUNS: Dict[int, int] = {1: 2, 2: 2, 3: 2}

    def __init__(
        self,
        fd: int,
        window_size: Optional[int] = None,
        max_rul: Optional[int] = None,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Optional[Union[float, List[int]]] = None,
        truncate_val: bool = False,
        run_split_dist: Optional[Dict[str, List[int]]] = None,
        first_time_to_predict: List[int] = None,
        norm_rul: bool = False,
    ) -> None:
        """
        Create a new FEMTO reader for one of the sub-datasets. By default, the RUL
        values are not capped. The default window size is 2560.

         Use `first_time_to_predict` to set an individual RUL inflection point for
        each run. It should be a list with an integer index for each run. The index
        is the time step after which RUL declines. Before the time step it stays
        constant. The `norm_rul` argument can then be used to scale the RUL of each
        run between zero and one.

        For more information about using readers refer to the [reader]
        [rul_datasets.reader] module page.

        Args:
            fd: Index of the selected sub-dataset
            window_size: Size of the sliding window. Defaults to 2560.
            max_rul: Maximum RUL value of targets.
            percent_broken: The maximum relative degradation per time series.
            percent_fail_runs: The percentage or index list of available time series.
            truncate_val: Truncate the validation data with `percent_broken`, too.
            run_split_dist: Dictionary that assigns each run idx to each split.
            first_time_to_predict: The time step for each time series before which RUL
                                   is constant.
            norm_rul: Normalize RUL between zero and one.
        """
        super().__init__(
            fd, window_size, max_rul, percent_broken, percent_fail_runs, truncate_val
        )

        if (first_time_to_predict is not None) and (max_rul is not None):
            raise ValueError(
                "FemtoReader cannot use 'first_time_to_predict' "
                "and 'max_rul' in conjunction."
            )

        self.first_time_to_predict = first_time_to_predict
        self.norm_rul = norm_rul

        self._preparator = FemtoPreparator(self.fd, self._FEMTO_ROOT, run_split_dist)

    @property
    def fds(self) -> List[int]:
        """Indices of available sub-datasets."""
        return list(self._NUM_TRAIN_RUNS)

    def prepare_data(self) -> None:
        """
        Prepare the FEMTO dataset. This function needs to be called before using the
        dataset and each custom split for the first time.

        The dataset is downloaded from a custom mirror and extracted into the data
        root directory. The whole dataset is converted from CSV files to NPY files to
        speed up loading it from disk. Afterwards, a scaler is fit on the development
        features. Previously completed steps are skipped.
        """
        if not os.path.exists(self._FEMTO_ROOT):
            _download_femto(get_data_root())
        self._preparator.prepare_split("dev")
        self._preparator.prepare_split("val")
        self._preparator.prepare_split("test")

    def load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, targets = self._preparator.load_runs(split)
        features = [f[:, -self.window_size :, :] for f in features]  # crop to window
        features = scaling.scale_features(features, self._preparator.load_scaler())
        if self.max_rul is not None:
            targets = [np.minimum(t, self.max_rul) for t in targets]
        elif self.first_time_to_predict is not None:
            targets = self._clip_first_time_to_predict(targets, split)

        if self.norm_rul:
            targets = [t / np.max(t) for t in targets]

        return features, targets

    def _clip_first_time_to_predict(self, targets, split):
        fttp = [
            self.first_time_to_predict[i - 1]
            for i in self._preparator.run_split_dist[split]
        ]
        targets = [np.minimum(t, len(t) - fttp) for t, fttp in zip(targets, fttp)]

        return targets

    def default_window_size(self, fd: int) -> int:
        return FemtoPreparator.DEFAULT_WINDOW_SIZE


class FemtoPreparator:
    DEFAULT_WINDOW_SIZE = 2560
    _SPLIT_FOLDERS = {
        "dev": "Learning_set",
        "val": "Full_Test_Set",
        "test": "Full_Test_Set",
    }
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
        file_name = f"scaler_{self.fd}_{self.run_split_dist['dev']}.pkl"
        file_path = os.path.join(self._data_root, file_name)

        return file_path

    def _get_run_file_path(self, split: str, run_idx: int) -> str:
        return os.path.join(self._data_root, f"run_{self.fd}_{run_idx}.npy")

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
