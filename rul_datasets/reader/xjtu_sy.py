"""The XJTU-SY Bearing dataset is a collection of run-to-failure experiments on
bearings. Three different operation conditions were used, resulting in three
sub-datasets. Each sub-dataset contains five runs without an official training/test
split. """

import os.path
import tempfile
import zipfile
from typing import Tuple, List, Union, Dict, Optional

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore

from rul_datasets import utils
from rul_datasets.reader import saving, scaling
from rul_datasets.reader.abstract import AbstractReader
from rul_datasets.reader.data_root import get_data_root

XJTU_SY_URL = "https://kr0k0tsch.de/rul-datasets/XJTU-SY.zip"


class XjtuSyReader(AbstractReader):
    """
    This reader represents the XJTU-SY Bearing dataset. Each of its three
    sub-datasets contains five runs. By default, the reader assigns the first two to
    the development, the third to the validation and the remaining two to the test
    split. This run to split assignment can be overridden by setting `run_split_dist`.

    The features contain windows with two channels of acceleration data which are
    standardized to zero mean and one standard deviation. The scaler is fitted on the
    development data.

    Examples:
        Default splits:
        >>> import rul_datasets
        >>> fd1 = rul_datasets.reader.XjtuSyReader(fd=1)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> features[0].shape
        (123, 32768, 2)

        Custom splits:
        >>> import rul_datasets
        >>> splits = {"dev": [5], "val": [4], "test": [3]}
        >>> fd1 = rul_datasets.reader.XjtuSyReader(fd=1, run_split_dist=splits)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> features[0].shape
        (52, 32768, 2)

        Set first-time-to-predict:
        >>> import rul_datasets
        >>> fttp = [10, 20, 30, 40, 50]
        >>> fd1 = rul_datasets.reader.XjtuSyReader(fd=1, first_time_to_predict=fttp)
        >>> fd1.prepare_data()
        >>> features, labels = fd1.load_split("dev")
        >>> labels[0][:15]
        array([113., 113., 113., 113., 113., 113., 113., 113., 113., 113., 113.,
               112., 111., 110., 109.])
    """

    _XJTU_SY_ROOT: str = os.path.join(get_data_root(), "XJTU-SY")
    _NUM_TRAIN_RUNS: Dict[int, int] = {1: 5, 2: 5, 3: 5}

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
        Create a new XJTU-SY reader for one of the sub-datasets. By default, the RUL
        values are not capped. The default window size is 32768.

        Use `first_time_to_predict` to set an individual RUL inflection point for
        each run. It should be a list with an integer index for each run. The index
        is the time step after which RUL declines. Before the time step it stays
        constant. The `norm_rul` argument can then be used to scale the RUL of each
        run between zero and one.

        For more information about using readers refer to the [reader]
        [rul_datasets.reader] module page.

        Args:
            fd: Index of the selected sub-dataset
            window_size: Size of the sliding window. Defaults to 32768.
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

        self._preparator = XjtuSyPreparator(self.fd, self._XJTU_SY_ROOT, run_split_dist)

    @property
    def fds(self) -> List[int]:
        """Indices of available sub-datasets."""
        return list(self._NUM_TRAIN_RUNS)

    def prepare_data(self) -> None:
        """
        Prepare the XJTU-SY dataset. This function needs to be called before using the
        dataset and each custom split for the first time.

        The dataset is downloaded from a custom mirror and extracted into the data
        root directory. The whole dataset is converted com CSV files to NPY files to
        speed up loading it from disk. Afterwards, a scaler is fit on the development
        features. Previously completed steps are skipped.
        """
        if not os.path.exists(self._XJTU_SY_ROOT):
            _download_xjtu_sy(get_data_root())
        self._preparator.prepare_split()

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

    def prepare_split(self, split: Optional[str] = None) -> None:
        run_file_path = self._get_run_file_path(1)
        if not saving.exists(run_file_path):
            runs = self._load_raw_runs()
            runs = self._sort_runs(runs)
            self._save_efficient(runs)
        if not os.path.exists(self._get_scaler_path()):
            features, _ = self.load_runs("dev")
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
        file_name = f"scaler_{self.run_split_dist['dev']}.pkl"
        file_path = os.path.join(self._get_fd_folder_path(), file_name)

        return file_path

    def _get_run_file_path(self, run_idx: int) -> str:
        return os.path.join(self._get_fd_folder_path(), f"run_{run_idx}.npy")

    def _get_fd_folder_path(self) -> str:
        return os.path.join(self.data_root, self._FD_FOLDERS[self.fd])


def _download_xjtu_sy(data_root: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_path:
        print("Download XJTU-SY dataset")
        download_path = os.path.join(tmp_path, "XJTU-SY.zip")
        utils.download_file(XJTU_SY_URL, download_path)
        print("Extract XJTU-SY dataset")
        with zipfile.ZipFile(download_path, mode="r") as f:
            f.extractall(data_root)
