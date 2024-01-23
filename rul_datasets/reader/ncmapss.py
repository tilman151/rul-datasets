"""The New C-MAPSS Turbofan Degradation dataset is based on the same simulation as
[C-MAPSS][rul_datasets.reader.cmapss]. In contrast to the original dataset,
it contains fewer engine units, but each of them is recorded in more detail and under
more realistic operation conditions. Each unit has flight cycles recorded from the
healthy state until failure with RUL values assigned to the whole cycle. Inside a
flight cycle, data is recorded with a 1Hz resolution The dataset is split into seven
sub-datasets (`FD=1` to `FD=7`) that differ in the number of engine units and the
types of failures present.

Note:
    An eighth sub-dataset exists but is not present here as one of its data files seems
    corrupted. The dataset authors were already contacted about this issue."""

import os
import tempfile
import warnings
import zipfile
from typing import Tuple, List, Optional, Union, Dict

import h5py  # type: ignore[import]
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import]

from rul_datasets import utils
from rul_datasets.reader.data_root import get_data_root
from rul_datasets.reader import AbstractReader, scaling


NCMAPSS_DRIVE_ID = "1X9pHm2E3U0bZZbXIhJubVGSL3rtzqFkn"


class NCmapssReader(AbstractReader):
    """
    This reader provides access to the New C-MAPSS Turbofan Degradation dataset. Each
    of its seven sub-datasets contains a default train/val/test split which can be
    overridden by the `run_split_dist` argument.

    The features are provided as a windowed time series for each unit. The windows
    represent one flight cycle and are, by default, padded to the longest cycle in
    the sub-dataset. The window size can be overridden by the `window_size` argument
    which truncates each cycle at the end. Additionally, the features can be
    downsampled in time by taking the average of `resolution_seconds` consecutive
    time steps. The default channels are the four operating conditions,
    the 14 physical, and 14 virtual sensors in this order.

    The features are min-max scaled between zero and one. The scaler is fitted on the
    development data only. It is refit for each custom `run_split_dist` when
    `prepare_data` is called.

    Examples:
        Default channels
        >>> reader = NCmapssReader(fd=1)
        >>> reader.prepare_data()
        >>> features, labels = reader.load_split("dev")
        >>> features[0].shape
        (100, 20294, 32)

        Physical sensors only
        >>> reader = NCmapssReader(fd=1, feature_select=list(range(4, 18)))
        >>> reader.prepare_data()
        >>> features, labels = reader.load_split("dev")
        >>> features[0].shape
        (100, 20294, 14)

        Custom split and window size
        >>> reader = NCmapssReader(
        ...     fd=1,
        ...     run_split_dist={"dev": [0, 1], "val": [2], "test": [3]},
        ...     window_size=100,  # first 100 steps of each cycle
        ... )
        >>> reader.prepare_data()
        >>> features, labels = reader.load_split("dev")
        >>> features[0].shape
        (100, 100, 32)

        Downsampled features
        >>> reader = NCmapssReader(fd=1, resolution_seconds=10)
        >>> reader.prepare_data()
        >>> features, labels = reader.load_split("dev")
        >>> features[0].shape  # window size is automatically adjusted
        (100, 2029, 32)
    """

    _WINDOW_SIZES = {
        1: 20294,
        2: 27116,
        3: 20294,
        4: 20294,
        5: 20294,
        6: 20294,
        7: 20294,
    }
    _NUM_ENTITIES = {
        1: (6, 4),
        2: (6, 3),
        3: (9, 6),
        4: (6, 4),
        5: (6, 4),
        6: (6, 4),
        7: (6, 4),
    }
    _FILE_NAMES = {
        1: "N-CMAPSS_DS01-005.h5",
        2: "N-CMAPSS_DS02-006.h5",
        3: "N-CMAPSS_DS03-012.h5",
        4: "N-CMAPSS_DS04.h5",
        5: "N-CMAPSS_DS05.h5",
        6: "N-CMAPSS_DS06.h5",
        7: "N-CMAPSS_DS07.h5",
    }
    _NCMAPSS_ROOT: str = os.path.join(get_data_root(), "NCMAPSS")

    def __init__(
        self,
        fd: int,
        window_size: Optional[int] = None,
        max_rul: Optional[int] = 65,
        percent_broken: Optional[float] = None,
        percent_fail_runs: Optional[Union[float, List[int]]] = None,
        feature_select: Optional[List[int]] = None,
        truncate_val: bool = False,
        run_split_dist: Optional[Dict[str, List[int]]] = None,
        truncate_degraded_only: bool = False,
        resolution_seconds: int = 1,
        padding_value: float = 0.0,
    ) -> None:
        """
        Create a new reader for the New C-MAPSS dataset. The maximum RUL value is set
        to 65 by default. The default channels are the four operating conditions,
        the 14 physical, and 14 virtual sensors in this order.

        The default window size is, by default, the longest flight cycle in the
        sub-dataset. Shorter cycles are padded on the left. The default padding value
        is zero but can be overridden, e.g., as -1 to make filtering for padding easier
        later on.

        The default `run_split_dist` is the same as in the original dataset, but with
        the last unit of the original train split designated for validation.

        If the features are downsampled in time, the default window size is
        automatically adjusted to `window_size // resolution_seconds`. Any manually
        set `window_size` needs to take this into account as it is applied after
        downsampling.

        For more information about using readers, refer to the [reader]
        [rul_datasets.reader] module page.

        Args:
            fd: The sub-dataset to use. Must be in `[1, 7]`.
            max_rul: The maximum RUL value.
            percent_broken: The maximum relative degradation per unit.
            percent_fail_runs: The percentage or index list of available units.
            feature_select: The indices of the features to use.
            truncate_val: Truncate the validation data with `percent_broken`, too.
            run_split_dist: The assignment of units to each split.
            truncate_degraded_only: Only truncate the degraded part of the data
                                    (< max RUL).
            resolution_seconds: The number of consecutive seconds to average over for
                                downsampling.
            padding_value: The value to use for padding the flight cycles.
        """
        super().__init__(
            fd,
            window_size,
            max_rul,
            percent_broken,
            percent_fail_runs,
            truncate_val,
            truncate_degraded_only,
        )
        self.feature_select = feature_select or list(range(32))
        self.run_split_dist = run_split_dist or self._get_default_split(self.fd)
        self.resolution_seconds = resolution_seconds
        self.padding_value = padding_value

        if self.resolution_seconds > 1 and window_size is None:
            warnings.warn(
                "When `resolution_seconds` > 1 with the default `window_size`, "
                "the `window_size` is automatically adjusted to "
                "`window_size // resolution_seconds`."
            )
            self.window_size //= self.resolution_seconds

    @property
    def hparams(self):
        hparams = super().hparams
        hparams.update(
            {
                "run_split_dist": self.run_split_dist,
                "feature_select": self.feature_select,
                "padding_value": self.padding_value,
            }
        )

        return hparams

    @property
    def dataset_name(self) -> str:
        return "ncmapss"

    @property
    def fds(self) -> List[int]:
        """Indices of the available sub-datasets."""
        return list(self._WINDOW_SIZES)

    def prepare_data(self) -> None:
        """
        Prepare the N-C-MAPSS dataset. This function needs to be called before using the
        dataset for the first time.

        The dataset is assumed to be present in the data root directory. The training
        data is then split into development and validation set. Afterward, a scaler
        is fit on the development features if it was not already done previously.
        """
        if not os.path.exists(self._NCMAPSS_ROOT):
            _download_ncmapss(self._NCMAPSS_ROOT)
        if not os.path.exists(self._get_scaler_path()):
            features, _, _ = self._load_data("dev")
            scaler = scaling.fit_scaler(features, MinMaxScaler())
            scaling.save_scaler(scaler, self._get_scaler_path())

    def _get_scaler_path(self):
        file_name = f"scaler_{self.fd}_{self.run_split_dist['dev']}.pkl"
        file_path = os.path.join(self._NCMAPSS_ROOT, file_name)

        return file_path

    def default_window_size(self, fd: int) -> int:
        return self._WINDOW_SIZES[fd]

    def _get_default_split(self, fd: int) -> Dict[str, List[int]]:
        num_train, num_test = self._NUM_ENTITIES[fd]
        num_entities = num_train + num_test
        num_val = int(np.round(num_train * 0.2))
        num_dev = num_train - num_val
        run_split_dist = {
            "dev": list(range(num_dev)),
            "val": list(range(num_dev, num_train)),
            "test": list(range(num_train, num_entities)),
        }

        return run_split_dist

    def load_complete_split(
        self, split: str, alias: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features, targets, auxiliary = self._load_data(split)
        features = scaling.scale_features(features, self._load_scaler())
        features = [f[:, self.feature_select] for f in features]
        windowed = [
            self._window_by_cycle(*unit) for unit in zip(features, targets, auxiliary)
        ]
        features, targets = zip(*windowed)

        return list(features), list(targets)

    def _load_data(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        features, targets, auxiliary = self._load_raw_data()
        features, targets, auxiliary = self._split_by_unit(features, targets, auxiliary)
        features = self._select_units(features, split)
        targets = self._select_units(targets, split)
        auxiliary = self._select_units(auxiliary, split)

        return features, targets, auxiliary

    def _load_raw_data(self):
        file_path = os.path.join(self._NCMAPSS_ROOT, self._FILE_NAMES[self.fd])
        with h5py.File(file_path, mode="r") as hdf:
            work_cond = self._load_series(hdf, "W")
            sensor_phys = self._load_series(hdf, "X_s")
            sensor_virt = self._load_series(hdf, "X_v")
            rul = self._load_series(hdf, "Y")
            auxiliary = self._load_series(hdf, "A")
        features = np.concatenate([work_cond, sensor_phys, sensor_virt], axis=1)
        rul = rul.squeeze(axis=1)

        return features, rul, auxiliary

    @staticmethod
    def _load_series(hdf: h5py.File, name: str) -> np.ndarray:
        series_dev = np.array(hdf.get(f"{name}_dev"))
        series_test = np.array(hdf.get(f"{name}_test"))
        series = np.concatenate([series_dev, series_test])

        return series

    @staticmethod
    def _split_by_unit(
        features: np.ndarray, targets: np.ndarray, auxiliary: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        unit_split_idx = NCmapssReader._get_end_idx(auxiliary[:, 0])[:-1]
        split_features = np.split(features, unit_split_idx)
        split_targets = np.split(targets, unit_split_idx)
        split_auxiliary = np.split(auxiliary, unit_split_idx)

        return split_features, split_targets, split_auxiliary

    def _select_units(self, units, split):
        return [units[i] for i in self.run_split_dist[split]]

    def _window_by_cycle(
        self, features: np.ndarray, targets: np.ndarray, auxiliary: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cycle_end_idx = self._get_end_idx(auxiliary[:, 1])
        split_features = np.split(features, cycle_end_idx[:-1])
        split_features = self._downsample(split_features)
        split_features = [
            np.pad(
                f,
                ((self.window_size - len(f), 0), (0, 0)),
                "constant",
                constant_values=self.padding_value,
            )
            for f in (f[: self.window_size] for f in split_features)
        ]
        features = np.stack(split_features, axis=0)
        targets = targets[cycle_end_idx - 1]

        return features, targets

    @staticmethod
    def _get_end_idx(identifiers):
        _, split_idx = np.unique(identifiers, return_counts=True)
        split_idx = np.cumsum(split_idx)

        return split_idx

    def _downsample(self, features: List[np.ndarray]) -> List[np.ndarray]:
        if self.resolution_seconds == 1:
            return features

        downsampled = []
        for f in features:
            remainder = len(f) % self.resolution_seconds
            f = f[:-remainder] if remainder else f
            f = f.reshape(-1, self.resolution_seconds, f.shape[1]).mean(axis=1)
            downsampled.append(f)

        return downsampled

    def _load_scaler(self) -> MinMaxScaler:
        if not os.path.exists(self._get_scaler_path()):
            raise RuntimeError(
                "Scaler for this dataset does not exist. "
                "Please call `prepare_data` first."
            )

        return scaling.load_scaler(self._get_scaler_path())

    def _calc_default_window_size(self):
        """Only for development purposes."""
        features, targets, auxiliary = self._load_raw_data()
        features, targets, auxiliary = self._split_by_unit(features, targets, auxiliary)
        max_window_size = []
        for feat, aux in zip(features, auxiliary):
            cycle_end_idx = self._get_end_idx(aux[:, 1])
            split_features = np.split(feat, cycle_end_idx[:-1])
            max_window_size.append(max(*[len(f) for f in split_features]))

        return max(*max_window_size)


def _download_ncmapss(data_root):
    with tempfile.TemporaryDirectory() as tmp_path:
        print("Download N-C-MAPSS dataset from Google Drive")
        download_path = os.path.join(tmp_path, "data.zip")
        utils.download_gdrive_file(NCMAPSS_DRIVE_ID, download_path)
        print("Extract N-C-MAPSS dataset")
        os.makedirs(data_root)
        with zipfile.ZipFile(download_path, mode="r") as f:
            for zipinfo in f.infolist():
                zipinfo.filename = os.path.basename(zipinfo.filename)
                if zipinfo.filename:
                    f.extract(zipinfo, data_root)
