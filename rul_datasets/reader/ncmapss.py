import os
import warnings
from typing import Tuple, List, Optional, Union, Dict

import h5py  # type: ignore[import]
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import]

from rul_datasets.reader.data_root import get_data_root
from rul_datasets.reader import AbstractReader, scaling


class NCmapssReader(AbstractReader):
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
        return list(self._WINDOW_SIZES)

    def prepare_data(self) -> None:
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
        features, targets, auxiliary = self._load_raw_data()
        features, targets, auxiliary = self._split_by_unit(features, targets, auxiliary)
        max_window_size = []
        for feat, aux in zip(features, auxiliary):
            cycle_end_idx = self._get_end_idx(aux[:, 1])
            split_features = np.split(feat, cycle_end_idx[:-1])
            max_window_size.append(max(*[len(f) for f in split_features]))

        return max(*max_window_size)
