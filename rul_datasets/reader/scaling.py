"""A module with functions for scaling RUL features."""
import copy
import pickle
from typing import List, Optional, Union, Tuple

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

_Scaler = (
    scalers.StandardScaler,
    scalers.MinMaxScaler,
    scalers.MaxAbsScaler,
    scalers.RobustScaler,
)
Scaler = Union[
    scalers.StandardScaler,
    scalers.MinMaxScaler,
    scalers.MaxAbsScaler,
    scalers.RobustScaler,
]
"""
Supported scalers:

* [sklearn.preprocessing.StandardScaler][]
* [sklearn.preprocessing.MinMaxScaler][]
* [sklearn.preprocessing.MaxAbsScaler][]
* [sklearn.preprocessing.RobustScaler][]
"""


class OperationConditionAwareScaler(BaseEstimator, TransformerMixin):
    """This scaler is an ensemble of multiple base scalers, e.g. [
    sklearn.preprocessing.MinMaxScaler][]. It takes an additional operation condition
    array while fitting and transforming that controls which base scaler is used. The
    operation condition corresponding to a sample is compared against the boundaries
    defined during construction of the scaler. If the condition lies between the
    first set of boundaries, the first base scaler is used, and so forth.
    If any condition does not fall between any boundaries, an exception will be
    raised and the boundaries should be adjusted."""

    def __init__(
        self, base_scaler: Scaler, boundaries: List[Tuple[float, float]]
    ) -> None:
        """
        Create a new scaler aware of operation conditions.

        Each pair in `boundaries` represents the lower and upper value of an
        inclusive interval. For each interval a copy of the `base_scaler` is
        maintained. If an operation condition value falls inside an interval,
        the corresponding scaler is used. The boundaries have to be mutually exclusive.

        Args:
            base_scaler: The scaler that should be used for each condition.
            boundaries: The pairs that form the inclusive boundaries of each condition.
        """
        self.base_scalers = [copy.deepcopy(base_scaler) for _ in boundaries]
        self.boundaries = boundaries

        self._check_boundaries_mutually_exclusive()

    def _check_boundaries_mutually_exclusive(self):
        b = sorted(self.boundaries, key=lambda x: x[0])
        exclusive = all(up < low for (_, up), (low, _) in zip(b[:-1], b[1:]))
        if not exclusive:
            raise ValueError(
                "Boundaries are not mutually exclusive. Be aware that "
                "the boundaries are inclusive, i.e. lower <= value <= upper."
            )

    @property
    def n_features_in_(self):
        """Number of expected input features."""
        return self.base_scalers[0].n_features_in_

    def partial_fit(
        self, features: np.ndarray, operation_conditions: np.ndarray
    ) -> "OperationConditionAwareScaler":
        """
        Fit the base scalers partially.

        This function calls `partial_fit` on each of the base scalers with the
        samples that fall into the corresponding condition boundaries. If any sample
        does not fall into one of the boundaries, an exception is raised.

        Args:
            features: The feature array to be scaled.
            operation_conditions: The condition values compared against the boundaries.
        Returns:
            The partially fitted scaler.
        """
        total = 0
        for i, (lower, upper) in enumerate(self.boundaries):
            idx = self._between(operation_conditions, lower, upper)
            if num_elem := np.sum(idx):  # guard against empty array
                self.base_scalers[i].partial_fit(features[idx])
                total += num_elem
        self._check_all_transformed(features, total, "fitted")

        return self

    def transform(
        self, features: np.ndarray, operation_conditions: np.ndarray
    ) -> np.ndarray:
        """
        Scale the features with the appropriate condition aware scaler.

        This function calls `transform` on each of the base scalers for the
        samples that fall into the corresponding condition boundaries. If any sample
        does not fall into one of the boundaries, an exception is raised.

        Args:
            features: The features to be scaled.
            operation_conditions: The condition values compared against the boundaries.
        Returns:
            The scaled features.
        """
        scaled = np.empty_like(features)
        total = 0
        for i, (lower, upper) in enumerate(self.boundaries):
            idx = self._between(operation_conditions, lower, upper)
            if num_elem := np.sum(idx):  # guard against empty array
                scaled[idx] = self.base_scalers[i].transform(features[idx])
                total += num_elem
        self._check_all_transformed(features, total, "scaled")

        return scaled

    def _check_all_transformed(self, features, total, activity):
        """Guard against unknown conditions"""
        if diff := (len(features) - total):
            raise RuntimeError(
                f"{diff} samples had an unknown condition and could not be {activity}."
                "Please adjust the boundaries."
            )

    def _between(self, inputs: np.ndarray, lower: float, upper: float) -> np.ndarray:
        """Inclusive between."""
        return (lower <= inputs) & (inputs <= upper)


def fit_scaler(
    features: List[np.ndarray],
    scaler: Optional[Union[Scaler, OperationConditionAwareScaler]] = None,
    operation_conditions: Optional[List[np.ndarray]] = None,
) -> Union[Scaler, OperationConditionAwareScaler]:
    """
    Fit a given scaler to the RUL features. If the scaler is omitted,
    a StandardScaler will be created.

    If the scaler is an [OperationConditionAwareScaler][
    rul_datasets.reader.scaling.OperationConditionAwareScaler] and
    `operation_conditions` are passed, the scaler will be fit aware of operation
    conditions.

    The scaler assumes that the last axis of the features are the channels. Only
    scalers unaware of operation conditions can be fit with windowed data.

    Args:
        features: The RUL features.
        scaler: The scaler to be fit. Defaults to a StandardScaler.
        operation_conditions: The operation conditions for condition aware scaling.
    Returns:
        The fitted scaler.
    """
    scaler = scaler or scalers.StandardScaler()
    if isinstance(scaler, Scaler.__args__):  # type: ignore[attr-defined]
        scaler = _fit_scaler_naive(features, scaler)
    elif operation_conditions is not None and isinstance(
        scaler, OperationConditionAwareScaler
    ):
        scaler = _fit_scaler_operation_condition_aware(
            features, scaler, operation_conditions
        )
    else:
        raise ValueError(
            "Unsupported combination of scaler type and operation conditions."
        )

    return scaler


def _fit_scaler_naive(features: List[np.ndarray], scaler: Scaler) -> Scaler:
    for run in features:
        run = run.reshape(-1, run.shape[-1])
        scaler.partial_fit(run)

    return scaler


def _fit_scaler_operation_condition_aware(
    features: List[np.ndarray],
    scaler: OperationConditionAwareScaler,
    operation_conditions: List[np.ndarray],
) -> OperationConditionAwareScaler:
    assert len(features[0].shape) == 2, "Condition aware scaling can't fit window data"
    for run, cond in zip(features, operation_conditions):
        scaler.partial_fit(run, cond)

    return scaler


def save_scaler(scaler: Scaler, save_path: str) -> None:
    """
    Save a scaler to disk.

    Args:
        scaler: The scaler to be saved.
        save_path: The path to save the scaler to.
    """
    with open(save_path, mode="wb") as f:
        pickle.dump(scaler, f)


def load_scaler(save_path: str) -> Scaler:
    """
    Load a scaler from disk.

    Args:
        save_path: The path the scaler was saved to.
    Returns:
        The loaded scaler.
    """
    with open(save_path, mode="rb") as f:
        scaler = pickle.load(f)

    return scaler


def scale_features(
    features: List[np.ndarray],
    scaler: Union[Scaler, OperationConditionAwareScaler],
    operation_conditions: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    Scaler the RUL features with a given scaler.

    The features can have a shape of `[num_time_steps, channels]` or `[num_windows,
    window_size, channels]`. The scaler needs to work on the channel dimension. If it
    was not fit with the right number of channels, a `ValueError` is thrown.

    If the scaler is operation condition aware, the `operation_conditions` argument
    needs to be passed. Windowed data cannot be fit this way.

    Args:
        features: The RUL features to be scaled.
        scaler: The already fitted scaler.
        operation_conditions: The operation conditions for condition aware scaling.
    Returns:
        The scaled features.
    """
    if operation_conditions is None:
        features = _scale_features_naive(features, scaler)
    else:
        features = _scale_features_condition_aware(
            features, scaler, operation_conditions
        )

    return features


def _scale_features_naive(
    features: List[np.ndarray], scaler: Scaler
) -> List[np.ndarray]:
    features = copy.copy(features)
    for i, run in enumerate(features):
        _check_channels(run, scaler)
        if len(run.shape) == 3:
            features[i] = _scale_windowed_features(run, scaler)
        else:
            features[i] = scaler.transform(run)

    return features


def _scale_features_condition_aware(
    features: List[np.ndarray],
    scaler: OperationConditionAwareScaler,
    operation_conditions: List[np.ndarray],
) -> List[np.ndarray]:
    assert len(features[0].shape) == 2, "No condition aware scaling for window data"
    features = copy.copy(features)
    for i, (run, cond) in enumerate(zip(features, operation_conditions)):
        _check_channels(run, scaler)
        features[i] = scaler.transform(run, cond)

    return features


def _check_channels(
    run: np.ndarray, scaler: Union[Scaler, OperationConditionAwareScaler]
) -> None:
    if not run.shape[-1] == scaler.n_features_in_:
        raise ValueError(
            f"The scaler was fit on {scaler.n_features_in_} "
            f"channels but the features have {run.shape[1]} channels."
        )


def _scale_windowed_features(features: np.ndarray, scaler: Scaler) -> np.ndarray:
    num_channels = features.shape[2]
    window_size = features.shape[1]
    features = features.reshape(-1, num_channels)
    features = scaler.transform(features)
    features = features.reshape(-1, window_size, num_channels)

    return features
