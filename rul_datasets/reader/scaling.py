"""A module with functions for scaling RUL features."""

import pickle
from typing import List, Optional, Union

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore


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


def fit_scaler(features: List[np.ndarray], scaler: Optional[Scaler] = None) -> Scaler:
    """
    Fit a given scaler to the RUL features. If the scaler is omitted,
    a StandardScaler will be created.

    Args:
        features: The RUL features.
        scaler: The scaler to be fit. Defaults to a StandardScaler.
    Returns:
        The fitted scaler
    """
    if scaler is None:
        scaler = scalers.StandardScaler()
    for run in features:
        run = run.reshape(-1, run.shape[-1])
        scaler.partial_fit(run)

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


def scale_features(features: List[np.ndarray], scaler: Scaler) -> List[np.ndarray]:
    """
    Scaler the RUL features with a given scaler.

    The features can have a shape of `[num_time_steps, channels]` or `[num_windows,
    channels, window_size]`. The scaler needs to work on the channel dimension. If it
    was not fit with the right number of channels, a `ValueError` is thrown.

    Args:
        features: The RUL features to be scaled.
        scaler: The already fitted scaler.
    Returns:
        The scaled features.
    """
    for i, run in enumerate(features):
        _check_channels(run, scaler)
        if len(run.shape) == 3:
            features[i] = _scale_windowed_features(run, scaler)
        else:
            features[i] = scaler.transform(run)

    return features


def _check_channels(run: np.ndarray, scaler: Scaler) -> None:
    if not run.shape[1] == scaler.n_features_in_:
        raise ValueError(
            f"The scaler was fit on {scaler.n_features_in_} "
            f"channels but the features have {run.shape[1]} channels."
        )


def _scale_windowed_features(features: np.ndarray, scaler: Scaler) -> np.ndarray:
    num_channels = features.shape[1]
    window_size = features.shape[2]
    features = features.reshape(-1, num_channels)
    features = scaler.transform(features)
    features = features.reshape(-1, num_channels, window_size)

    return features
