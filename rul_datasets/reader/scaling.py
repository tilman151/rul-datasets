import pickle
from typing import List, Optional

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore


def fit_scaler(
    features: List[np.ndarray], scaler: Optional[BaseEstimator] = None
) -> BaseEstimator:
    if scaler is None:
        scaler = scalers.StandardScaler()
    for run in features:
        run = run.reshape(-1, run.shape[-1])
        scaler.partial_fit(run)

    return scaler


def save_scaler(scaler: BaseEstimator, save_path: str) -> None:
    with open(save_path, mode="wb") as f:
        pickle.dump(scaler, f)


def load_scaler(save_path: str) -> BaseEstimator:
    with open(save_path, mode="rb") as f:
        scaler = pickle.load(f)

    return scaler


def scale_features(
    features: List[np.ndarray], scaler: BaseEstimator
) -> List[np.ndarray]:
    for i, run in enumerate(features):
        _check_channels(run, scaler)
        if len(run.shape) == 3:
            features[i] = _scale_windowed_features(run, scaler)
        else:
            features[i] = scaler.transform(run)

    return features


def _check_channels(run: np.ndarray, scaler: BaseEstimator) -> None:
    if not run.shape[1] == scaler.n_features_in_:
        raise ValueError(
            f"The scaler was fit on {scaler.n_features_in_} "
            f"channels but the features have {run.shape[1]} channels."
        )


def _scale_windowed_features(features: np.ndarray, scaler: BaseEstimator) -> np.ndarray:
    num_channels = features.shape[1]
    window_size = features.shape[2]
    features = features.reshape(-1, num_channels)
    features = scaler.transform(features)
    features = features.reshape(-1, num_channels, window_size)

    return features
