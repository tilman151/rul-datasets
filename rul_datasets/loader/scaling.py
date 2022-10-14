import pickle
from typing import List

import numpy as np
from sklearn import preprocessing as scalers  # type: ignore


def fit_scaler(features: List[np.ndarray]) -> scalers.StandardScaler:
    scaler = scalers.StandardScaler()
    for run in features:
        run = run.reshape(-1, run.shape[-1])
        scaler.partial_fit(run)

    return scaler


def save_scaler(scaler: scalers.StandardScaler, save_path: str) -> None:
    with open(save_path, mode="wb") as f:
        pickle.dump(scaler, f)


def load_scaler(save_path: str) -> scalers.StandardScaler:
    with open(save_path, mode="rb") as f:
        scaler = pickle.load(f)

    return scaler


def scale_features(
    features: List[np.ndarray], scaler: scalers.StandardScaler
) -> List[np.ndarray]:
    num_channels = scaler.n_features_in_
    for i, run in enumerate(features):
        window_size = run.shape[1]
        run = run.reshape(-1, num_channels)
        run = scaler.transform(run)
        features[i] = run.reshape(-1, window_size, num_channels)

    return features
