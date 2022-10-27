import os.path
from typing import Tuple

import numpy as np
import numpy.testing as npt
import pytest

import sklearn.preprocessing as scalers
from sklearn.utils.validation import check_is_fitted

from rul_datasets.reader import scaling


@pytest.fixture
def fitted_scaler():
    scaler = scalers.StandardScaler()
    scaler.fit(np.random.randn(10000, 5) * 2 + 1)

    return scaler


@pytest.mark.parametrize("feature_shape", [(1000, 5), (1000, 2, 5)])
def test_fit_scaler(feature_shape: Tuple[int]):
    features = [np.random.randn(*feature_shape) * 2 + 1 for _ in range(10)]
    scaler = scaling.fit_scaler(features)

    npt.assert_almost_equal(scaler.mean_, 1.0, decimal=1)
    npt.assert_almost_equal(scaler.var_, 4.0, decimal=1)


def test_fit_scaler_custom_scaler():
    features = [np.random.randn(10, 5)]
    scaler = scaling.fit_scaler(features, scalers.MinMaxScaler())

    assert isinstance(scaler, scalers.MinMaxScaler)


def test_save_load_scaler(tmp_path, fitted_scaler):
    save_path = os.path.join(tmp_path, "scaler.pkl")
    scaling.save_scaler(fitted_scaler, save_path)
    loaded_scaler = scaling.load_scaler(save_path)

    check_is_fitted(loaded_scaler)
    assert loaded_scaler.n_features_in_ == fitted_scaler.n_features_in_
    npt.assert_almost_equal(loaded_scaler.mean_, fitted_scaler.mean_)
    npt.assert_almost_equal(loaded_scaler.var_, fitted_scaler.var_)


@pytest.mark.parametrize(
    "feature_shape",
    [(10000, 5), (10000, 2, 5), pytest.param((1000, 4), marks=pytest.mark.xfail)],
)
def test_scale_features(feature_shape, fitted_scaler):
    mean = fitted_scaler.mean_
    std = np.sqrt(fitted_scaler.var_)
    features = [np.random.randn(*feature_shape) * std + mean]
    scaled_features = scaling.scale_features(features, fitted_scaler)

    (scaled_features,) = scaled_features
    assert scaled_features.shape == feature_shape
    npt.assert_almost_equal(np.mean(scaled_features), 0.0, decimal=2)
    npt.assert_almost_equal(np.std(scaled_features), 1.0, decimal=2)
