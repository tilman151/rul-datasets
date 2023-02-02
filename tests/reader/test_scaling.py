import os.path
from typing import Tuple
from unittest import mock

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


@pytest.fixture()
def conditioned_inputs():
    cond_idx = range(1, 4)
    features = [np.random.choice([-i, i], (10, 5)) for i in cond_idx]
    conditions = [np.ones(10) * i for i in cond_idx]
    boundaries = [(i, i + 0.9) for i in cond_idx]

    return features, conditions, boundaries


@pytest.fixture()
def fitted_conditioned_scaler(conditioned_inputs):
    features, conditions, boundaries = conditioned_inputs
    features = np.concatenate(features)
    conditions = np.concatenate(conditions)
    scaler = scaling.OperationConditionAwareScaler(scalers.MinMaxScaler(), boundaries)
    scaler = scaler.partial_fit(features, conditions)

    return scaler


class TestOperationConditionAwareScaler:
    @pytest.mark.parametrize(
        "boundaries",
        [
            [(0, 1), (2, 3), [4, 5]],
            pytest.param([(0, 1), (1, 2)], marks=pytest.mark.xfail(strict=True)),
        ],
    )
    def test_boundary_mutually_exclusive_check(self, boundaries):
        scaling.OperationConditionAwareScaler(scalers.MinMaxScaler(), boundaries)

    def test_partial_fit(self, conditioned_inputs):
        features, conditions, boundaries = conditioned_inputs
        scaler = scaling.OperationConditionAwareScaler(
            scalers.MinMaxScaler(), boundaries
        )

        for f, c in zip(features, conditions):
            scaler.partial_fit(f, c)

        assert len(scaler.base_scalers) == len(boundaries)
        for i in range(len(boundaries)):
            assert np.all(scaler.base_scalers[i].data_min_ == features[i].min(0))
            assert np.all(scaler.base_scalers[i].data_max_ == features[i].max(0))

    def test_partial_fit_unknown_condition(self, conditioned_inputs):
        features, conditions, boundaries = conditioned_inputs
        conditions[0][5] = 99  # condition not covered by boundaries
        scaler = scaling.OperationConditionAwareScaler(
            scalers.MinMaxScaler(), boundaries
        )

        with pytest.raises(RuntimeError):
            for f, c in zip(features, conditions):
                scaler.partial_fit(f, c)

    def test_transform(self, fitted_conditioned_scaler):
        num_conditions = len(fitted_conditioned_scaler.boundaries)
        conditions = np.random.choice(range(1, num_conditions + 1), 100)
        features = np.ones((100, 5)) * conditions[:, None]

        scaled = fitted_conditioned_scaler.transform(features, conditions)
        npt.assert_almost_equal(scaled, 1)

    def test_transform_unknown_condition(self, fitted_conditioned_scaler):
        num_conditions = len(fitted_conditioned_scaler.boundaries)
        conditions = np.random.choice(range(1, num_conditions + 1), 100)
        conditions[5] = 99  # condition not covered by boundaries
        features = np.random.randn(100, 5)

        with pytest.raises(RuntimeError):
            fitted_conditioned_scaler.partial_fit(features, conditions)


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


def test_fit_condition_aware_scaler():
    features = [np.random.uniform(0, 3, (10, 5)), np.random.uniform(0, 2, (10, 5))]
    operation_conditions = [np.zeros(10), np.ones(10)]
    scaler = mock.MagicMock(scaling.OperationConditionAwareScaler)

    out_scaler = scaling.fit_scaler(features, scaler, operation_conditions)

    assert scaler is out_scaler
    scaler.partial_fit.assert_has_calls(
        [mock.call(f, c) for f, c in zip(features, operation_conditions)]
    )


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
    [
        (10000, 5),
        (5000, 2, 5),
        pytest.param((25000, 2), marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_scale_features(feature_shape, fitted_scaler):
    mean = fitted_scaler.mean_
    std = np.sqrt(fitted_scaler.var_)
    features = [np.reshape(np.random.randn(10000, 5) * std + mean, feature_shape)]
    scaled_features = scaling.scale_features(features, fitted_scaler)

    (scaled_features,) = scaled_features
    assert scaled_features.shape == feature_shape
    npt.assert_almost_equal(np.mean(scaled_features), 0.0, decimal=2)
    npt.assert_almost_equal(np.std(scaled_features), 1.0, decimal=2)


@pytest.mark.parametrize(
    "feature_shape",
    [
        (10000, 5),
        pytest.param((5000, 2, 5), marks=pytest.mark.xfail(strict=True)),
        pytest.param((5000, 4), marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_scale_features_condition_aware(feature_shape, fitted_conditioned_scaler):
    fitted_conditioned_scaler.transform = mock.MagicMock(name="transform")
    features = [np.random.randn(*feature_shape)]
    operation_condition = [np.ones(feature_shape[0])]

    scaled = scaling.scale_features(
        features, fitted_conditioned_scaler, operation_condition
    )

    fitted_conditioned_scaler.transform.assert_called_with(
        features[0], operation_condition[0]
    )
    assert isinstance(scaled, list)
    assert scaled[0] is fitted_conditioned_scaler.transform()
