import os

import numpy as np
import pytest
from numpy import testing as npt

from rul_datasets import loader


def femto_preperator_class(fd):
    return loader.FemtoPreparator(fd, loader.FemtoLoader._FEMTO_ROOT)


def xjtu_sy_preparator_class(fd):
    return loader.XjtuSyPreparator(fd, loader.XjtuSyLoader._XJTU_SY_ROOT)


FEMTO_NUM_SAMPLES = {
    1: {"dev": 3674, "val": 2375, "test": 7245},
    2: {"dev": 1708, "val": 1955, "test": 3358},
    3: {"dev": 515, "val": 1637, "test": 352},
}
XJTU_SY_NUM_SAMPLES = {
    1: {"dev": 284, "val": 158, "test": 174},
    2: {"dev": 652, "val": 533, "test": 381},
    3: {"dev": 5034, "val": 371, "test": 1629},
}


@pytest.mark.needs_data
@pytest.mark.parametrize(
    ["preperator_class", "num_samples"],
    [
        (femto_preperator_class, FEMTO_NUM_SAMPLES),
        (xjtu_sy_preparator_class, XJTU_SY_NUM_SAMPLES),
    ],
)
class TestPreperatorsShared:
    @pytest.mark.skip
    def test_loading_one_file(self, preperator_class, num_samples):
        preparator = preperator_class(1)
        csv_paths = preparator._get_csv_file_paths("dev")
        features = preparator._load_feature_file(csv_paths[0][0])
        assert isinstance(features, np.ndarray)
        assert features.shape == (preparator.DEFAULT_WINDOW_SIZE, 2)

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_converted_files(self, preperator_class, num_samples, fd, split):
        preparator = preperator_class(fd)
        features, targets = preparator.load_runs(split)
        expected_num_time_steps = num_samples[fd][split]
        actual_num_target_steps = sum(len(f) for f in features)
        assert actual_num_target_steps == expected_num_time_steps
        actual_num_target_steps = sum(len(t) for t in targets)
        assert actual_num_target_steps == expected_num_time_steps

    @pytest.mark.parametrize("fd", [1, 2, 3])
    def test_scaler(self, preperator_class, num_samples, fd):
        preparator = preperator_class(fd)
        scaler = preparator.load_scaler()
        expected_samples = num_samples[fd]["dev"] * preparator.DEFAULT_WINDOW_SIZE
        assert 2 == scaler.n_features_in_
        assert scaler.n_samples_seen_ == expected_samples

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_runs_are_ordered(self, preperator_class, num_samples, fd, split):
        preperator = preperator_class(fd)
        features, targets = preperator.load_runs(split)
        for targ in targets:
            npt.assert_equal(targ, np.sort(targ)[::-1])


FEMTO_NUM_FILES = {
    1: {"dev": 3674, "val": 10973, "test": 9047},
    2: {"dev": 1708, "val": 5948, "test": 4560},
    3: {"dev": 2152, "val": 434, "test": 352},
}


@pytest.mark.parametrize("fd", [1, 2, 3])
@pytest.mark.parametrize("split", ["dev", "val", "test"])
def test_file_discovery_femto(fd, split):
    preparator = loader.FemtoPreparator(fd, loader.FemtoLoader._FEMTO_ROOT)
    csv_paths = preparator._get_csv_file_paths(split)
    expected_num_files = FEMTO_NUM_FILES[fd][split]
    actual_num_files = len(sum(csv_paths.values(), []))
    assert actual_num_files == expected_num_files
