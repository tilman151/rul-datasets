import os

import numpy as np
import pytest
from numpy import testing as npt

from rul_datasets import reader


@pytest.fixture(scope="module", autouse=True)
def prepare_femto():
    for fd in range(1, 4):
        reader.FemtoReader(fd).prepare_data()


@pytest.fixture(scope="module", autouse=True)
def prepare_xjtu_sy():
    for fd in range(1, 4):
        reader.XjtuSyReader(fd).prepare_data()


def femto_preperator_class(fd, run_split_dist=None):
    return reader.FemtoPreparator(fd, reader.FemtoReader._FEMTO_ROOT, run_split_dist)


def xjtu_sy_preparator_class(fd, run_split_dist=None):
    return reader.XjtuSyPreparator(
        fd, reader.XjtuSyReader._XJTU_SY_ROOT, run_split_dist
    )


FEMTO_NUM_SAMPLES = {
    1: {"dev": 3674, "val": 2375, "test": 8598},
    2: {"dev": 1708, "val": 1955, "test": 3993},
    3: {"dev": 515, "val": 1637, "test": 434},
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

    def test_scaler_refitted_for_custom_split(self, preperator_class, num_samples):
        custom_split = {"dev": [1], "val": [2], "test": [3]}
        default_preperator = preperator_class(fd=1)
        custom_preperator = preperator_class(fd=1, run_split_dist=custom_split)
        custom_preperator.prepare_split("dev")

        default_scaler = default_preperator.load_scaler()
        custom_scaler = custom_preperator.load_scaler()
        assert np.all(default_scaler.mean_ != custom_scaler.mean_)
        assert np.all(default_scaler.var_ != custom_scaler.var_)

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_runs_are_ordered(self, preperator_class, num_samples, fd, split):
        preperator = preperator_class(fd)
        features, targets = preperator.load_runs(split)
        for targ in targets:
            npt.assert_equal(targ, np.sort(targ)[::-1])


FEMTO_NUM_FILES = {
    1: {"dev": 3674, "val": 10973, "test": 10973},
    2: {"dev": 1708, "val": 5948, "test": 5948},
    3: {"dev": 2152, "val": 434, "test": 434},
}


@pytest.mark.needs_data
@pytest.mark.parametrize("fd", [1, 2, 3])
@pytest.mark.parametrize("split", ["dev", "val", "test"])
def test_file_discovery_femto(fd, split):
    preparator = reader.FemtoPreparator(fd, reader.FemtoReader._FEMTO_ROOT)
    csv_paths = preparator._get_csv_file_paths(split)
    expected_num_files = FEMTO_NUM_FILES[fd][split]
    actual_num_files = len(sum(csv_paths.values(), []))
    assert actual_num_files == expected_num_files
