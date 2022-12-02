import numpy as np
import pytest
import numpy.testing as npt

from rul_datasets import reader


@pytest.mark.parametrize("fd", [1, 2])
def test_run_shape_and_dtype(fd):
    rul_loader = reader.DummyReader(fd)
    for split in ["dev", "val", "test"]:
        _check_split(rul_loader, split)


def _check_split(rul_loader, split):
    features, targets = rul_loader.load_split(split)
    for run, run_target in zip(features, targets):
        _assert_run_correct(run, run_target, 10)


def _assert_run_correct(run, run_target, win):
    assert win == run.shape[1]
    assert 1 == run.shape[2]
    assert len(run) == len(run_target)
    assert np.float == run.dtype
    assert np.float == run_target.dtype


def test_test_split_has_only_single_windows():
    rul_reader = reader.DummyReader(1)

    features, targets = rul_reader.load_split("test")

    assert all(len(f) == 1 for f in features)
    assert all(len(t) == 1 for t in targets)
    assert all(t > 1 for t in targets)


@pytest.mark.parametrize("fd", [1, 2])
def test_normalization_min_max(fd):
    full_dataset = reader.DummyReader(fd)
    full_dev, full_dev_targets = full_dataset.load_split("dev")

    npt.assert_almost_equal(max(np.max(r) for r in full_dev), 1.0)
    npt.assert_almost_equal(min(np.min(r) for r in full_dev), -1.0)

    trunc_dataset = reader.DummyReader(fd, percent_fail_runs=0.8)
    trunc_dev, _ = trunc_dataset.load_split("dev")
    assert np.round(max(np.max(r).item() for r in trunc_dev), decimals=7) <= 1.0
    assert np.round(min(np.min(r).item() for r in trunc_dev), decimals=7) >= -1.0

    trunc_dataset = reader.DummyReader(fd, percent_broken=0.2)
    trunc_dev, _ = trunc_dataset.load_split("dev")
    assert np.round(max(np.max(r).item() for r in trunc_dev), decimals=7) <= 1.0
    assert np.round(min(np.min(r).item() for r in trunc_dev), decimals=7) >= -1.0
