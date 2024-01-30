import os
import random
import sys

import numpy as np
import numpy.testing as npt
import pytest
import torch

from rul_datasets import utils


def dummy_file_name_to_timestep(file_path):
    file_name = os.path.basename(file_path)
    time_step = int(file_name.replace(".csv", ""))

    return time_step


def ordered_file_paths():
    file_paths = {}
    time_steps = {}
    for i in range(2):
        run_steps = list(range(1, random.randint(10, 20)))
        time_steps[i] = run_steps
        file_paths[i] = [f"foo/bar/{i}.csv" for i in run_steps]

    return file_paths, time_steps


def shuffled_file_paths():
    file_paths = {}
    time_steps = {}
    for i in range(2):
        run_steps = list(range(1, random.randint(10, 20)))
        random.shuffle(run_steps)
        time_steps[i] = run_steps
        file_paths[i] = [f"foo/bar/{i}.csv" for i in run_steps]

    return file_paths, time_steps


@pytest.mark.parametrize("file_path_func", [ordered_file_paths, shuffled_file_paths])
def test_get_targets_from_file_paths(file_path_func):
    file_paths, time_steps = file_path_func()
    targets = utils.get_targets_from_file_paths(file_paths, dummy_file_name_to_timestep)

    for run_targets, run_steps in zip(targets.values(), time_steps.values()):
        sorted_idx = np.argsort(run_targets)
        sorted_targets = run_targets[sorted_idx]
        sorted_steps = np.array(run_steps)[sorted_idx[::-1]]  # reverse sorted
        assert np.all(sorted_targets == sorted_steps)


@pytest.mark.parametrize(
    "window_size", [1, 5, 10, pytest.param(11, marks=pytest.mark.xfail)]
)
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_extract_windows(window_size, dilation):
    inputs = np.arange(30)
    windows = utils.extract_windows(inputs, window_size, dilation)

    expected_num_windows = len(inputs) - (window_size - 1) * dilation
    for i in range(expected_num_windows):
        expected_window = inputs[i : (i + window_size * dilation) : dilation]
        npt.assert_equal(windows[i], expected_window)


def test_extract_windows_memmap_identical(tmp_path):
    inputs = np.random.randn(100, 16)

    windows = utils.extract_windows(inputs, 10, 1, memmap=False)
    windows_memmap = utils.extract_windows(inputs, 10, 1, memmap=True)

    npt.assert_almost_equal(windows, windows_memmap)


def test_extract_windows_memmap_auto_deletes():
    tmp_file_name = None

    def _extract_tmp_file_name(event_name, args):
        """Grabs temporary file name from 'tempfile.mkstemp' audit event."""
        if event_name == "tempfile.mkstemp":
            nonlocal tmp_file_name
            tmp_file_name = args[0]

    sys.addaudithook(_extract_tmp_file_name)
    inputs = np.random.randn(100, 16)

    windows = utils.extract_windows(inputs, 10, 1, memmap=True)

    windows.max()  # check if memmap is accessible
    del windows
    assert not os.path.exists(tmp_file_name)  # check if memmap is deleted


@pytest.mark.parametrize("num_targets", [0, 1, 2])
@pytest.mark.parametrize("num_batch_dims", [0, 1, 2, 3])
def test_to_tensor(num_targets, num_batch_dims):
    batch_dims = (10,) * num_batch_dims
    features = [np.random.randn(*batch_dims, 100, 2)]
    targets = [[np.arange(10)]] * num_targets

    tensor_features, *tensor_targets = utils.to_tensor(features, *targets)

    assert isinstance(tensor_features, list)
    assert tensor_features[0].shape == (*batch_dims, 2, 100)
    assert tensor_features[0].dtype == torch.float32

    assert len(tensor_targets) == num_targets
    for tensor_t in tensor_targets:
        assert isinstance(tensor_t, list)
        assert tensor_t[0].dtype == torch.float32
