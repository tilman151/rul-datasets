import os.path
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest

from rul_datasets.reader import saving


@pytest.mark.parametrize("file_name", ["run", "run.npy", "run.foo"])
def test_save(tmp_path, file_name):
    features = np.empty((10, 2, 5))
    targets = np.empty((10,))
    save_path = os.path.join(tmp_path, file_name)
    saving.save(save_path, features, targets)

    exp_save_path = save_path.replace(".npy", "")
    loaded_features = np.load(exp_save_path + "_features.npy")
    loaded_targets = np.load(exp_save_path + "_targets.npy")
    npt.assert_equal(loaded_features, features)
    npt.assert_equal(loaded_targets, targets)


@pytest.mark.parametrize("file_name", ["run", "run.npy", "run.foo"])
def test_save_multi_target(tmp_path, file_name):
    features = np.empty((10, 2, 5))
    targets_0 = np.empty((10,))
    targets_1 = np.empty((10,))
    save_path = os.path.join(tmp_path, file_name)
    saving.save(save_path, features, targets_0, targets_1)

    exp_save_path = save_path.replace(".npy", "")
    loaded_features = np.load(exp_save_path + "_features.npy")
    loaded_targets_0 = np.load(exp_save_path + "_targets_0.npy")
    loaded_targets_1 = np.load(exp_save_path + "_targets_1.npy")
    npt.assert_equal(loaded_features, features)
    npt.assert_equal(loaded_targets_0, targets_0)
    npt.assert_equal(loaded_targets_1, targets_1)


@pytest.mark.parametrize("file_name", ["run", "run.npy", "run.foo"])
def test_load(tmp_path, file_name):
    features = np.empty((10, 2, 5))
    targets = np.empty((10,))
    exp_file_name = file_name.replace(".npy", "")
    np.save(os.path.join(tmp_path, f"{exp_file_name}_features.npy"), features)
    np.save(os.path.join(tmp_path, f"{exp_file_name}_targets.npy"), targets)

    save_path = os.path.join(tmp_path, file_name)
    loaded_features, loaded_targets = saving.load(save_path)
    npt.assert_equal(loaded_features, features)
    npt.assert_equal(loaded_targets, targets)


@pytest.mark.parametrize("file_name", ["run", "run.npy", "run.foo"])
def test_load_multi_target(tmp_path, file_name):
    features = np.empty((10, 2, 5))
    targets_0 = np.empty((10,))
    targets_1 = np.empty((10,))
    exp_file_name = file_name.replace(".npy", "")
    np.save(os.path.join(tmp_path, f"{exp_file_name}_features.npy"), features)
    np.save(os.path.join(tmp_path, f"{exp_file_name}_targets_0.npy"), targets_0)
    np.save(os.path.join(tmp_path, f"{exp_file_name}_targets_1.npy"), targets_1)

    save_path = os.path.join(tmp_path, file_name)
    loaded_features, loaded_targets_0, loaded_targets_1 = saving.load(save_path)
    npt.assert_equal(loaded_features, features)
    npt.assert_equal(loaded_targets_0, targets_0)
    npt.assert_equal(loaded_targets_1, targets_1)


@mock.patch("rul_datasets.reader.saving.load", return_value=(None, None))
@pytest.mark.parametrize("file_names", [["run1", "run2"], []])
def test_load_multiple(mock_load, file_names):
    features, targets = saving.load_multiple(file_names)

    mock_load.assert_has_calls([mock.call(name, False) for name in file_names])
    assert len(features) == len(file_names)
    assert len(targets) == len(file_names)


@pytest.mark.parametrize("file_name", ["run", "run.npy"])
def test_exists(tmp_path, file_name):
    save_path = os.path.join(tmp_path, file_name)
    assert not saving.exists(save_path)

    Path(os.path.join(tmp_path, "run_features.npy")).touch()
    assert not saving.exists(save_path)

    Path(os.path.join(tmp_path, "run_targets.npy")).touch()
    assert saving.exists(save_path)


@pytest.mark.parametrize("file_name", ["run", "run.npy"])
def test_exists_multi_target(tmp_path, file_name):
    save_path = os.path.join(tmp_path, file_name)
    assert not saving.exists(save_path)

    Path(os.path.join(tmp_path, "run_features.npy")).touch()
    assert not saving.exists(save_path)

    Path(os.path.join(tmp_path, "run_targets_0.npy")).touch()
    assert saving.exists(save_path)


@pytest.mark.parametrize("columns", [[0], [0, 1]])
@pytest.mark.parametrize(
    "file_name", ["raw_features.csv", "raw_features_corrupted.csv"]
)
def test_load_raw_features(file_name, columns):
    script_path = os.path.dirname(__file__)
    file_path = os.path.join(script_path, "..", "assets", file_name)
    window_size = 2
    features = saving.load_raw(
        {0: [file_path]}, window_size, columns=columns, skip_rows=1
    )

    assert len(features) == 1
    run = features[0]
    assert run.shape == (1, window_size, len(columns))
