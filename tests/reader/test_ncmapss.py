import numpy as np
import pytest

from rul_datasets.reader.ncmapss import NCmapssReader


@pytest.fixture()
def prepared_ncmapss():
    for fd in range(1, 8):
        NCmapssReader(fd).prepare_data()


@pytest.mark.parametrize("fd", list(range(1, 8)))
def test_dataset_name(fd):
    assert NCmapssReader(fd).dataset_name == "ncmapss"


@pytest.mark.parametrize("fd", list(range(1, 8)))
def test_fds(fd):
    assert NCmapssReader(fd).fds == list(range(1, 8))


def test_additional_hparams():
    femto = NCmapssReader(1)
    assert femto.hparams["feature_select"] == list(range(32))
    assert femto.hparams["padding_value"] == 0.0
    assert femto.hparams["run_split_dist"] == {
        "dev": [0, 1, 2, 3, 4],
        "val": [5],
        "test": [6, 7, 8, 9],
    }


@pytest.mark.needs_data
@pytest.mark.parametrize("should_run", [True, False])
def test_prepare_data(should_run, mocker):
    mocker.patch("os.path.exists", return_value=not should_run)
    mock_save_scaler = mocker.patch("rul_datasets.reader.ncmapss.scaling.save_scaler")

    NCmapssReader(1).prepare_data()
    if should_run:
        mock_save_scaler.assert_called_once()
    else:
        mock_save_scaler.assert_not_called()


@pytest.mark.needs_data
@pytest.mark.parametrize("fd", list(range(1, 8)))
@pytest.mark.parametrize("split", ["dev", "val", "test"])
def test_load_complete_split(fd, split, prepared_ncmapss):
    reader = NCmapssReader(fd)
    features, targets = reader.load_complete_split(split, split)

    assert len(features) == len(targets)
    for feat, targ in zip(features, targets):
        assert feat.ndim == 3
        assert targ.ndim == 1


@pytest.mark.needs_data
@pytest.mark.parametrize("fd", list(range(1, 8)))
def test_scaling(fd, prepared_ncmapss):
    reader = NCmapssReader(fd)
    features, targets = reader.load_complete_split("dev", "dev")

    for feat, targ in zip(features, targets):
        # rounded due to float tolerance
        assert np.all(np.max(feat, axis=(0, 1)).round(6) <= 1)
        assert np.all(np.min(feat, axis=(0, 1)).round(6) >= 0)


@pytest.mark.needs_data
def test__split_by_unit(prepared_ncmapss):
    reader = NCmapssReader(1)
    features, targets, auxiliary = reader._load_raw_data()
    features, targets, auxiliary = reader._split_by_unit(features, targets, auxiliary)

    for i in range(len(auxiliary)):
        assert len(features[i]) == len(targets[i])
        assert np.unique(auxiliary[i][:, 0]).size == 1  # only one unit id present


@pytest.mark.needs_data
@pytest.mark.parametrize("window_size", [10, 100])
def test_padding_and_window_size(window_size, prepared_ncmapss):
    default_reader = NCmapssReader(1, padding_value=-1)
    reader = NCmapssReader(1, window_size=window_size)

    default_features, _ = default_reader.load_complete_split("dev", "dev")
    features, _ = reader.load_complete_split("dev", "dev")

    for feat, default_feat in zip(features, default_features):
        # extract first window of window_size after padding
        first_data_idx = np.argmax(np.all(default_feat != -1, axis=2), axis=1)
        manual_windows = [
            f[idx : idx + window_size] for f, idx in zip(default_feat, first_data_idx)
        ]
        manual_windows = np.stack(manual_windows, axis=0)
        np.testing.assert_almost_equal(manual_windows, feat)


@pytest.mark.needs_data
@pytest.mark.parametrize("resolution_seconds", [10, 60])
def test_resolution_seconds(resolution_seconds):
    unreduced_reader = NCmapssReader(1, window_size=2 * resolution_seconds)
    reader = NCmapssReader(1, window_size=10, resolution_seconds=resolution_seconds)

    unreduced_features, _ = unreduced_reader.load_complete_split("dev", "dev")
    features, _ = reader.load_complete_split("dev", "dev")

    assert reader.window_size == 10
    for unreduced_feat, feat in zip(unreduced_features, features):
        # check if mean of second half of unreduced feature equals second reduced
        # feature to see if reshaping while downsampling works as expected
        assert feat.shape[1] == reader.window_size
        manually_reduced = unreduced_feat[:, resolution_seconds:].mean(axis=1)
        np.testing.assert_almost_equal(manually_reduced, feat[:, 1])


def test_window_size_auto_adjust():
    with pytest.warns(UserWarning):
        reader = NCmapssReader(1, resolution_seconds=10)

    assert reader.window_size == reader.default_window_size(reader.fd) // 10


@pytest.mark.needs_data
def test_feature_select(prepared_ncmapss):
    reader = NCmapssReader(1, feature_select=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    features, _ = reader.load_complete_split("dev", "dev")

    assert features[0].shape[2] == 10
