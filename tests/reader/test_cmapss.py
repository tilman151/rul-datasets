from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest

from rul_datasets import reader


@pytest.fixture(scope="module", autouse=True)
def prepare_cmapss():
    for fd in range(1, 5):
        reader.CmapssReader(fd).prepare_data()


@pytest.mark.needs_data
class TestCMAPSSLoader:
    NUM_CHANNELS = len(reader.CmapssReader._DEFAULT_CHANNELS)

    @pytest.mark.parametrize("fd", [1, 2, 3, 4])
    @pytest.mark.parametrize("window_size", [30, 15])
    def test_run_shape_and_dtype(self, fd, window_size):
        rul_loader = reader.CmapssReader(fd, window_size=window_size)
        for split in ["dev", "val", "test"]:
            self._check_split(rul_loader, split, window_size)

    def _check_split(self, rul_loader, split, window_size):
        features, targets = rul_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        assert win == run.shape[1]
        assert self.NUM_CHANNELS == run.shape[2]
        assert len(run) == len(run_target)
        assert np.float == run.dtype
        assert np.float == run_target.dtype

    @pytest.mark.parametrize(
        ("fd", "window_size"), [(1, 30), (2, 20), (3, 30), (4, 15)]
    )
    def test_default_window_size(self, fd, window_size):
        rul_loader = reader.CmapssReader(fd)
        assert window_size == rul_loader.window_size

    def test_default_feature_select(self):
        rul_loader = reader.CmapssReader(1)
        assert rul_loader._DEFAULT_CHANNELS == rul_loader.feature_select

    def test_feature_select(self):
        dataset = reader.CmapssReader(1, feature_select=[4, 9, 10, 13, 14, 15, 22])
        dataset.prepare_data()  # fit new scaler for these features
        for split in ["dev", "val", "test"]:
            features, _ = dataset.load_split(split)
            for run in features:
                assert 7 == run.shape[2]

    def test_prepare_data_not_called_for_feature_select(self):
        dataset = reader.CmapssReader(1, feature_select=[4])
        with pytest.raises(RuntimeError):
            dataset.load_split("dev")

    @pytest.mark.parametrize("fd", [1, 2, 3, 4])
    @pytest.mark.parametrize("ops_aware", [True, False])
    def test_normalization_min_max(self, fd, ops_aware):
        full_dataset = reader.CmapssReader(
            fd, operation_condition_aware_scaling=ops_aware
        )
        full_dataset.prepare_data()
        full_dev, full_dev_targets = full_dataset.load_split("dev")

        npt.assert_almost_equal(max(np.max(r) for r in full_dev), 1.0)
        npt.assert_almost_equal(min(np.min(r) for r in full_dev), -1.0)

        trunc_dataset = reader.CmapssReader(fd, percent_fail_runs=0.8)
        trunc_dev, _ = trunc_dataset.load_split("dev")
        assert np.round(max(np.max(r).item() for r in trunc_dev), decimals=7) <= 1.0
        assert np.round(min(np.min(r).item() for r in trunc_dev), decimals=7) >= -1.0

        trunc_dataset = reader.CmapssReader(fd, percent_broken=0.2)
        trunc_dev, _ = trunc_dataset.load_split("dev")
        assert np.round(max(np.max(r).item() for r in trunc_dev), decimals=7) <= 1.0
        assert np.round(min(np.min(r).item() for r in trunc_dev), decimals=7) >= -1.0

    @pytest.mark.parametrize("fd", [1, 2, 3, 4])
    @pytest.mark.parametrize("split", ["dev", "val", "test"])
    def test_operation_condition_boundaries_cover_all_samples(self, fd, split):
        dataset = reader.CmapssReader(fd, operation_condition_aware_scaling=True)
        dataset.prepare_data()
        dataset.load_split(split)

    def test_crop_data_pads_correctly(self):
        """Check test samples smaller than window_size are zero-padded on the left."""
        dataset = reader.CmapssReader(1, window_size=30)
        inputs = np.ones((15, 14))
        targets = np.arange(20, 5, -1)

        output_features, output_targets = dataset._crop_data([inputs], [targets])

        assert np.all(output_features[0][0, :15] == 0.0)
        npt.assert_equal(output_features[0][0, 15:], inputs)
        assert output_targets[0][0] == targets[-1]

    @pytest.mark.parametrize(
        ["split", "generated"], [("dev", True), ("val", True), ("test", False)]
    )
    @pytest.mark.parametrize(
        ["alias", "windowed"], [("dev", True), ("val", True), ("test", False)]
    )
    def test_alias(self, split, generated, alias, windowed):
        dataset = reader.CmapssReader(1)
        dataset._generate_targets = mock.Mock(wraps=dataset._generate_targets)
        dataset._load_targets = mock.Mock(wraps=dataset._load_targets)
        dataset._window_data = mock.Mock(wraps=dataset._window_data)
        dataset._crop_data = mock.Mock(wraps=dataset._crop_data)

        dataset.load_complete_split(split, alias)

        if generated:
            dataset._generate_targets.assert_called()
            dataset._load_targets.assert_not_called()
        else:
            dataset._generate_targets.assert_not_called()
            dataset._load_targets.assert_called()

        if windowed:
            dataset._window_data.assert_called()
            dataset._crop_data.assert_not_called()
        else:
            dataset._window_data.assert_not_called()
            dataset._crop_data.assert_called()
