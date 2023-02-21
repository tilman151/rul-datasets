import numpy as np
import pytest
import torch
from numpy import testing as npt

from rul_datasets import reader
from rul_datasets.reader.xjtu_sy import _download_xjtu_sy


@pytest.fixture(scope="module", autouse=True)
def prepare_xjtu_sy():
    for fd in range(1, 4):
        reader.XjtuSyReader(fd).prepare_data()


@pytest.mark.needs_data
class TestXjtuSyLoader:
    NUM_CHANNELS = 2

    def test_default_window_size(self):
        xjtu = reader.XjtuSyReader(1)
        assert xjtu.window_size == reader.XjtuSyPreparator.DEFAULT_WINDOW_SIZE

    @pytest.mark.parametrize("fd", [1, 2, 3])
    def test_standardization(self, fd):
        full_dataset = reader.XjtuSyReader(fd)
        full_train, full_train_targets = full_dataset.load_split("dev")

        npt.assert_almost_equal(0.0, np.mean(np.concatenate(full_train)), decimal=4)
        npt.assert_almost_equal(1.0, np.std(np.concatenate(full_train)), decimal=4)

        truncated_dataset = reader.XjtuSyReader(fd, percent_fail_runs=0.8)
        trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
        npt.assert_almost_equal(0.0, np.mean(np.concatenate(trunc_train)), decimal=1)

        # percent_broken is supposed to change the std but not the mean
        truncated_dataset = reader.XjtuSyReader(fd, percent_broken=0.2)
        trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
        npt.assert_almost_equal(0.0, np.mean(np.concatenate(trunc_train)), decimal=1)

    @pytest.mark.parametrize("window_size", [1500, 100])
    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("split", ["dev", "val", "test"])
    def test_run_shape_and_dtype(self, window_size, fd, split):
        rul_loader = reader.XjtuSyReader(fd, window_size=window_size)
        features, targets = rul_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        assert win == run.shape[1]
        assert self.NUM_CHANNELS == run.shape[2]
        assert len(run) == len(run_target)
        assert np.float64 == run.dtype
        assert np.float64 == run_target.dtype

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize(
        ["split", "exp_length"], [("dev", 2), ("val", 1), ("test", 2)]
    )
    def test_run_split_dist(self, fd, split, exp_length):
        rul_loader = reader.XjtuSyReader(fd)
        features, targets = rul_loader.load_split(split)
        assert len(features) == len(targets) == exp_length

    def test_first_time_to_predict(self):
        fttp = [10, 20, 30, 40, 50]
        dataset = reader.XjtuSyReader(1, first_time_to_predict=fttp)
        targets = (
            dataset.load_split("dev")[1]
            + dataset.load_split("val")[1]
            + dataset.load_split("test")[1]
        )
        for target, first_time in zip(targets, fttp):
            max_rul = len(target) - first_time
            assert np.all(target[:first_time] == max_rul)

    def test_norm_rul_with_max_rul(self):
        dataset = reader.XjtuSyReader(1, max_rul=50, norm_rul=True)
        for split in ["dev", "val", "test"]:
            _, targets = dataset.load_split(split)
            for target in targets:
                assert np.max(target) == 1

    def test_norm_rul_with_fttp(self):
        fttp = [10, 20, 30, 40, 50]
        dataset = reader.XjtuSyReader(1, first_time_to_predict=fttp, norm_rul=True)
        for split in ["dev", "val", "test"]:
            _, targets = dataset.load_split(split)
            for target in targets:
                assert np.max(target) == 1
