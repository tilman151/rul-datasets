import pytest
import torch
from numpy import testing as npt

from rul_datasets import loader


@pytest.fixture(scope="module", autouse=True)
def prepare_xjtu_sy():
    for fd in range(1, 4):
        loader.XjtuSyLoader(fd).prepare_data()


@pytest.mark.needs_data
class TestXjtuSyLoader:
    NUM_CHANNELS = 2

    def test_default_window_size(self):
        xjtu = loader.XjtuSyLoader(1)
        assert xjtu.window_size == loader.XjtuSyPreparator.DEFAULT_WINDOW_SIZE

    @pytest.mark.parametrize("fd", [1, 2, 3])
    def test_standardization(self, fd):
        full_dataset = loader.XjtuSyLoader(fd)
        full_train, full_train_targets = full_dataset.load_split("dev")

        npt.assert_almost_equal(
            0.0, torch.mean(torch.cat(full_train)).item(), decimal=4
        )
        npt.assert_almost_equal(1.0, torch.std(torch.cat(full_train)).item(), decimal=4)

        truncated_dataset = loader.XjtuSyLoader(fd, percent_fail_runs=0.8)
        trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
        npt.assert_almost_equal(
            0.0, torch.mean(torch.cat(trunc_train)).item(), decimal=1
        )

        # percent_broken is supposed to change the std but not the mean
        truncated_dataset = loader.XjtuSyLoader(fd, percent_broken=0.2)
        trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
        npt.assert_almost_equal(
            0.0, torch.mean(torch.cat(trunc_train)).item(), decimal=1
        )

    @pytest.mark.parametrize("window_size", [1500, 100])
    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("split", ["dev", "val", "test"])
    def test_run_shape_and_dtype(self, window_size, fd, split):
        rul_loader = loader.XjtuSyLoader(fd, window_size=window_size)
        features, targets = rul_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        assert win == run.shape[2]
        assert self.NUM_CHANNELS == run.shape[1]
        assert len(run) == len(run_target)
        assert torch.float32 == run.dtype
        assert torch.float32 == run_target.dtype

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize(
        ["split", "exp_length"], [("dev", 2), ("val", 1), ("test", 2)]
    )
    def test_run_split_dist(self, fd, split, exp_length):
        rul_loader = loader.XjtuSyLoader(fd)
        features, targets = rul_loader.load_split(split)
        assert len(features) == len(targets) == exp_length
