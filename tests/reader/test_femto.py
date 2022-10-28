import numpy as np
import numpy.testing as npt
import pytest
import torch

from rul_datasets import reader


@pytest.fixture(scope="module", autouse=True)
def prepare_femto():
    for fd in range(1, 4):
        reader.FemtoReader(fd).prepare_data()


@pytest.mark.needs_data
class TestFEMTOLoader:
    NUM_CHANNELS = 2

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("window_size", [2560, 1500, 1000, 100])
    @pytest.mark.parametrize("split", ["dev", "val", "test"])
    def test_run_shape_and_dtype(self, fd, window_size, split):
        femto_loader = reader.FemtoReader(fd, window_size=window_size)
        features, targets = femto_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        assert win == run.shape[2]
        assert self.NUM_CHANNELS == run.shape[1]
        assert len(run) == len(run_target)
        assert torch.float32 == run.dtype
        assert torch.float32 == run_target.dtype

    def test_standardization(self):
        for i in range(1, 3):
            full_dataset = reader.FemtoReader(fd=i)
            full_train, full_train_targets = full_dataset.load_split("dev")

            npt.assert_almost_equal(
                0.0, torch.mean(torch.cat(full_train)).item(), decimal=3
            )
            npt.assert_almost_equal(
                1.0, torch.std(torch.cat(full_train)).item(), decimal=3
            )

            truncated_dataset = reader.FemtoReader(fd=i, percent_fail_runs=0.8)
            trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
            npt.assert_almost_equal(
                0.0, torch.mean(torch.cat(trunc_train)).item(), decimal=2
            )
            npt.assert_almost_equal(
                1.0, torch.std(torch.cat(trunc_train)).item(), decimal=1
            )

            # percent_broken is supposed to change the std but not the mean
            truncated_dataset = reader.FemtoReader(fd=i, percent_broken=0.2)
            trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
            npt.assert_almost_equal(
                0.0, torch.mean(torch.cat(trunc_train)).item(), decimal=1
            )

    @pytest.mark.parametrize("max_rul", [125, None])
    def test_max_rul(self, max_rul):
        dataset = reader.FemtoReader(fd=1, max_rul=max_rul)
        _, targets = dataset.load_split("dev")
        for t in targets:
            t = t.numpy()
            if max_rul is None:
                npt.assert_equal(t, np.arange(len(t), 0, -1))  # is linear
            else:
                assert np.max(t) <= max_rul  # capped at max_rul
