import numpy as np
import numpy.testing as npt
import pytest

from rul_datasets import reader


def test_additional_hparams():
    femto = reader.FemtoReader(1, first_time_to_predict=[10] * 5, norm_rul=True)
    assert femto.hparams["first_time_to_predict"] == [10] * 5
    assert femto.hparams["norm_rul"]
    assert femto.hparams["run_split_dist"] is None


@pytest.mark.needs_data
class TestFemtoReader:
    NUM_CHANNELS = 2

    @pytest.fixture(scope="class", autouse=True)
    def prepare_femto(self):
        for fd in range(1, 4):
            reader.FemtoReader(fd).prepare_data()

    @pytest.mark.parametrize("fd", [1, 2, 3])
    @pytest.mark.parametrize("window_size", [2560, 1500, 1000, 100])
    @pytest.mark.parametrize("split", ["dev", "val", "test"])
    def test_run_shape_and_dtype(self, fd, window_size, split):
        femto_reader = reader.FemtoReader(fd, window_size=window_size)
        features, targets = femto_reader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        assert win == run.shape[1]
        assert self.NUM_CHANNELS == run.shape[2]
        assert len(run) == len(run_target)
        assert np.float64 == run.dtype
        assert np.float64 == run_target.dtype

    def test_standardization(self):
        for i in range(1, 3):
            full_dataset = reader.FemtoReader(fd=i)
            full_train, full_train_targets = full_dataset.load_split("dev")

            npt.assert_almost_equal(
                0.0, np.mean(np.concatenate(full_train)).item(), decimal=3
            )
            npt.assert_almost_equal(
                1.0, np.std(np.concatenate(full_train)).item(), decimal=3
            )

            truncated_dataset = reader.FemtoReader(fd=i, percent_fail_runs=0.8)
            trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
            npt.assert_almost_equal(
                0.0, np.mean(np.concatenate(trunc_train)).item(), decimal=2
            )
            npt.assert_almost_equal(
                1.0, np.std(np.concatenate(trunc_train)).item(), decimal=1
            )

            # percent_broken is supposed to change the std but not the mean
            truncated_dataset = reader.FemtoReader(fd=i, percent_broken=0.2)
            trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
            npt.assert_almost_equal(
                0.0, np.mean(np.concatenate(trunc_train)).item(), decimal=1
            )

    @pytest.mark.parametrize("max_rul", [125, None])
    def test_max_rul(self, max_rul):
        dataset = reader.FemtoReader(fd=1, max_rul=max_rul)
        _, targets = dataset.load_split("dev")
        for t in targets:
            if max_rul is None:
                npt.assert_equal(t, np.arange(len(t), 0, -1))  # is linear
            else:
                assert np.max(t) <= max_rul  # capped at max_rul

    def test_first_time_to_predict(self):
        fttp = [10, 20, 30, 40, 50, 60, 70]
        dataset = reader.FemtoReader(1, first_time_to_predict=fttp)
        targets = (
            dataset.load_split("dev")[1]
            + dataset.load_split("val")[1]
            + dataset.load_split("test")[1]
        )
        for target, first_time in zip(targets, fttp):
            max_rul = len(target) - first_time
            assert np.all(target[:first_time] == max_rul)

    def test_norm_rul_with_max_rul(self):
        dataset = reader.FemtoReader(1, max_rul=50, norm_rul=True)
        for split in ["dev", "val", "test"]:
            _, targets = dataset.load_split(split)
            for target in targets:
                assert np.max(target) == 1

    def test_norm_rul_with_fttp(self):
        fttp = [10, 20, 30, 40, 50, 60, 70]
        dataset = reader.FemtoReader(1, first_time_to_predict=fttp, norm_rul=True)
        for split in ["dev", "val", "test"]:
            _, targets = dataset.load_split(split)
            for target in targets:
                assert np.max(target) == 1
