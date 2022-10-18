import unittest

import pytest
import torch

from rul_datasets import loader


@pytest.fixture(scope="module", autouse=True)
def prepare_femto():
    for fd in range(1, 4):
        loader.FemtoLoader(fd).prepare_data()


@pytest.mark.needs_data
class TestFEMTOLoader(unittest.TestCase):
    NUM_CHANNELS = 2

    def test_run_shape_and_dtype(self):
        window_sizes = [2560, 1500, 1000, 100]
        for win in window_sizes:
            for fd in range(1, 4):
                femto_loader = loader.FemtoLoader(fd, window_size=win)
                for split in ["dev", "test"]:
                    with self.subTest(fd=fd, split=split):
                        self._check_split(femto_loader, split, win)

    def _check_split(self, rul_loader, split, window_size):
        features, targets = rul_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        self.assertEqual(win, run.shape[2])
        self.assertEqual(self.NUM_CHANNELS, run.shape[1])
        self.assertEqual(len(run), len(run_target))
        self.assertEqual(torch.float32, run.dtype)
        self.assertEqual(torch.float32, run_target.dtype)

    def test_standardization(self):
        for i in range(1, 3):
            with self.subTest(fd=i):
                full_dataset = loader.FemtoLoader(fd=i)
                full_train, full_train_targets = full_dataset.load_split("dev")

                self.assertAlmostEqual(
                    0.0, torch.mean(torch.cat(full_train)).item(), delta=0.0001
                )
                self.assertAlmostEqual(
                    1.0, torch.std(torch.cat(full_train)).item(), delta=0.0001
                )

                truncated_dataset = loader.FemtoLoader(fd=i, percent_fail_runs=0.8)
                trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
                self.assertAlmostEqual(
                    0.0, torch.mean(torch.cat(trunc_train)).item(), delta=0.1
                )
                self.assertAlmostEqual(
                    1.0, torch.std(torch.cat(trunc_train)).item(), delta=0.1
                )

                # percent_broken is supposed to change the std but not the mean
                truncated_dataset = loader.FemtoLoader(fd=i, percent_broken=0.2)
                trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")
                self.assertAlmostEqual(
                    0.0, torch.mean(torch.cat(trunc_train)).item(), delta=0.1
                )
