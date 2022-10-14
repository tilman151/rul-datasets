import unittest

import pytest
import torch

from rul_datasets import loader


@pytest.mark.needs_data
class TestCMAPSSLoader(unittest.TestCase):
    NUM_CHANNELS = len(loader.CmapssLoader._DEFAULT_CHANNELS)

    @classmethod
    def setUpClass(cls):
        for fd in range(1, 5):
            loader.CmapssLoader(fd).prepare_data()

    def test_run_shape_and_dtype(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            rul_loader = loader.CmapssLoader(n)
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=n, split=split):
                    self._check_split(rul_loader, split, win)

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

    def test_default_window_size(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            rul_loader = loader.CmapssLoader(n)
            self.assertEqual(win, rul_loader.window_size)

    def test_default_feature_select(self):
        rul_loader = loader.CmapssLoader(1)
        self.assertListEqual(rul_loader._DEFAULT_CHANNELS, rul_loader.feature_select)

    def test_feature_select(self):
        dataset = loader.CmapssLoader(1, feature_select=[4, 9, 10, 13, 14, 15, 22])
        for split in ["dev", "val", "test"]:
            features, _ = dataset.load_split(split)
            for run in features:
                self.assertEqual(7, run.shape[1])

    def test_normalization_min_max(self):
        for i in range(1, 5):
            with self.subTest(fd=i):
                full_dataset = loader.CmapssLoader(fd=i, window_size=30)
                full_dev, full_dev_targets = full_dataset.load_split("dev")

                self.assertAlmostEqual(max(torch.max(r).item() for r in full_dev), 1.0)
                self.assertAlmostEqual(min(torch.min(r).item() for r in full_dev), -1.0)

                truncated_dataset = loader.CmapssLoader(
                    fd=i, window_size=30, percent_fail_runs=0.8
                )
                trunc_dev, trunc_dev_targets = truncated_dataset.load_split("dev")
                self.assertLessEqual(max(torch.max(r).item() for r in trunc_dev), 1.0)
                self.assertGreaterEqual(
                    min(torch.min(r).item() for r in trunc_dev), -1.0
                )

                truncated_dataset = loader.CmapssLoader(
                    fd=i, window_size=30, percent_broken=0.2
                )
                trunc_dev, trunc_dev_targets = truncated_dataset.load_split("dev")
                self.assertLessEqual(max(torch.max(r).item() for r in trunc_dev), 1.0)
                self.assertGreaterEqual(
                    min(torch.min(r).item() for r in trunc_dev), -1.0
                )
