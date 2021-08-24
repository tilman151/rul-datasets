import os
import unittest
from unittest import mock

import numpy as np
import torch

from datasets import loader
from tests.dataset_tests.templates import CmapssTestTemplate, FemtoTestTemplate


class TestCMAPSSLoader(CmapssTestTemplate, unittest.TestCase):
    NUM_CHANNELS = len(loader.CMAPSSLoader.DEFAULT_CHANNELS)

    def test_run_shape_and_dtype(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            cmapss_loader = loader.CMAPSSLoader(n)
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=n, split=split):
                    self._check_split(cmapss_loader, split, win)

    def _check_split(self, cmapss_loader, split, window_size):
        features, targets = cmapss_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        self.assertEqual(win, run.shape[2])
        self.assertEqual(self.NUM_CHANNELS, run.shape[1])
        self.assertEqual(len(run), len(run_target))
        self.assertEqual(torch.float32, run.dtype)
        self.assertEqual(torch.float32, run_target.dtype)

    def test_override_window_size(self):
        window_size = 40
        for n in range(1, 5):
            dataset = loader.CMAPSSLoader(n, window_size=window_size)
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=n, split=split):
                    features, targets = dataset.load_split(split)
                    for run in features:
                        self.assertEqual(window_size, run.shape[2])

    def test_feature_select(self):
        dataset = loader.CMAPSSLoader(1, feature_select=[4, 9, 10, 13, 14, 15, 22])
        for split in ["dev", "val", "test"]:
            features, _ = dataset.load_split(split)
            for run in features:
                self.assertEqual(7, run.shape[1])

    def test_percent_fail_runs(self):
        full_dataset = loader.CMAPSSLoader(fd=1, window_size=30)
        full_dev, full_dev_targets = full_dataset.load_split("dev")
        full_val = full_dataset.load_split("val")[0]
        full_test = full_dataset.load_split("test")[0]

        dataset = loader.CMAPSSLoader(fd=1, window_size=30, percent_fail_runs=0.8)
        trunc_dev, trunc_dev_targets = dataset.load_split("dev")

        self.assertGreater(len(full_dev), len(trunc_dev))
        self.assertAlmostEqual(0.8, len(trunc_dev) / len(full_dev), delta=0.01)
        self.assertEqual(len(full_val), len(dataset.load_split("val")[0]))
        self.assertEqual(len(full_test), len(dataset.load_split("test")[0]))

        for full_run, trunc_run in zip(full_dev, trunc_dev):
            self.assertEqual(0.0, torch.dist(trunc_run, full_run))
        for full_targets, trunc_targets in zip(full_dev_targets, trunc_dev_targets):
            self.assertEqual(0.0, torch.dist(trunc_targets, full_targets))

    def test_percent_broken(self):
        full_dataset = loader.CMAPSSLoader(fd=1, window_size=30)
        full_dev, full_dev_targets = full_dataset.load_split("dev")
        full_val = full_dataset.load_split("val")[0]
        full_test = full_dataset.load_split("test")[0]

        dataset = loader.CMAPSSLoader(fd=1, window_size=30, percent_broken=0.2)
        truncated_dev, truncated_dev_targets = dataset.load_split("dev")

        full_length = sum(len(r) for r in full_dev)
        truncated_length = sum(len(r) for r in truncated_dev)
        self.assertGreater(full_length, truncated_length)
        self.assertAlmostEqual(0.2, truncated_length / full_length, delta=0.01)

        # No failure data in truncated dev data
        self.assertFalse(any(torch.any(r == 1) for r in truncated_dev_targets))

        for full_run, trunc_run in zip(full_dev, truncated_dev):
            trunc_run_length = trunc_run.shape[0]
            self.assertEqual(0.0, torch.dist(trunc_run, full_run[:trunc_run_length]))
        for full_run, trunc_run in zip(full_dev_targets, truncated_dev_targets):
            trunc_run_length = trunc_run.shape[0]
            self.assertEqual(0.0, torch.dist(trunc_run, full_run[:trunc_run_length]))

        # Val dataset_tests not truncated
        for full_run, trunc_run in zip(full_val, dataset.load_split("val")[0]):
            self.assertEqual(len(full_run), len(trunc_run))

        # Test dataset_tests not truncated
        for full_run, trunc_run in zip(full_test, dataset.load_split("test")[0]):
            self.assertEqual(len(full_run), len(trunc_run))

    @mock.patch(
        "datasets.loader.CMAPSSLoader._truncate_runs", wraps=lambda x, y, *args: (x, y)
    )
    def test_val_truncation(self, mock_truncate):
        dataset = loader.CMAPSSLoader(fd=1, window_size=30)
        with self.subTest(truncate_val=False):
            dataset.load_split("dev")
            mock_truncate.assert_called_once()
            mock_truncate.reset_mock()
            dataset.load_split("val")
            mock_truncate.assert_not_called()

        dataset = loader.CMAPSSLoader(fd=1, window_size=30, truncate_val=True)
        with self.subTest(truncate_val=True):
            dataset.load_split("dev")
            mock_truncate.assert_called_once()
            mock_truncate.reset_mock()
            dataset.load_split("val")
            mock_truncate.assert_called_once()

    def test_normalization_min_max(self):
        for i in range(1, 5):
            with self.subTest(fd=i):
                full_dataset = loader.CMAPSSLoader(fd=i, window_size=30)
                full_dev, full_dev_targets = full_dataset.load_split("dev")

                self.assertAlmostEqual(max(torch.max(r).item() for r in full_dev), 1.0)
                self.assertAlmostEqual(min(torch.min(r).item() for r in full_dev), -1.0)

                truncated_dataset = loader.CMAPSSLoader(
                    fd=i, window_size=30, percent_fail_runs=0.8
                )
                trunc_dev, trunc_dev_targets = truncated_dataset.load_split("dev")
                self.assertLessEqual(max(torch.max(r).item() for r in trunc_dev), 1.0)
                self.assertGreaterEqual(
                    min(torch.min(r).item() for r in trunc_dev), -1.0
                )

                truncated_dataset = loader.CMAPSSLoader(
                    fd=i, window_size=30, percent_broken=0.2
                )
                trunc_dev, trunc_dev_targets = truncated_dataset.load_split("dev")
                self.assertLessEqual(max(torch.max(r).item() for r in trunc_dev), 1.0)
                self.assertGreaterEqual(
                    min(torch.min(r).item() for r in trunc_dev), -1.0
                )

    def test_truncation_by_index(self):
        full_dataset = loader.CMAPSSLoader(1)
        full_train, full_train_targets = full_dataset.load_split("dev")
        indices = [50, 60, 70]
        truncated_dataset = loader.CMAPSSLoader(1, percent_fail_runs=indices)
        trunc_train, trunc_train_targets = truncated_dataset.load_split("dev")

        self.assertEqual(len(indices), len(trunc_train))
        self.assertEqual(len(indices), len(trunc_train_targets))
        for trunc_idx, full_idx in enumerate(indices):
            self.assertEqual(
                0, torch.dist(full_train[full_idx], trunc_train[trunc_idx])
            )
            self.assertEqual(
                0,
                torch.dist(
                    full_train_targets[full_idx], trunc_train_targets[trunc_idx]
                ),
            )


class TestFEMTOLoader(FemtoTestTemplate, unittest.TestCase):
    NUM_CHANNELS = 2

    def test_run_shape_and_dtype(self):
        window_sizes = [2560, 1500, 1000, 100]
        for win in window_sizes:
            for fd in range(1, 4):
                femto_loader = loader.FEMTOLoader(fd, window_size=win)
                for split in ["train", "test"]:
                    with self.subTest(fd=fd, split=split):
                        self._check_split(femto_loader, split, win)

    def _check_split(self, cmapss_loader, split, window_size):
        features, targets = cmapss_loader.load_split(split)
        for run, run_target in zip(features, targets):
            self._assert_run_correct(run, run_target, window_size)

    def _assert_run_correct(self, run, run_target, win):
        self.assertEqual(win, run.shape[2])
        self.assertEqual(self.NUM_CHANNELS, run.shape[1])
        self.assertEqual(len(run), len(run_target))
        self.assertEqual(torch.float32, run.dtype)
        self.assertEqual(torch.float32, run_target.dtype)

    def test_percent_fail_runs(self):
        full_dataset = loader.FEMTOLoader(fd=1)
        full_train, full_train_targets = full_dataset.load_split("train")
        full_test = full_dataset.load_split("test")[0]

        dataset = loader.FEMTOLoader(fd=1, percent_fail_runs=0.5)
        trunc_train, trunc_train_targets = dataset.load_split("train")

        self.assertGreater(len(full_train), len(trunc_train))
        self.assertAlmostEqual(0.5, len(trunc_train) / len(full_train), delta=0.01)
        self.assertEqual(len(full_test), len(dataset.load_split("test")[0]))

        for full_run, trunc_run in zip(full_train, trunc_train):
            self.assertEqual(0.0, torch.dist(trunc_run, full_run))
        for full_targets, trunc_targets in zip(full_train_targets, trunc_train_targets):
            self.assertEqual(0.0, torch.dist(trunc_targets, full_targets))

    def test_percent_broken(self):
        full_dataset = loader.FEMTOLoader(fd=1)
        full_train, full_train_targets = full_dataset.load_split("train")
        full_test = full_dataset.load_split("test")[0]

        dataset = loader.FEMTOLoader(fd=1, percent_broken=0.2)
        truncated_train, truncated_train_targets = dataset.load_split("train")

        full_length = sum(len(r) for r in full_train)
        truncated_length = sum(len(r) for r in truncated_train)
        self.assertGreater(full_length, truncated_length)
        self.assertAlmostEqual(0.2, truncated_length / full_length, delta=0.01)

        # No failure data in truncated train data
        self.assertFalse(any(torch.any(r == 1) for r in truncated_train_targets))

        for full_run, trunc_run in zip(full_train, truncated_train):
            trunc_run_length = trunc_run.shape[0]
            self.assertEqual(0.0, torch.dist(trunc_run, full_run[:trunc_run_length]))
        for full_run, trunc_run in zip(full_train_targets, truncated_train_targets):
            trunc_run_length = trunc_run.shape[0]
            self.assertEqual(0.0, torch.dist(trunc_run, full_run[:trunc_run_length]))

        # Test dataset_tests not truncated
        for full_run, trunc_run in zip(full_test, dataset.load_split("test")[0]):
            self.assertEqual(len(full_run), len(trunc_run))

    @unittest.skip("No val set yet.")
    @mock.patch(
        "datasets.loader.CMAPSSLoader._truncate_runs", wraps=lambda x, y: (x, y)
    )
    def test_val_truncation(self, mock_truncate):
        dataset = loader.CMAPSSLoader(fd=1, window_size=30)
        with self.subTest(truncate_val=False):
            dataset.load_split("dev")
            mock_truncate.assert_called_once()
            mock_truncate.reset_mock()
            dataset.load_split("val")
            mock_truncate.assert_not_called()

        dataset = loader.CMAPSSLoader(fd=1, window_size=30, truncate_val=True)
        with self.subTest(truncate_val=True):
            dataset.load_split("dev")
            mock_truncate.assert_called_once()
            mock_truncate.reset_mock()
            dataset.load_split("val")
            mock_truncate.assert_called_once()

    def test_standardization(self):
        for i in range(1, 3):
            with self.subTest(fd=i):
                full_dataset = loader.FEMTOLoader(fd=i)
                full_train, full_train_targets = full_dataset.load_split("train")

                self.assertAlmostEqual(
                    0.0, torch.mean(torch.cat(full_train)).item(), delta=0.0001
                )
                self.assertAlmostEqual(
                    1.0, torch.std(torch.cat(full_train)).item(), delta=0.0001
                )

                truncated_dataset = loader.FEMTOLoader(fd=i, percent_fail_runs=0.8)
                trunc_train, trunc_train_targets = truncated_dataset.load_split("train")
                self.assertAlmostEqual(
                    0.0, torch.mean(torch.cat(trunc_train)).item(), delta=0.1
                )
                self.assertAlmostEqual(
                    1.0, torch.std(torch.cat(trunc_train)).item(), delta=0.1
                )

                # percent_broken is supposed to change the std but not the mean
                truncated_dataset = loader.FEMTOLoader(fd=i, percent_broken=0.2)
                trunc_train, trunc_train_targets = truncated_dataset.load_split("train")
                self.assertAlmostEqual(
                    0.0, torch.mean(torch.cat(trunc_train)).item(), delta=0.1
                )

    def test_truncation_by_index(self):
        full_dataset = loader.FEMTOLoader(1)
        full_train, full_train_targets = full_dataset.load_split("train")
        truncated_dataset = loader.FEMTOLoader(1, percent_fail_runs=[1])
        trunc_train, trunc_train_targets = truncated_dataset.load_split("train")

        self.assertEqual(1, len(trunc_train))
        self.assertEqual(1, len(trunc_train_targets))
        self.assertEqual(0, torch.dist(full_train[1], trunc_train[0]))
        self.assertEqual(0, torch.dist(full_train_targets[1], trunc_train_targets[0]))


class TestFEMTOPreperator(FemtoTestTemplate, unittest.TestCase):
    NUM_SAMPLES = {
        1: {"train": 3674, "test": 10973},
        2: {"train": 1708, "test": 5948},
        3: {"train": 2152, "test": 434},
    }

    def test_file_discovery(self):
        for fd in range(1, 4):
            preparator = loader.FEMTOPreparator(fd, loader.FEMTOLoader.DATA_ROOT)
            for split in ["train", "test"]:
                csv_paths = preparator._get_csv_file_paths(split)
                expected_num_files = self.NUM_SAMPLES[fd][split]
                actual_num_files = len(sum(csv_paths, []))
                self.assertEqual(expected_num_files, actual_num_files)

    def test_loading_one_file(self):
        preparator = loader.FEMTOPreparator(1, loader.FEMTOLoader.DATA_ROOT)
        csv_paths = preparator._get_csv_file_paths("train")
        features = preparator._load_feature_file(csv_paths[0][0])
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (preparator.DEFAULT_WINDOW_SIZE, 2))

    def test_target_generation(self):
        preparator = loader.FEMTOPreparator(1, loader.FEMTOLoader.DATA_ROOT)
        csv_paths = preparator._get_csv_file_paths("train")
        targets = preparator._targets_from_file_paths(csv_paths)
        for run_targets, run_files in zip(targets, csv_paths):
            self.assertIsInstance(run_targets, np.ndarray)
            self.assertEqual(run_targets.shape, (len(run_files),))
            self.assertEqual(len(run_targets), np.max(run_targets))
            self.assertEqual(1, np.min(run_targets))

    @unittest.skip("Takes a lot of time.")
    def test_preparation(self):
        for fd in range(1, 4):
            preparator = loader.FEMTOPreparator(fd, loader.FEMTOLoader.DATA_ROOT)
            for split in ["train", "test"]:
                preparator.prepare_split(split)
                expected_file_path = preparator._get_run_file_path(split)
                self.assertTrue(os.path.exists(expected_file_path))
            expected_scaler_path = preparator._get_scaler_path()
            self.assertTrue(os.path.exists(expected_scaler_path))

    def test_converted_files(self):
        for fd in range(1, 4):
            preparator = loader.FEMTOPreparator(fd, loader.FEMTOLoader.DATA_ROOT)
            for split in ["train", "test"]:
                features, targets = preparator.load_runs(split)
                expected_num_time_steps = self.NUM_SAMPLES[fd][split]
                actual_num_target_steps = sum(len(f) for f in features)
                self.assertEqual(expected_num_time_steps, actual_num_target_steps)
                actual_num_target_steps = sum(len(t) for t in targets)
                self.assertEqual(expected_num_time_steps, actual_num_target_steps)

    def test_scaler(self):
        for fd in range(1, 4):
            preparator = loader.FEMTOPreparator(fd, loader.FEMTOLoader.DATA_ROOT)
            scaler = preparator.load_scaler()
            expected_samples = (
                self.NUM_SAMPLES[fd]["train"] * preparator.DEFAULT_WINDOW_SIZE
            )
            self.assertEqual(2, scaler.n_features_in_)
            self.assertEqual(expected_samples, scaler.n_samples_seen_)
