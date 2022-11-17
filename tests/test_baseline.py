import unittest
import warnings
from unittest import mock

import numpy as np
import torch
import torch.utils.data

import rul_datasets
from tests.templates import PretrainingDataModuleTemplate


class TestBaselineDataModule(unittest.TestCase):
    def setUp(self):
        self.mock_loader = mock.MagicMock(name="AbstractLoader")
        self.mock_loader.fd = 1
        self.mock_loader.fds = [1, 2, 3]
        self.mock_loader.hparams = {
            "fd": self.mock_loader.fd,
            "window_size": self.mock_loader.window_size,
        }
        self.mock_runs = [torch.zeros(1, 1, 1)], [torch.zeros(1)]
        self.mock_loader.load_split.return_value = self.mock_runs

        self.base_module = rul_datasets.RulDataModule(self.mock_loader, batch_size=16)
        self.dataset = rul_datasets.BaselineDataModule(self.base_module)
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_test_sets_created_correctly(self):
        for fd in self.mock_loader.fds:
            self.assertIn(fd, self.dataset.subsets)
            self.assertEqual(fd, self.dataset.subsets[fd].reader.fd)
            if fd == self.dataset.hparams["fd"]:
                self.assertIs(self.dataset.data_module, self.dataset.subsets[fd])
            else:
                self.assertIsNone(self.dataset.subsets[fd].reader.percent_fail_runs)
                self.assertIsNone(self.dataset.subsets[fd].reader.percent_broken)

    def test_selected_source_on_train(self):
        baseline_train_dataset = self.dataset.train_dataloader().dataset
        source_train_dataset = self.dataset.data_module.train_dataloader().dataset
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_source_on_val(self):
        baseline_val_dataset = self.dataset.val_dataloader().dataset
        source_val_dataset = self.dataset.data_module.val_dataloader().dataset
        self._assert_datasets_equal(baseline_val_dataset, source_val_dataset)

    def test_selected_all_on_test(self):
        baseline_test_loaders = self.dataset.test_dataloader()
        for fd, baseline_test_loader in enumerate(baseline_test_loaders, start=1):
            baseline_test_dataset = baseline_test_loader.dataset
            test_dataset = self.dataset.subsets[fd].test_dataloader().dataset
            self._assert_datasets_equal(baseline_test_dataset, test_dataset)

    def _assert_datasets_equal(self, baseline_dataset, inner_dataset):
        num_samples = len(baseline_dataset)
        baseline_data = baseline_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))

    def test_hparams(self):
        self.assertDictEqual(self.base_module.hparams, self.dataset.hparams)


class TestPretrainingBaselineDataModuleFullData(
    PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        self.mock_runs = [np.random.randn(16, 1, 14)] * 8, [np.random.rand(16)] * 8

        self.failed_loader = mock.MagicMock(name="CMAPSSLoader")
        self.failed_loader.fd = 1
        self.failed_loader.percent_fail_runs = list(range(8))
        self.failed_loader.percent_broken = None
        self.failed_loader.window_size = 1
        self.failed_loader.max_rul = 125
        self.failed_loader.hparams = {
            "fd": self.failed_loader.fd,
            "window_size": self.failed_loader.window_size,
        }
        self.failed_loader.load_split.return_value = self.mock_runs
        self.failed_data = rul_datasets.RulDataModule(self.failed_loader, batch_size=16)

        self.unfailed_loader = mock.MagicMock(name="CMAPSSLoader")
        self.unfailed_loader.fd = 1
        self.unfailed_loader.percent_fail_runs = list(range(8, 16))
        self.unfailed_loader.percent_broken = 0.8
        self.unfailed_loader.window_size = 1
        self.unfailed_loader.max_rul = 125
        self.unfailed_loader.hparams = {
            "fd": self.unfailed_loader.fd,
            "window_size": self.unfailed_loader.window_size,
        }
        self.unfailed_loader.load_split.return_value = self.mock_runs
        self.unfailed_data = rul_datasets.RulDataModule(
            self.unfailed_loader, batch_size=16
        )

        self.dataset = rul_datasets.PretrainingBaselineDataModule(
            self.failed_data,
            self.unfailed_data,
            num_samples=10000,
            min_distance=2,
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 2
        self.window_size = 1

    def test_loader_compatability_checked(self):
        self.failed_loader.check_compatibility.assert_called_with(self.unfailed_loader)

    def test_distance_mode_passed_correctly(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            self.failed_data, self.unfailed_data, 10, distance_mode="labeled"
        )
        data_loader = dataset.train_dataloader()
        self.assertEqual(dataset.distance_mode, data_loader.dataset.mode)

    def test_both_source_datasets_used(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            self.failed_data, self.unfailed_data, 10
        )
        for split in ["dev", "val"]:
            with self.subTest(split):
                num_broken_runs = len(dataset.unfailed_loader.load_split(split)[0])
                num_fail_runs = len(dataset.failed_loader.load_split(split)[0])
                paired_dataset = dataset._get_paired_dataset(split)
                self.assertEqual(
                    num_broken_runs + num_fail_runs, len(paired_dataset._features)
                )

    def test_error_on_fd_missmatch(self):
        self.failed_data.reader.fd = 2
        self.assertRaises(
            ValueError,
            rul_datasets.PretrainingBaselineDataModule,
            self.failed_data,
            self.unfailed_data,
            10,
        )

    def test_error_on_float_fail_runs(self):
        with self.subTest("failed data"):
            self.failed_data.reader.percent_fail_runs = 0.9
            self.assertRaises(
                ValueError,
                rul_datasets.PretrainingBaselineDataModule,
                self.failed_data,
                self.unfailed_data,
                10,
            )

        with self.subTest("failed data"):
            self.failed_data.reader.percent_fail_runs = list(range(0, 8))
            self.unfailed_data.reader.percent_fail_runs = 0.9
            self.assertRaises(
                ValueError,
                rul_datasets.PretrainingBaselineDataModule,
                self.failed_data,
                self.unfailed_data,
                10,
            )

    def test_error_on_overlapping_runs(self):
        self.unfailed_data.reader.percent_fail_runs = list(range(6, 14))
        self.assertRaises(
            ValueError,
            rul_datasets.PretrainingBaselineDataModule,
            self.failed_data,
            self.unfailed_data,
            10,
        )

    def test_error_on_no_percent_broken_for_unfailed_data(self):
        with self.subTest("none"):
            self.unfailed_data.reader.percent_broken = None
            self.assertRaises(
                ValueError,
                rul_datasets.PretrainingBaselineDataModule,
                self.failed_data,
                self.unfailed_data,
                10,
            )
        with self.subTest("1.0"):
            self.unfailed_data.reader.percent_broken = 1.0
            self.assertRaises(
                ValueError,
                rul_datasets.PretrainingBaselineDataModule,
                self.failed_data,
                self.unfailed_data,
                10,
            )

    def test_error_on_percent_broken_for_failed_data(self):
        self.failed_data.reader.percent_broken = 0.8
        self.assertRaises(
            ValueError,
            rul_datasets.PretrainingBaselineDataModule,
            self.failed_data,
            self.unfailed_data,
            10,
        )

    def test_warning_on_non_truncated_val_data(self):
        self.unfailed_data.reader.truncate_val = False
        with warnings.catch_warnings(record=True) as warn:
            rul_datasets.PretrainingBaselineDataModule(
                self.failed_data, self.unfailed_data, 10
            )
        self.assertTrue(warn)


class TestPretrainingBaselineDataModuleLowData(
    PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        self.mock_runs = [np.random.randn(16, 1, 14)] * 2, [np.random.rand(16)] * 2

        self.failed_loader = mock.MagicMock(name="CMAPSSLoader")
        self.failed_loader.fd = 1
        self.failed_loader.percent_fail_runs = list(range(8))
        self.failed_loader.percent_broken = None
        self.failed_loader.window_size = 1
        self.failed_loader.max_rul = 125
        self.failed_loader.hparams = {
            "fd": self.failed_loader.fd,
            "window_size": self.failed_loader.window_size,
        }
        self.failed_loader.load_split.return_value = self.mock_runs
        self.failed_data = rul_datasets.RulDataModule(self.failed_loader, batch_size=16)

        self.unfailed_loader = mock.MagicMock(name="CMAPSSLoader")
        self.unfailed_loader.fd = 1
        self.unfailed_loader.percent_fail_runs = list(range(8, 16))
        self.unfailed_loader.percent_broken = 0.2
        self.unfailed_loader.window_size = 1
        self.unfailed_loader.max_rul = 125
        self.unfailed_loader.hparams = {
            "fd": self.unfailed_loader.fd,
            "window_size": self.unfailed_loader.window_size,
        }
        self.unfailed_loader.load_split.return_value = self.mock_runs
        self.unfailed_data = rul_datasets.RulDataModule(
            self.unfailed_loader, batch_size=16
        )

        self.dataset = rul_datasets.PretrainingBaselineDataModule(
            self.failed_data,
            self.unfailed_data,
            num_samples=10000,
            min_distance=2,
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 2
        self.window_size = 1
