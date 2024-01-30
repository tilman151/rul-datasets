import unittest
import warnings
from unittest import mock

import numpy as np
import pytest
import torch
import torch.utils.data

import rul_datasets
from tests.templates import PretrainingDataModuleTemplate


class TestBaselineDataModule:
    @pytest.fixture()
    def mock_reader(self):
        mock_reader = mock.MagicMock(name="AbstractLoader")
        mock_reader.fd = 1
        mock_reader.fds = [1, 2, 3]
        mock_reader.hparams = {
            "fd": mock_reader.fd,
            "window_size": mock_reader.window_size,
        }
        mock_runs = [np.zeros((1, 1, 1))], [np.zeros(1)]
        mock_reader.load_split.return_value = mock_runs

        return mock_reader

    @pytest.fixture()
    def dataset(self, mock_reader):
        base_module = rul_datasets.RulDataModule(mock_reader, batch_size=16)
        dataset = rul_datasets.BaselineDataModule(base_module)
        dataset.prepare_data()
        dataset.setup()

        return dataset

    def test_test_sets_created_correctly(self, mock_reader, dataset):
        for fd in mock_reader.fds:
            assert fd in dataset.subsets
            assert fd == dataset.subsets[fd].reader.fd
            if fd == dataset.data_module.reader.fd:
                assert dataset.data_module is dataset.subsets[fd]
            else:
                assert dataset.subsets[fd].reader.percent_fail_runs is None
                assert dataset.subsets[fd].reader.percent_broken is None

    def test_selected_source_on_train(self, dataset, mocker):
        mocker.patch.object(
            dataset.data_module, "train_dataloader", return_value=mocker.sentinel.dl
        )
        assert dataset.train_dataloader() is mocker.sentinel.dl

    def test_selected_source_on_val(self, dataset, mocker):
        mocker.patch.object(
            dataset.data_module, "val_dataloader", return_value=mocker.sentinel.dl
        )
        assert dataset.val_dataloader() is mocker.sentinel.dl

    def test_selected_all_on_test(self, dataset, mocker):
        for fd in [1, 2, 3]:
            sentinel = getattr(mocker.sentinel, f"dl_{fd}")
            mocker.patch.object(
                dataset.subsets[fd], "test_dataloader", return_value=sentinel
            )
        for fd, baseline_test_loader in enumerate(dataset.test_dataloader(), start=1):
            assert baseline_test_loader is getattr(mocker.sentinel, f"dl_{fd}")

    def test_hparams(self, dataset):
        assert dataset.hparams, dataset.data_module.hparams


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
                num_broken_runs = len(dataset.unfailed.reader.load_split(split)[0])
                num_fail_runs = len(dataset.failed.reader.load_split(split)[0])
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
        with pytest.warns(UserWarning):
            rul_datasets.PretrainingBaselineDataModule(
                self.failed_data, self.unfailed_data, 10
            )


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
