import unittest

import torch
import torch.utils.data
from torch.utils.data import RandomSampler, SequentialSampler

import rul_datasets
from tests.templates import (
    CmapssTestTemplate,
    PretrainingDataModuleTemplate,
)


class TestCMAPSSBaseline(CmapssTestTemplate, unittest.TestCase):
    def setUp(self):
        self.dataset = rul_datasets.BaselineDataModule(
            3, batch_size=16, percent_fail_runs=0.8
        )
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_override_window_size(self):
        dataset = rul_datasets.BaselineDataModule(
            3, batch_size=16, window_size=40, percent_fail_runs=0.8
        )
        for fd in dataset.cmapss.values():
            self.assertEqual(40, fd.window_size)

    def test_default_window_size(self):
        window_sizes = [30, 20, 30, 15]
        for i, win in enumerate(window_sizes, start=1):
            dataset = rul_datasets.BaselineDataModule(
                i, batch_size=16, percent_fail_runs=0.8
            )
            for fd in dataset.cmapss.values():
                self.assertEqual(win, fd.window_size)

    def test_train_batch_structure(self):
        train_loader = self.dataset.train_dataloader()
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self._assert_batch_structure(train_loader)

    def test_val_batch_structure(self):
        val_loader = self.dataset.val_dataloader()
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self._assert_batch_structure(val_loader)

    def test_test_batch_structure(self):
        test_loaders = self.dataset.test_dataloader()
        for test_loader in test_loaders:
            self.assertIsInstance(test_loader.sampler, SequentialSampler)
            self._assert_batch_structure(test_loader)

    def _assert_batch_structure(self, loader):
        batch = next(iter(loader))
        self.assertEqual(2, len(batch))
        features, labels = batch
        self.assertEqual(torch.Size((16, 14, 30)), features.shape)
        self.assertEqual(torch.Size((16,)), labels.shape)

    def test_selected_source_on_train(self):
        fd_source = self.dataset.fd_source
        baseline_train_dataset = self.dataset.train_dataloader().dataset
        source_train_dataset = self.dataset.cmapss[fd_source].train_dataloader().dataset
        self._assert_datasets_equal(baseline_train_dataset, source_train_dataset)

    def test_selected_source_on_val(self):
        fd_source = self.dataset.fd_source
        baseline_val_dataset = self.dataset.val_dataloader().dataset
        source_val_dataset = self.dataset.cmapss[fd_source].val_dataloader().dataset
        self._assert_datasets_equal(baseline_val_dataset, source_val_dataset)

    def test_selected_all_on_test(self):
        baseline_test_loaders = self.dataset.test_dataloader()
        for fd, baseline_test_loader in enumerate(baseline_test_loaders, start=1):
            baseline_test_dataset = baseline_test_loader.dataset
            test_dataset = self.dataset.cmapss[fd].test_dataloader().dataset
            self._assert_datasets_equal(baseline_test_dataset, test_dataset)

    def _assert_datasets_equal(self, baseline_dataset, inner_dataset):
        num_samples = len(baseline_dataset)
        baseline_data = baseline_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.sum(baseline - inner))

    def test_fail_runs_passed_correctly(self):
        for i in range(1, 5):
            if i == self.dataset.fd_source:
                self.assertEqual(0.8, self.dataset.cmapss[i].percent_fail_runs)
            else:
                self.assertIsNone(self.dataset.cmapss[i].percent_fail_runs)

    def test_hparams(self):
        dataset = rul_datasets.BaselineDataModule(3, 16)
        expected_hparams = {
            "fd_source": 3,
            "batch_size": 16,
            "window_size": 30,
            "max_rul": 125,
            "percent_fail_runs": None,
        }
        self.assertDictEqual(expected_hparams, dataset.hparams)


class TestPretrainingBaselineDataModuleFullData(
    CmapssTestTemplate, PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        self.dataset = rul_datasets.PretrainingBaselineDataModule(
            3, num_samples=10000, batch_size=16, min_distance=2
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 2
        self.window_size = self.dataset.broken_source_loader.window_size

    def test_val_truncation(self):
        with self.subTest(truncation=False):
            dataset = rul_datasets.PretrainingBaselineDataModule(
                3, num_samples=10000, batch_size=16
            )
            self.assertFalse(dataset.broken_source_loader.truncate_val)
            self.assertFalse(dataset.source.truncate_val)

        with self.subTest(truncation=True):
            dataset = rul_datasets.PretrainingBaselineDataModule(
                3, num_samples=10000, batch_size=16, truncate_val=True
            )
            self.assertTrue(dataset.broken_source_loader.truncate_val)
            self.assertTrue(dataset.source.truncate_val)

    def test_override_window_size(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            3, num_samples=10000, batch_size=16, window_size=40
        )
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()

        anchors, queries, _, _ = next(iter(train_loader))
        self.assertEqual(40, anchors.shape[2])
        self.assertEqual(40, queries.shape[2])

    def test_truncation_passed_correctly(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            3, 1000, 16, percent_broken=0.2, percent_fail_runs=0.5
        )
        self.assertEqual(0.2, dataset.broken_source_loader.percent_broken)
        self.assertIsNone(dataset.broken_source_loader.percent_fail_runs)
        self.assertEqual(0.5, dataset.fails_source_loader.percent_fail_runs)
        self.assertIsNone(dataset.fails_source_loader.percent_broken)

    def test_distance_mode_passed_correctly(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            3, 1000, 16, distance_mode="labeled"
        )
        data_loader = dataset.train_dataloader()
        self.assertEqual(dataset.distance_mode, data_loader.dataset.mode)

    def test_both_source_datasets_used(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            3, 1000, 16, percent_broken=0.2, percent_fail_runs=0.5
        )
        for split in ["dev", "val"]:
            with self.subTest(split):
                num_broken_runs = len(dataset.broken_source_loader.load_split(split)[0])
                num_fail_runs = len(dataset.fails_source_loader.load_split(split)[0])
                paired_dataset = dataset._get_paired_dataset(split)
                self.assertEqual(
                    num_broken_runs + num_fail_runs, len(paired_dataset._features)
                )

    def test_get_unfailed_runs(self):
        fail_runs = [1, 5, 40, 79]
        dataset = rul_datasets.PretrainingBaselineDataModule(
            3, 1000, 16, percent_broken=0.2, percent_fail_runs=fail_runs
        )
        self.assertListEqual(fail_runs, dataset.fails_source_loader.percent_fail_runs)
        for r in fail_runs:
            self.assertNotIn(r, dataset.broken_source_loader.percent_fail_runs)

    def test_percent_fails_zero(self):
        dataset = rul_datasets.PretrainingBaselineDataModule(
            3, 1000, 16, percent_broken=0.2, percent_fail_runs=0.0
        )
        num_broken_runs = len(dataset.broken_source_loader.load_split("dev")[0])
        num_fail_runs = len(dataset.fails_source_loader.load_split("dev")[0])
        self.assertEqual(0, num_fail_runs)
        self.assertNotEqual(0, num_broken_runs)


class TestPretrainingBaselineDataModuleLowData(
    CmapssTestTemplate, PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        self.dataset = rul_datasets.PretrainingBaselineDataModule(
            3, num_samples=10000, batch_size=16, percent_broken=0.2
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 2
        self.window_size = self.dataset.broken_source_loader.window_size
