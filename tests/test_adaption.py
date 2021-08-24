import unittest

import torch
from torch.utils.data import TensorDataset

import datasets
import datasets.cmapss
from tests.dataset_tests.templates import (
    CmapssTestTemplate,
    PretrainingDataModuleTemplate,
)


class TestCMAPSSAdaption(CmapssTestTemplate, unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.DomainAdaptionDataModule(3, 2, batch_size=16)
        self.dataset.prepare_data()
        self.dataset.setup()

    def test_default_window_size(self):
        with self.subTest(case="bigger2smaller"):
            self.assertEqual(self.dataset.target.window_size, self.dataset.window_size)
            self.assertEqual(
                self.dataset.target.window_size, self.dataset.source.window_size
            )
            self.assertEqual(
                self.dataset.target.window_size,
                self.dataset.target_truncated.window_size,
            )

        with self.subTest(case="smaller2bigger"):
            dataset = datasets.DomainAdaptionDataModule(2, 3, batch_size=16)
            self.assertEqual(self.dataset.target.window_size, self.dataset.window_size)
            self.assertEqual(dataset.target.window_size, dataset.source.window_size)
            self.assertEqual(
                dataset.target.window_size, dataset.target_truncated.window_size
            )

    def test_override_window_size(self):
        dataset = datasets.DomainAdaptionDataModule(3, 2, batch_size=16, window_size=40)
        self.assertEqual(self.dataset.target.window_size, self.dataset.window_size)
        self.assertEqual(40, dataset.target.window_size)
        self.assertEqual(dataset.target.window_size, dataset.source.window_size)
        self.assertEqual(
            dataset.target.window_size, dataset.target_truncated.window_size
        )

    def test_val_source_target_order(self):
        val_source_loader, val_target_loader, _ = self.dataset.val_dataloader()
        self._assert_datasets_equal(
            val_source_loader.dataset,
            self.dataset.source.to_dataset("val"),
        )
        self._assert_datasets_equal(
            val_target_loader.dataset,
            self.dataset.target.to_dataset("val"),
        )

    def test_test_source_target_order(self):
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self._assert_datasets_equal(
            test_source_loader.dataset,
            self.dataset.source.to_dataset("test"),
        )
        self._assert_datasets_equal(
            test_target_loader.dataset,
            self.dataset.target.to_dataset("test"),
        )

    def _assert_datasets_equal(self, adaption_dataset, inner_dataset):
        num_samples = len(adaption_dataset)
        baseline_data = adaption_dataset[:num_samples]
        inner_data = inner_dataset[:num_samples]
        for baseline, inner in zip(baseline_data, inner_data):
            self.assertEqual(0, torch.dist(baseline, inner))

    def test_train_batch_structure(self):
        window_size = self.dataset.target.window_size
        train_loader = self.dataset.train_dataloader()
        self.assertIsInstance(train_loader.dataset, datasets.cmapss.AdaptionDataset)
        batch = next(iter(train_loader))
        self.assertEqual(3, len(batch))
        source, source_labels, target = batch
        self.assertEqual(torch.Size((16, 14, window_size)), source.shape)
        self.assertEqual(torch.Size((16,)), source_labels.shape)
        self.assertEqual(torch.Size((16, 14, window_size)), target.shape)

    def test_val_batch_structure(self):
        window_size = self.dataset.target.window_size
        val_source_loader, val_target_loader, _ = self.dataset.val_dataloader()
        self.assertIsInstance(val_source_loader.dataset, TensorDataset)
        self.assertIsInstance(val_target_loader.dataset, TensorDataset)
        self._assert_val_test_batch_structure(val_source_loader, window_size)
        self._assert_val_test_batch_structure(val_target_loader, window_size)

    def test_test_batch_structure(self):
        window_size = self.dataset.target.window_size
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self.assertIsInstance(test_source_loader.dataset, TensorDataset)
        self.assertIsInstance(test_target_loader.dataset, TensorDataset)
        self._assert_val_test_batch_structure(test_source_loader, window_size)
        self._assert_val_test_batch_structure(test_target_loader, window_size)

    def _assert_val_test_batch_structure(self, loader, window_size):
        batch = next(iter(loader))
        self.assertEqual(2, len(batch))
        features, labels = batch
        self.assertEqual(torch.Size((16, 14, window_size)), features.shape)
        self.assertEqual(torch.Size((16,)), labels.shape)

    def test_truncation_passed_correctly(self):
        dataset = datasets.DomainAdaptionDataModule(
            3, 2, 16, percent_broken=0.2, percent_fail_runs=0.5
        )
        self.assertIsNone(dataset.source.percent_broken)
        self.assertIsNone(dataset.source.percent_fail_runs)
        self.assertEqual(0.2, dataset.target.percent_broken)
        self.assertEqual(0.5, dataset.target.percent_fail_runs)
        self.assertEqual(0.2, dataset.target_truncated.percent_broken)
        self.assertEqual(0.5, dataset.target_truncated.percent_fail_runs)

    def test_distance_mode_passed_correctly(self):
        dataset = datasets.PretrainingAdaptionDataModule(
            3, 2, 1000, 16, distance_mode="labeled"
        )
        data_loader = dataset.train_dataloader()
        self.assertEqual(dataset.distance_mode, data_loader.dataset.mode)


class TestPretrainingDataModuleFullData(
    CmapssTestTemplate, PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        self.dataset = datasets.PretrainingAdaptionDataModule(
            3, 2, num_samples=10000, batch_size=16, min_distance=2
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 3
        self.window_size = self.dataset.target_loader.window_size

    def test_target_val_truncation(self):
        with self.subTest(truncation=False):
            dataset = datasets.PretrainingAdaptionDataModule(
                3, 2, num_samples=10000, batch_size=16
            )
            self.assertFalse(dataset.target_loader.truncate_val)

        with self.subTest(truncation=True):
            dataset = datasets.PretrainingAdaptionDataModule(
                3, 2, num_samples=10000, batch_size=16, truncate_target_val=True
            )
            self.assertTrue(dataset.target_loader.truncate_val)

    def test_override_window_size(self):
        dataset = datasets.PretrainingAdaptionDataModule(
            3, 2, num_samples=10000, batch_size=16, window_size=40
        )
        dataset.prepare_data()
        dataset.setup()
        train_loader = dataset.train_dataloader()

        anchors, queries, _, _ = next(iter(train_loader))
        self.assertEqual(40, anchors.shape[2])
        self.assertEqual(40, queries.shape[2])

    def test_truncation_passed_correctly(self):
        dataset = datasets.PretrainingAdaptionDataModule(
            3, 2, 10000, 16, percent_broken=0.2, percent_fail_runs=0.5
        )
        self.assertIsNone(dataset.source.percent_broken)
        self.assertIsNone(dataset.source.percent_fail_runs)
        self.assertEqual(0.2, dataset.target.percent_broken)
        self.assertEqual(0.5, dataset.target.percent_fail_runs)


class TestPretrainingDataModuleLowData(
    CmapssTestTemplate, PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        self.dataset = datasets.PretrainingAdaptionDataModule(
            1, 3, num_samples=10000, batch_size=16, percent_broken=0.2
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 3
        self.window_size = self.dataset.target_loader.window_size
