import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from rul_datasets import cmapss, loader


class TestCMAPSS(unittest.TestCase):
    def setUp(self):
        self.mock_loader = mock.MagicMock(name="AbstractLoader")
        self.mock_loader.hparams = {"test": 0}
        self.mock_runs = [torch.zeros(1, 1, 1)], [torch.zeros(1)]
        self.mock_loader.load_split.return_value = self.mock_runs

    def test_created_correctly(self):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)

        self.assertIs(self.mock_loader, dataset.loader)
        self.assertEqual(16, dataset.batch_size)
        self.assertDictEqual({"test": 0, "batch_size": 16}, dataset.hparams)

    def test_prepare_data(self):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.prepare_data()

        self.mock_loader.prepare_data.assert_called_once()

    def test_setup(self):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()

        self.mock_loader.load_split.assert_has_calls(
            [mock.call("dev"), mock.call("val"), mock.call("test")]
        )
        mock_runs = tuple(torch.cat(r) for r in self.mock_runs)
        self.assertDictEqual(
            {"dev": mock_runs, "val": mock_runs, "test": mock_runs}, dataset.data
        )

    def test_empty_dataset(self):
        self.mock_loader.load_split.return_value = [], []
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=4)
        dataset.setup()

    @mock.patch(
        "rul_datasets.cmapss.CMAPSSDataModule.to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_train_dataloader(self, mock_to_dataset):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        dataloader = dataset.train_dataloader()

        mock_to_dataset.assert_called_once_with("dev")
        self.assertIs(mock_to_dataset.return_value, dataloader.dataset)
        self.assertEqual(16, dataloader.batch_size)
        self.assertIsInstance(dataloader.sampler, RandomSampler)
        self.assertTrue(dataloader.pin_memory)

    @mock.patch(
        "rul_datasets.cmapss.CMAPSSDataModule.to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_val_dataloader(self, mock_to_dataset):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        dataloader = dataset.val_dataloader()

        mock_to_dataset.assert_called_once_with("val")
        self.assertIs(mock_to_dataset.return_value, dataloader.dataset)
        self.assertEqual(16, dataloader.batch_size)
        self.assertIsInstance(dataloader.sampler, SequentialSampler)
        self.assertTrue(dataloader.pin_memory)

    @mock.patch(
        "rul_datasets.cmapss.CMAPSSDataModule.to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_test_dataloader(self, mock_to_dataset):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        dataloader = dataset.test_dataloader()

        mock_to_dataset.assert_called_once_with("test")
        self.assertIs(mock_to_dataset.return_value, dataloader.dataset)
        self.assertEqual(16, dataloader.batch_size)
        self.assertIsInstance(dataloader.sampler, SequentialSampler)
        self.assertTrue(dataloader.pin_memory)

    def test_train_batch_structure(self):
        self.mock_loader.load_split.return_value = (
            [torch.zeros(8, 14, 30)] * 4,
            [torch.zeros(8)] * 4,
        )
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        train_loader = dataset.train_dataloader()
        self._assert_batch_structure(train_loader)

    def test_val_batch_structure(self):
        self.mock_loader.load_split.return_value = (
            [torch.zeros(8, 14, 30)] * 4,
            [torch.zeros(8)] * 4,
        )
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        val_loader = dataset.val_dataloader()
        self._assert_batch_structure(val_loader)

    def test_test_batch_structure(self):
        self.mock_loader.load_split.return_value = (
            [torch.zeros(8, 14, 30)] * 4,
            [torch.zeros(8)] * 4,
        )
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        test_loader = dataset.test_dataloader()
        self._assert_batch_structure(test_loader)

    def _assert_batch_structure(self, loader):
        batch = next(iter(loader))
        self.assertEqual(2, len(batch))
        features, labels = batch
        self.assertEqual(torch.Size((16, 14, 30)), features.shape)
        self.assertEqual(torch.Size((16,)), labels.shape)

    def test_to_dataset(self):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        mock_data = {
            "dev": [torch.zeros(0)] * 2,
            "val": [torch.zeros(1)] * 2,
            "test": [torch.zeros(2)] * 2,
        }
        dataset.data = mock_data

        for i, split in enumerate(["dev", "val", "test"]):
            tensor_dataset = dataset.to_dataset(split)
            self.assertIsInstance(tensor_dataset, TensorDataset)
            self.assertEqual(i, len(tensor_dataset.tensors[0]))

    def test_check_compatability(self):
        dataset = cmapss.CMAPSSDataModule(self.mock_loader, batch_size=16)
        dataset.check_compatibility(dataset)
        self.mock_loader.check_compatibility.assert_called_once_with(self.mock_loader)
        self.assertRaises(
            ValueError,
            dataset.check_compatibility,
            cmapss.CMAPSSDataModule(self.mock_loader, batch_size=8),
        )


class DummyCMAPSS(loader.AbstractLoader):
    fd: int = 1
    window_size: int = 30
    max_rul: int = 125

    def __init__(self, length):
        self.data = {
            "dev": (
                [torch.zeros(length, self.window_size, 5)],
                [torch.clamp_max(torch.arange(length, 0, step=-1), 125)],
            ),
        }

    def prepare_data(self):
        pass

    def load_split(self, split):
        assert split == "dev", "Can only use dev data."
        return self.data["dev"]


@dataclass
class DummyCMAPSSShortRuns(loader.AbstractLoader):
    """Contains runs that are too short with zero features to distinguish them."""

    fd: int = 1
    window_size: int = 30
    max_rul: int = 125
    data = {
        "dev": (
            [
                torch.ones(100, window_size, 5)
                * torch.arange(1, 101).view(100, 1, 1),  # normal run
                torch.zeros(2, window_size, 5),  # too short run
                torch.ones(100, window_size, 5)
                * torch.arange(1, 101).view(100, 1, 1),  # normal run
                torch.zeros(1, window_size, 5),  # empty run
            ],
            [
                torch.clamp_max(torch.arange(100, 0, step=-1), 125),
                torch.ones(2) * 500,
                torch.clamp_max(torch.arange(100, 0, step=-1), 125),
                torch.ones(1) * 500,
            ],
        ),
    }

    def prepare_data(self):
        pass

    def load_split(self, split):
        assert split == "dev", "Can only use dev data."
        return self.data["dev"]


class TestPairedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for fd in range(1, 5):
            loader.CMAPSSLoader(fd).prepare_data()
        cls.fd1 = loader.CMAPSSLoader(1)
        cls.fd3 = loader.CMAPSSLoader(3)

    def setUp(self):
        self.length = 300
        self.cmapss_normal = DummyCMAPSS(self.length)
        self.cmapss_short = DummyCMAPSSShortRuns()

    def test_get_pair_idx_piecewise(self):
        data = cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        middle_idx = self.length // 2
        for _ in range(512):
            run_idx, anchor_idx, query_idx, distance, _ = data._get_pair_idx_piecewise()
            run = data._features[run_idx]
            self.assertEqual(middle_idx, len(run) // 2)
            if anchor_idx < middle_idx:
                self.assertEqual(0, distance)
            else:
                self.assertLessEqual(0, distance)

    def test_get_pair_idx_linear(self):
        data = cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            run, anchor_idx, query_idx, distance, _ = data._get_pair_idx()
            self.assertLess(0, distance)
            self.assertGreaterEqual(125, distance)

    def test_get_labeled_pair_idx(self):
        data = cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            run_idx, anchor_idx, query_idx, distance, _ = data._get_labeled_pair_idx()
            run = data._features[run_idx]
            self.assertEqual(self.length, len(run))
            expected_distance = data._labels[0][anchor_idx] - data._labels[0][query_idx]
            self.assertLessEqual(0, distance)
            self.assertEqual(expected_distance, distance)

    def test_pair_func_selection(self):
        with self.subTest("default"):
            data = cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
            self.assertEqual(data._get_pair_idx, data._get_pair_func)
        with self.subTest("piecewise"):
            data = cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, mode="linear"
            )
            self.assertEqual(data._get_pair_idx, data._get_pair_func)
        with self.subTest("piecewise"):
            data = cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, mode="piecewise"
            )
            self.assertEqual(data._get_pair_idx_piecewise, data._get_pair_func)
        with self.subTest("labeled"):
            data = cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, mode="labeled"
            )
            self.assertEqual(data._get_labeled_pair_idx, data._get_pair_func)

    def test_sampled_data(self):
        fixed_idx = [0, 60, 80, 1, 55, 99]  # two samples with run, anchor and query idx
        data = cmapss.PairedCMAPSS([self.cmapss_short], "dev", 2, 2)
        data._rng = mock.MagicMock()
        data._rng.integers = mock.MagicMock(side_effect=fixed_idx)
        for i, sample in enumerate(data):
            idx = 3 * i
            expected_run = data._features[fixed_idx[idx]]
            expected_anchor = expected_run[fixed_idx[idx + 1]]
            expected_query = expected_run[fixed_idx[idx + 2]]
            expected_distance = min(125, fixed_idx[idx + 2] - fixed_idx[idx + 1]) / 125
            expected_domain_idx = 0
            self.assertEqual(0, torch.dist(expected_anchor, sample[0]))
            self.assertEqual(0, torch.dist(expected_query, sample[1]))
            self.assertAlmostEqual(expected_distance, sample[2].item())
            self.assertEqual(expected_domain_idx, sample[3].item())

    def test_discarding_too_short_runs(self):
        data = cmapss.PairedCMAPSS([self.cmapss_short], "dev", 512, 2)
        for run, labels in zip(data._features, data._labels):
            self.assertTrue((run >= 1).all())
            self.assertTrue((labels < 500).all())
            self.assertLess(2, len(run))

    def test_determinisim(self):
        with self.subTest("non-deterministic"):
            data = cmapss.PairedCMAPSS([self.cmapss_short], "dev", 512, 2)
            self.assertTrue(self._two_epochs_different(data))
        with self.subTest("non-deterministic"):
            data = cmapss.PairedCMAPSS(
                [self.cmapss_short], "dev", 512, 2, deterministic=True
            )
            self.assertFalse(self._two_epochs_different(data))

    def _two_epochs_different(self, data):
        first_epoch = [samples for samples in data]
        second_epoch = [samples for samples in data]
        different = False
        for first_samples, second_samples in zip(first_epoch, second_epoch):
            for first, second in zip(first_samples, second_samples):
                if torch.dist(first, second) > 0:
                    different = True

        return different

    def test_min_distance(self):
        for mode in ["linear", "piecewise", "labeled"]:
            with self.subTest(mode):
                dataset = cmapss.PairedCMAPSS(
                    [self.cmapss_short], "dev", 512, min_distance=30, mode=mode
                )
                pairs = self._all_get_pairs(dataset)
                distances = pairs[:, 1] - pairs[:, 0]
                self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
                self.assertTrue(np.all(distances >= 30))

    def test_get_pairs(self):
        for split in ["dev", "val"]:
            for mode in ["linear", "piecewise", "labeled"]:
                with self.subTest(split=split, mode=mode):
                    paired_dataset = cmapss.PairedCMAPSS(
                        [self.fd1, self.fd3], split, 1000, 1, mode=mode
                    )
                    pairs = self._all_get_pairs(paired_dataset)
                    self.assertTrue(
                        np.all(pairs[:, 0] < pairs[:, 1])
                    )  # query always after anchor
                    self.assertTrue(
                        np.all(pairs[:, 3] <= 1)
                    )  # domain label is either 1
                    self.assertTrue(np.all(pairs[:, 3] >= 0))  # or zero

    def _all_get_pairs(self, paired):
        pairs = [paired._get_pair_func() for _ in range(paired.num_samples)]
        pairs = [
            (anchor_idx, query_idx, distance, domain_idx)
            for _, anchor_idx, query_idx, distance, domain_idx in pairs
        ]

        return np.array(pairs)

    def test_domain_labels(self):
        dataset = cmapss.PairedCMAPSS(
            [self.cmapss_normal, self.cmapss_short], "dev", 512, min_distance=30
        )
        for _ in range(512):
            run_idx, _, _, _, domain_idx = dataset._get_pair_idx_piecewise()
            run = dataset._features[run_idx]
            if len(run) == self.length:
                self.assertEqual(0, domain_idx)  # First domain is self.length long
            else:
                self.assertEqual(1, domain_idx)  # Second is not

    def test_no_determinism_in_multiprocessing(self):
        dataset = cmapss.PairedCMAPSS(
            [self.cmapss_normal, self.cmapss_short], "dev", 100, 1, deterministic=True
        )
        dataloader = DataLoader(dataset, num_workers=2)
        with self.assertRaises(RuntimeError):
            for _ in dataloader:
                pass

    def test_no_duplicate_batches_in_multiprocessing(self):
        dataset = cmapss.PairedCMAPSS(
            [self.cmapss_normal, self.cmapss_short], "dev", 100, 1
        )
        dataloader = DataLoader(dataset, batch_size=10, num_workers=2)
        batches = [batch for batch in dataloader]
        are_duplicated = []
        for b0, b1 in zip(batches[::2], batches[1::2]):
            are_duplicated.append(self._is_same_batch(b0, b1))
        self.assertFalse(all(are_duplicated))

    def test_no_repeating_epochs_in_multiprocessing(self):
        dataset = cmapss.PairedCMAPSS(
            [self.cmapss_normal, self.cmapss_short], "dev", 100, 1
        )
        dataloader = DataLoader(dataset, batch_size=10, num_workers=2)
        epoch0 = [batch for batch in dataloader]
        epoch1 = [batch for batch in dataloader]
        for b0, b1 in zip(epoch0, epoch1):
            self.assertFalse(self._is_same_batch(b0, b1))

    def _is_same_batch(self, b0, b1):
        return all(torch.dist(a, b) == 0.0 for a, b in zip(b0, b1))

    def test_compatability_check(self):
        self.assertRaises(
            ValueError,
            cmapss.PairedCMAPSS,
            [DummyCMAPSSShortRuns(), DummyCMAPSSShortRuns(window_size=20)],
            "dev",
            1000,
            1,
        )
