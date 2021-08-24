import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import TensorDataset

import datasets
from datasets import cmapss, loader
from tests.dataset_tests.templates import CmapssTestTemplate


class TestCMAPSS(CmapssTestTemplate, unittest.TestCase):
    def test_data(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16)
            dataset.prepare_data()
            dataset.setup()
            for split in ["dev", "val", "test"]:
                with self.subTest(fd=n, split=split):
                    features, targets = dataset.data[split]
                    self.assertEqual(win, features.shape[2])
                    self.assertEqual(len(features), len(targets))
                    self.assertEqual(torch.float32, features.dtype)
                    self.assertEqual(torch.float32, targets.dtype)

    def test_override_window_size(self):
        window_size = 40
        for n in range(1, 5):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16, window_size=window_size)
            self.assertEqual(window_size, dataset.window_size)
            self.assertEqual(window_size, dataset._loader.window_size)

    def test_default_window_size(self):
        window_sizes = [30, 20, 30, 15]
        for n, win in enumerate(window_sizes, start=1):
            dataset = cmapss.CMAPSSDataModule(n, batch_size=16)
            self.assertEqual(win, dataset.window_size)
            self.assertEqual(win, dataset._loader.window_size)

    def test_feature_select(self):
        feature_idx = [4, 9, 10, 13, 14, 15, 22]
        dataset = cmapss.CMAPSSDataModule(1, batch_size=16, feature_select=feature_idx)
        self.assertListEqual(feature_idx, dataset.feature_select)
        self.assertListEqual(feature_idx, dataset._loader.feature_select)

    def test_default_feature_select(self):
        dataset = cmapss.CMAPSSDataModule(1, batch_size=16)
        self.assertListEqual(dataset._loader.DEFAULT_CHANNELS, dataset.feature_select)
        self.assertListEqual(
            dataset._loader.DEFAULT_CHANNELS, dataset._loader.feature_select
        )

    def test_truncation_functions(self):
        full_dataset = cmapss.CMAPSSDataModule(fd=1, batch_size=4, window_size=30)
        full_dataset.prepare_data()
        full_dataset.setup()

        dataset = cmapss.CMAPSSDataModule(
            fd=1, batch_size=4, window_size=30, percent_fail_runs=0.8
        )
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(
            len(full_dataset.data["dev"][0]), len(dataset.data["dev"][0])
        )
        self.assertEqual(len(full_dataset.data["val"][0]), len(dataset.data["val"][0]))
        self.assertEqual(
            len(full_dataset.data["test"][0]), len(dataset.data["test"][0])
        )

        dataset = cmapss.CMAPSSDataModule(
            fd=1, batch_size=4, window_size=30, percent_broken=0.2
        )
        dataset.prepare_data()
        dataset.setup()
        self.assertGreater(
            len(full_dataset.data["dev"][0]), len(dataset.data["dev"][0])
        )
        self.assertAlmostEqual(
            0.2,
            len(dataset.data["dev"][0]) / len(full_dataset.data["dev"][0]),
            delta=0.01,
        )
        self.assertEqual(
            len(full_dataset.data["val"][0]), len(dataset.data["val"][0])
        )  # Val dataset_tests not truncated
        self.assertEqual(
            len(full_dataset.data["test"][0]), len(dataset.data["test"][0])
        )  # Test dataset_tests not truncated
        self.assertFalse(
            torch.any(dataset.data["dev"][1] == 1)
        )  # No failure dataset_tests in truncated dataset_tests
        self.assertEqual(
            full_dataset.data["dev"][1][0], dataset.data["dev"][1][0]
        )  # First target has to be equal

    def test_truncation_passed_correctly(self):
        dataset = cmapss.CMAPSSDataModule(
            1, 4, percent_broken=0.2, percent_fail_runs=0.5
        )
        self.assertEqual(dataset.percent_broken, dataset._loader.percent_broken)
        self.assertEqual(dataset.percent_fail_runs, dataset._loader.percent_fail_runs)

    def test_from_loader(self):
        cmapss_loader = loader.CMAPSSLoader(3, 40, 130, 0.2, 0.5, truncate_val=True)
        cmapss_dataset = cmapss.CMAPSSDataModule.from_loader(cmapss_loader, 128)
        self.assertEqual(cmapss_loader.fd, cmapss_dataset.fd)
        self.assertEqual(cmapss_loader.window_size, cmapss_dataset.window_size)
        self.assertEqual(cmapss_loader.max_rul, cmapss_dataset.max_rul)
        self.assertEqual(cmapss_loader.percent_broken, cmapss_dataset.percent_broken)
        self.assertEqual(
            cmapss_loader.percent_fail_runs, cmapss_dataset.percent_fail_runs
        )
        self.assertEqual(cmapss_loader.feature_select, cmapss_dataset.feature_select)
        self.assertEqual(cmapss_loader.truncate_val, cmapss_dataset.truncate_val)
        self.assertEqual(128, cmapss_dataset.batch_size)

    def test_empty_dataset(self):
        dataset = cmapss.CMAPSSDataModule(
            1, 4, percent_broken=0.2, percent_fail_runs=0.0
        )
        dataset.setup()
        dataset = cmapss.CMAPSSDataModule(
            1, 4, percent_broken=0.0, percent_fail_runs=0.5
        )
        dataset.setup()


class DummyCMAPSS:
    def __init__(self, length):
        self.window_size = 30
        self.max_rul = 125
        self.data = {
            "dev": (
                [torch.zeros(length, self.window_size, 5)],
                [torch.clamp_max(torch.arange(length, 0, step=-1), 125)],
            ),
        }

    def load_split(self, split):
        assert split == "dev", "Can only use dev data."
        return self.data["dev"]


class DummyCMAPSSShortRuns:
    """Contains runs that are too short with zero features to distinguish them."""

    def __init__(self):
        self.window_size = 30
        self.max_rul = 125
        self.data = {
            "dev": (
                [
                    torch.ones(100, self.window_size, 5)
                    * torch.arange(1, 101).view(100, 1, 1),  # normal run
                    torch.zeros(2, self.window_size, 5),  # too short run
                    torch.ones(100, self.window_size, 5)
                    * torch.arange(1, 101).view(100, 1, 1),  # normal run
                    torch.zeros(1, self.window_size, 5),  # empty run
                ],
                [
                    torch.clamp_max(torch.arange(100, 0, step=-1), 125),
                    torch.ones(2) * 500,
                    torch.clamp_max(torch.arange(100, 0, step=-1), 125),
                    torch.ones(1) * 500,
                ],
            ),
        }

    def load_split(self, split):
        assert split == "dev", "Can only use dev data."
        return self.data["dev"]


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.length = 300
        self.cmapss_normal = DummyCMAPSS(self.length)
        self.cmapss_short = DummyCMAPSSShortRuns()
        self.fd1 = loader.CMAPSSLoader(1)
        self.fd3 = loader.CMAPSSLoader(3)

    def test_get_pair_idx_piecewise(self):
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        middle_idx = self.length // 2
        for _ in range(512):
            run, anchor_idx, query_idx, distance, _ = data._get_pair_idx_piecewise()
            self.assertEqual(middle_idx, len(run) // 2)
            if anchor_idx < middle_idx:
                self.assertEqual(0, distance)
            else:
                self.assertLessEqual(0, distance)

    def test_get_pair_idx_linear(self):
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            run, anchor_idx, query_idx, distance, _ = data._get_pair_idx()
            self.assertLess(0, distance)
            self.assertGreaterEqual(125, distance)

    def test_get_labeled_pair_idx(self):
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            run, anchor_idx, query_idx, distance, _ = data._get_labeled_pair_idx()
            self.assertEqual(self.length, len(run))
            expected_distance = data._labels[0][anchor_idx] - data._labels[0][query_idx]
            self.assertLessEqual(0, distance)
            self.assertEqual(expected_distance, distance)

    def test_pair_func_selection(self):
        with self.subTest("default"):
            data = datasets.cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True
            )
            self.assertEqual(data._get_pair_idx, data._get_pair_func)
        with self.subTest("piecewise"):
            data = datasets.cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, mode="linear"
            )
            self.assertEqual(data._get_pair_idx, data._get_pair_func)
        with self.subTest("piecewise"):
            data = datasets.cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, mode="piecewise"
            )
            self.assertEqual(data._get_pair_idx_piecewise, data._get_pair_func)
        with self.subTest("labeled"):
            data = datasets.cmapss.PairedCMAPSS(
                [self.cmapss_normal], "dev", 512, 1, True, mode="labeled"
            )
            self.assertEqual(data._get_labeled_pair_idx, data._get_pair_func)

    def test_sampled_data(self):
        fixed_idx = [0, 60, 80, 1, 55, 99]  # two samples with run, anchor and query idx
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_short], "dev", 2, 2)
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
        data = datasets.cmapss.PairedCMAPSS([self.cmapss_short], "dev", 512, 2)
        for run, labels in zip(data._features, data._labels):
            self.assertTrue((run >= 1).all())
            self.assertTrue((labels < 500).all())
            self.assertLess(2, len(run))

    def test_determinisim(self):
        with self.subTest("non-deterministic"):
            data = datasets.cmapss.PairedCMAPSS([self.cmapss_short], "dev", 512, 2)
            self.assertTrue(self._two_epochs_different(data))
        with self.subTest("non-deterministic"):
            data = datasets.cmapss.PairedCMAPSS(
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
        dataset = datasets.cmapss.PairedCMAPSS(
            [self.cmapss_short], "dev", 512, min_distance=30
        )
        pairs = self._get_pairs(dataset)
        distances = pairs[:, 1] - pairs[:, 0]
        self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
        self.assertTrue(np.all(distances >= 30))

    def test_build_pairs(self):
        for split in ["dev", "val"]:
            with self.subTest(split=split):
                paired_dataset = datasets.cmapss.PairedCMAPSS(
                    [self.fd1, self.fd3], split, 1000, 1
                )
                pairs = self._get_pairs(paired_dataset)
                self.assertTrue(
                    np.all(pairs[:, 0] < pairs[:, 1])
                )  # query always after anchor
                self.assertTrue(np.all(pairs[:, 3] <= 1))  # domain label is either 1
                self.assertTrue(np.all(pairs[:, 3] >= 0))  # or zero

    def test_domain_labels(self):
        dataset = datasets.cmapss.PairedCMAPSS(
            [self.cmapss_normal, self.cmapss_short], "dev", 512, min_distance=30
        )
        for _ in range(512):
            run, _, _, _, domain_idx = dataset._get_pair_idx_piecewise()
            if len(run) == self.length:
                self.assertEqual(0, domain_idx)  # First domain is self.length long
            else:
                self.assertEqual(1, domain_idx)  # Second is not

    def _get_pairs(self, paired):
        pairs = [paired._get_pair_idx_piecewise() for _ in range(paired.num_samples)]
        pairs = [
            (anchor_idx, query_idx, distance, domain_idx)
            for _, anchor_idx, query_idx, distance, domain_idx in pairs
        ]

        return np.array(pairs)


class TestAdaptionDataset(unittest.TestCase):
    def setUp(self):
        self.source = TensorDataset(torch.arange(100), torch.arange(100))
        self.target = TensorDataset(torch.arange(150), torch.arange(150))
        self.dataset = datasets.cmapss.AdaptionDataset(
            self.source,
            self.target,
        )

    def test_len(self):
        self.assertEqual(len(self.dataset.source), len(self.dataset))

    def test_source_target_shuffeled(self):
        for i in range(len(self.dataset)):
            source_one, label_one, target_one = self.dataset[i]
            source_another, label_another, target_another = self.dataset[i]
            self.assertEqual(source_one, source_another)
            self.assertEqual(label_one, label_another)
            self.assertNotEqual(target_one, target_another)

    def test_source_target_deterministic(self):
        dataset = datasets.cmapss.AdaptionDataset(
            self.source, self.target, deterministic=True
        )
        for i in range(len(dataset)):
            source_one, label_one, target_one = dataset[i]
            source_another, label_another, target_another = dataset[i]
            self.assertEqual(source_one, source_another)
            self.assertEqual(label_one, label_another)
            self.assertEqual(target_one, target_another)

    def test_source_sampled_completely(self):
        for i in range(len(self.dataset)):
            source, labels, _ = self.dataset[i]
            self.assertEqual(i, source.item())
            self.assertEqual(i, labels.item())
