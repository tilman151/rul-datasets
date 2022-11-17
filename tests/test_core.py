import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from rul_datasets import core, reader


class TestRulDataModule(unittest.TestCase):
    def setUp(self):
        self.mock_loader = mock.MagicMock(name="AbstractLoader")
        self.mock_loader.hparams = {
            "test": 0,
            "window_size": 30,
        }
        self.mock_runs = [torch.zeros(1, 1, 1)], [torch.zeros(1)]
        self.mock_loader.load_split.return_value = self.mock_runs

    def test_created_correctly(self):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)

        self.assertIs(self.mock_loader, dataset.reader)
        self.assertEqual(16, dataset.batch_size)
        self.assertDictEqual(
            {"test": 0, "batch_size": 16, "window_size": 30, "feature_extractor": None},
            dataset.hparams,
        )

    def test_created_correctly_with_feature_extractor(self):
        fe = lambda x: np.mean(x, axis=1)
        dataset = core.RulDataModule(
            self.mock_loader, batch_size=16, feature_extractor=fe, window_size=2
        )

        self.assertIs(self.mock_loader, dataset.reader)
        self.assertEqual(16, dataset.batch_size)
        self.assertDictEqual(
            {
                "test": 0,
                "batch_size": 16,
                "window_size": 2,
                "feature_extractor": str(fe),
            },
            dataset.hparams,
        )

    def test_prepare_data(self):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.prepare_data()

        self.mock_loader.prepare_data.assert_called_once()

    def test_setup(self):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.setup()

        self.mock_loader.load_split.assert_has_calls(
            [mock.call("dev"), mock.call("val"), mock.call("test")]
        )
        mock_runs = tuple(torch.cat(r) for r in self.mock_runs)
        self.assertDictEqual(
            {"dev": mock_runs, "val": mock_runs, "test": mock_runs}, dataset._data
        )

    def test_empty_dataset(self):
        self.mock_loader.load_split.return_value = [], []
        dataset = core.RulDataModule(self.mock_loader, batch_size=4)
        dataset.setup()

    @mock.patch(
        "rul_datasets.core.RulDataModule.to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_train_dataloader(self, mock_to_dataset):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        dataloader = dataset.train_dataloader()

        mock_to_dataset.assert_called_once_with("dev")
        self.assertIs(mock_to_dataset.return_value, dataloader.dataset)
        self.assertEqual(16, dataloader.batch_size)
        self.assertIsInstance(dataloader.sampler, RandomSampler)
        self.assertTrue(dataloader.pin_memory)

    @mock.patch(
        "rul_datasets.core.RulDataModule.to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_val_dataloader(self, mock_to_dataset):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        dataloader = dataset.val_dataloader()

        mock_to_dataset.assert_called_once_with("val")
        self.assertIs(mock_to_dataset.return_value, dataloader.dataset)
        self.assertEqual(16, dataloader.batch_size)
        self.assertIsInstance(dataloader.sampler, SequentialSampler)
        self.assertTrue(dataloader.pin_memory)

    @mock.patch(
        "rul_datasets.core.RulDataModule.to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_test_dataloader(self, mock_to_dataset):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
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
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        train_loader = dataset.train_dataloader()
        self._assert_batch_structure(train_loader)

    def test_val_batch_structure(self):
        self.mock_loader.load_split.return_value = (
            [torch.zeros(8, 14, 30)] * 4,
            [torch.zeros(8)] * 4,
        )
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.setup()
        val_loader = dataset.val_dataloader()
        self._assert_batch_structure(val_loader)

    def test_test_batch_structure(self):
        self.mock_loader.load_split.return_value = (
            [torch.zeros(8, 14, 30)] * 4,
            [torch.zeros(8)] * 4,
        )
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
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
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        mock_data = {
            "dev": [torch.zeros(0)] * 2,
            "val": [torch.zeros(1)] * 2,
            "test": [torch.zeros(2)] * 2,
        }
        dataset._data = mock_data

        for i, split in enumerate(["dev", "val", "test"]):
            tensor_dataset = dataset.to_dataset(split)
            self.assertIsInstance(tensor_dataset, TensorDataset)
            self.assertEqual(i, len(tensor_dataset.tensors[0]))

    def test_check_compatability(self):
        fe = lambda x: np.mean(x, axis=2)
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        other = core.RulDataModule(
            self.mock_loader, batch_size=16, feature_extractor=fe, window_size=2
        )
        dataset.check_compatibility(dataset)
        self.mock_loader.check_compatibility.assert_called_once_with(self.mock_loader)
        self.assertRaises(
            ValueError,
            dataset.check_compatibility,
            core.RulDataModule(self.mock_loader, batch_size=8),
        )
        self.assertRaises(
            ValueError,
            dataset.check_compatibility,
            other,
        )
        self.assertRaises(
            ValueError,
            other.check_compatibility,
            core.RulDataModule(
                self.mock_loader, batch_size=16, feature_extractor=fe, window_size=3
            ),
        )

    def test_is_mutually_exclusive(self):
        dataset = core.RulDataModule(self.mock_loader, batch_size=16)
        dataset.is_mutually_exclusive(dataset)
        self.mock_loader.is_mutually_exclusive.assert_called_once_with(dataset.reader)

    def test_feature_extractor(self):
        self.mock_loader.load_split.return_value = (
            [torch.zeros(8, 14, 30) + torch.arange(8)[:, None, None]],
            [torch.arange(8)],
        )
        feature_extractor = lambda x: np.mean(x, axis=1)
        dataset = core.RulDataModule(
            self.mock_loader,
            batch_size=16,
            feature_extractor=feature_extractor,
            window_size=2,
        )
        dataset.setup()

        dev_data = dataset.to_dataset("dev")
        self.assertEqual(len(dev_data), 7)
        for i, (feat, targ) in enumerate(dev_data):
            self.assertEqual(feat.shape, torch.Size([14, 2]))
            self.assertTrue(
                torch.dist(torch.arange(i, i + 2)[None, :].repeat(14, 1), feat) == 0
            )
            self.assertEqual(targ, i + 1)  # targets start window_size + 1 steps later


class DummyRul(reader.AbstractReader):
    fd: int = 1
    window_size: int = 30
    max_rul: int = 125

    def __init__(self, length):
        self.data = {
            "dev": (
                [torch.zeros(length, self.window_size, 5)],
                [torch.clamp_max(torch.arange(length, 0, step=-1), 125)],
            ),
            "val": (
                [torch.zeros(100, self.window_size, 5)],
                [torch.clamp_max(torch.arange(100, 0, step=-1), 125)],
            ),
        }

    @property
    def fds(self):
        return [1]

    def default_window_size(self, fd):
        return self.window_size

    def check_compatibility(self, other) -> None:
        pass

    def prepare_data(self):
        pass

    def load_complete_split(self, split):
        return self.data[split]

    def load_split(self, split):
        return self.load_complete_split(split)


@dataclass
class DummyRulShortRuns(reader.AbstractReader):
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

    @property
    def fds(self):
        return [1]

    def default_window_size(self, fd):
        return self.window_size

    def check_compatibility(self, other) -> None:
        pass

    def prepare_data(self):
        pass

    def load_complete_split(self, split):
        if not split == "dev":
            raise ValueError(f"DummyRulShortRuns does not have a '{split}' split")

        return self.data["dev"]

    def load_split(self, split):
        return self.load_complete_split(split)


@pytest.fixture(scope="module")
def length():
    return 300


@pytest.fixture
def cmapss_normal(length):
    return DummyRul(length)


@pytest.fixture
def cmapss_short():
    return DummyRulShortRuns()


class TestPairedDataset:
    def test_get_pair_idx_piecewise(self, cmapss_normal, length):
        data = core.PairedRulDataset([cmapss_normal], "dev", 512, 1, True)
        middle_idx = length // 2
        for _ in range(512):
            run_idx, anchor_idx, query_idx, distance, _ = data._get_pair_idx_piecewise()
            run = data._features[run_idx]
            assert middle_idx == (len(run) // 2)
            if anchor_idx < middle_idx:
                assert 0 == distance
            else:
                assert 0 <= distance

    def test_get_pair_idx_linear(self, cmapss_normal):
        data = core.PairedRulDataset([cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            run, anchor_idx, query_idx, distance, _ = data._get_pair_idx()
            assert 0 < distance
            assert 125 >= distance

    def test_get_labeled_pair_idx(self, cmapss_normal, length):
        data = core.PairedRulDataset([cmapss_normal], "dev", 512, 1, True)
        for _ in range(512):
            run_idx, anchor_idx, query_idx, distance, _ = data._get_labeled_pair_idx()
            run = data._features[run_idx]
            assert length == len(run)
            expected_distance = data._labels[0][anchor_idx] - data._labels[0][query_idx]
            assert 0 <= distance
            assert expected_distance == distance

    @pytest.mark.parametrize(
        ["mode", "expected_func"],
        [
            ("linear", "_get_pair_idx"),
            ("piecewise", "_get_pair_idx_piecewise"),
            ("labeled", "_get_labeled_pair_idx"),
        ],
    )
    def test_pair_func_selection(self, cmapss_normal, mode, expected_func):
        data = core.PairedRulDataset([cmapss_normal], "dev", 512, 1, True, mode=mode)
        expected_func = getattr(data, expected_func)
        assert expected_func == data._get_pair_func

    def test_sampled_data(self, cmapss_short):
        fixed_idx = [0, 60, 80, 1, 55, 99]  # two samples with run, anchor and query idx
        data = core.PairedRulDataset([cmapss_short], "dev", 2, 2)
        data._rng = mock.MagicMock()
        data._rng.integers = mock.MagicMock(side_effect=fixed_idx)
        for i, sample in enumerate(data):
            idx = 3 * i
            expected_run = data._features[fixed_idx[idx]]
            expected_anchor = expected_run[fixed_idx[idx + 1]]
            expected_query = expected_run[fixed_idx[idx + 2]]
            expected_distance = min(125, fixed_idx[idx + 2] - fixed_idx[idx + 1]) / 125
            expected_domain_idx = 0
            assert 0 == torch.dist(expected_anchor, sample[0])
            assert 0 == torch.dist(expected_query, sample[1])
            npt.assert_almost_equal(expected_distance, sample[2].item())
            assert expected_domain_idx == sample[3].item()

    def test_discarding_too_short_runs(self, cmapss_short):
        data = core.PairedRulDataset([cmapss_short], "dev", 512, 2)
        for run, labels in zip(data._features, data._labels):
            assert (run >= 1).all()
            assert (labels < 500).all()
            assert 2 < len(run)

    @pytest.mark.parametrize("deterministic", [True, False])
    def test_determinisim(self, cmapss_short, deterministic):
        data = core.PairedRulDataset(
            [cmapss_short], "dev", 512, 2, deterministic=deterministic
        )
        assert deterministic != self._two_epochs_different(data)

    @staticmethod
    def _two_epochs_different(data):
        first_epoch = [samples for samples in data]
        second_epoch = [samples for samples in data]
        different = False
        for first_samples, second_samples in zip(first_epoch, second_epoch):
            for first, second in zip(first_samples, second_samples):
                if torch.dist(first, second) > 0:
                    different = True

        return different

    @pytest.mark.parametrize("mode", ["linear", "piecewise", "labeled"])
    def test_min_distance(self, cmapss_short, mode):
        dataset = core.PairedRulDataset(
            [cmapss_short], "dev", 512, min_distance=30, mode=mode
        )
        pairs = self._all_get_pairs(dataset)
        distances = pairs[:, 1] - pairs[:, 0]
        assert np.all(pairs[:, 0] < pairs[:, 1])
        assert np.all(distances >= 30)

    @pytest.mark.parametrize("split", ["dev", "val"])
    @pytest.mark.parametrize("mode", ["linear", "piecewise", "labeled"])
    def test_get_pairs(self, split, mode, cmapss_normal):
        paired_dataset = core.PairedRulDataset(
            [cmapss_normal],
            split,
            1000,
            1,
            mode=mode,
        )
        pairs = self._all_get_pairs(paired_dataset)
        assert np.all(pairs[:, 0] < pairs[:, 1])  # query always after anchor
        assert np.all(pairs[:, 3] <= 1)  # domain label is either 1
        assert np.all(pairs[:, 3] >= 0)  # or zero

    @staticmethod
    def _all_get_pairs(paired):
        pairs = [paired._get_pair_func() for _ in range(paired.num_samples)]
        pairs = [
            (anchor_idx, query_idx, distance, domain_idx)
            for _, anchor_idx, query_idx, distance, domain_idx in pairs
        ]

        return np.array(pairs)

    def test_domain_labels(self, cmapss_normal, cmapss_short, length):
        dataset = core.PairedRulDataset(
            [cmapss_normal, cmapss_short], "dev", 512, min_distance=30
        )
        for _ in range(512):
            run_idx, _, _, _, domain_idx = dataset._get_pair_idx_piecewise()
            run = dataset._features[run_idx]
            if len(run) == length:
                assert 0 == domain_idx  # First domain is self.length long
            else:
                assert 1 == domain_idx  # Second is not

    def test_no_determinism_in_multiprocessing(self, cmapss_normal, cmapss_short):
        dataset = core.PairedRulDataset(
            [cmapss_normal, cmapss_short], "dev", 100, 1, deterministic=True
        )
        dataloader = DataLoader(dataset, num_workers=2)
        with pytest.raises(RuntimeError):
            for _ in dataloader:
                pass

    def test_no_duplicate_batches_in_multiprocessing(self, cmapss_normal, cmapss_short):
        dataset = core.PairedRulDataset([cmapss_normal, cmapss_short], "dev", 100, 1)
        dataloader = DataLoader(dataset, batch_size=10, num_workers=2)
        batches = [batch for batch in dataloader]
        are_duplicated = []
        for b0, b1 in zip(batches[::2], batches[1::2]):
            are_duplicated.append(self._is_same_batch(b0, b1))
        assert not all(are_duplicated)

    def test_no_repeating_epochs_in_multiprocessing(self, cmapss_normal, cmapss_short):
        dataset = core.PairedRulDataset([cmapss_normal, cmapss_short], "dev", 100, 1)
        dataloader = DataLoader(dataset, batch_size=10, num_workers=2)
        epoch0 = [batch for batch in dataloader]
        epoch1 = [batch for batch in dataloader]
        for b0, b1 in zip(epoch0, epoch1):
            assert not self._is_same_batch(b0, b1)

    @staticmethod
    def _is_same_batch(b0, b1):
        return all(torch.dist(a, b) == 0.0 for a, b in zip(b0, b1))

    def test_compatability_check(self):
        mock_check_compat = mock.MagicMock(name="check_compatibility")
        loaders = [DummyRulShortRuns(), DummyRulShortRuns(window_size=20)]
        for lod in loaders:
            lod.check_compatibility = mock_check_compat

        core.PairedRulDataset(loaders, "dev", 1000, 1)

        assert 2 == mock_check_compat.call_count
