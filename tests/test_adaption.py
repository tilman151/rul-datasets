import unittest
import warnings
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, TensorDataset

from rul_datasets import adaption, core
from rul_datasets.reader import DummyReader
from tests.templates import PretrainingDataModuleTemplate


class TestDomainAdaptionDataModule(unittest.TestCase):
    def setUp(self):
        source_mock_runs = [np.random.randn(16, 14, 1)] * 3, [np.random.rand(16)] * 3
        self.source_loader = mock.MagicMock(name="CMAPSSLoader")
        self.source_loader.fd = 3
        self.source_loader.percent_fail_runs = None
        self.source_loader.percent_broken = None
        self.source_loader.window_size = 1
        self.source_loader.max_rul = 125
        self.source_loader.hparams = {
            "fd": self.source_loader.fd,
            "window_size": self.source_loader.window_size,
        }
        self.source_loader.load_split.return_value = source_mock_runs
        self.source_data = core.RulDataModule(self.source_loader, batch_size=16)

        target_mock_runs = [np.random.randn(16, 14, 1)] * 2, [np.random.rand(16)] * 2
        self.target_loader = mock.MagicMock(name="CMAPSSLoader")
        self.target_loader.fd = 1
        self.target_loader.percent_fail_runs = 0.8
        self.target_loader.percent_broken = 0.8
        self.target_loader.window_size = 1
        self.target_loader.max_rul = 125
        self.target_loader.hparams = {
            "fd": self.target_loader.fd,
            "window_size": self.target_loader.window_size,
        }
        self.target_loader.load_split.return_value = target_mock_runs
        self.target_data = core.RulDataModule(self.target_loader, batch_size=16)

        self.dataset = adaption.DomainAdaptionDataModule(
            self.source_data, self.target_data
        )
        self.dataset.prepare_data()
        self.dataset.setup()

    @mock.patch("rul_datasets.core.RulDataModule.check_compatibility")
    def test_compatibility_checked(self, _):
        self.dataset = adaption.DomainAdaptionDataModule(
            self.source_data, self.target_data
        )
        self.source_data.check_compatibility.assert_called_with(self.target_data)

        self.source_loader.fd = 1
        self.assertRaises(
            ValueError,
            adaption.DomainAdaptionDataModule,
            self.source_data,
            self.target_data,
        )

    def test_train_source_target_order(self):
        train_dataloader = self.dataset.train_dataloader()
        self._assert_datasets_equal(
            self.dataset.source.to_dataset("dev"), train_dataloader.dataset.labeled
        )
        self._assert_datasets_equal(
            self.dataset.target.to_dataset("dev"), train_dataloader.dataset.unlabeled[0]
        )

    def test_val_source_target_order(self):
        val_source_loader, val_target_loader = self.dataset.val_dataloader()
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

    @mock.patch(
        "rul_datasets.adaption.DomainAdaptionDataModule._to_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_train_dataloader(self, mock_to_dataset):
        dataloader = self.dataset.train_dataloader()

        mock_to_dataset.assert_called_once_with("dev")
        self.assertIs(mock_to_dataset.return_value, dataloader.dataset)
        self.assertEqual(16, dataloader.batch_size)
        self.assertIsInstance(dataloader.sampler, RandomSampler)
        self.assertTrue(dataloader.pin_memory)

    def test_val_dataloader(self):
        mock_source_val = mock.MagicMock()
        mock_target_val = mock.MagicMock()
        self.dataset.source.val_dataloader = mock_source_val
        self.dataset.target.val_dataloader = mock_target_val
        dataloaders = self.dataset.val_dataloader()

        self.assertEqual(2, len(dataloaders))
        mock_source_val.assert_called_once()
        mock_target_val.assert_called_once()

    def test_test_dataloader(self):
        mock_source_test = mock.MagicMock()
        mock_target_test = mock.MagicMock()
        self.dataset.source.test_dataloader = mock_source_test
        self.dataset.target.test_dataloader = mock_target_test
        dataloaders = self.dataset.test_dataloader()

        self.assertEqual(2, len(dataloaders))
        mock_source_test.assert_called_once()
        mock_target_test.assert_called_once()

    def test_truncated_loader(self):
        self.assertIsNot(self.dataset.target.reader, self.dataset.target_truncated)
        self.assertTrue(self.dataset.target_truncated.truncate_val)

    def test_hparams(self):
        expected_hparams = {
            "fd_source": 3,
            "fd_target": 1,
            "batch_size": 16,
            "window_size": 1,
            "max_rul": 125,
            "percent_broken": 0.8,
            "percent_fail_runs": 0.8,
        }
        self.assertDictEqual(expected_hparams, self.dataset.hparams)


class TestPretrainingDataModuleFullData(
    PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        source_mock_runs = [np.random.randn(16, 1, 14)] * 3, [np.random.rand(16)] * 3
        self.source_loader = mock.MagicMock(name="CMAPSSLoader")
        self.source_loader.fd = 3
        self.source_loader.percent_fail_runs = None
        self.source_loader.percent_broken = None
        self.source_loader.window_size = 1
        self.source_loader.max_rul = 125
        self.source_loader.hparams = {
            "fd": self.source_loader.fd,
            "window_size": self.source_loader.window_size,
        }
        self.source_loader.load_split.return_value = source_mock_runs
        self.source_data = core.RulDataModule(self.source_loader, batch_size=16)

        target_mock_runs = [np.random.randn(16, 1, 14)] * 2, [np.random.rand(16)] * 2
        self.target_loader = mock.MagicMock(name="CMAPSSLoader")
        self.target_loader.fd = 1
        self.target_loader.percent_fail_runs = 0.8
        self.target_loader.percent_broken = 0.8
        self.target_loader.window_size = 1
        self.target_loader.max_rul = 125
        self.target_loader.truncate_val = True
        self.target_loader.hparams = {
            "fd": self.target_loader.fd,
            "window_size": self.target_loader.window_size,
        }
        self.target_loader.load_split.return_value = target_mock_runs
        self.target_data = core.RulDataModule(self.target_loader, batch_size=16)

        self.dataset = adaption.PretrainingAdaptionDataModule(
            self.source_data, self.target_data, num_samples=10000, min_distance=2
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 3
        self.window_size = self.target_loader.window_size

    @mock.patch("rul_datasets.core.RulDataModule.check_compatibility")
    def test_compatibility_checked(self, mock_check_compat):
        self.dataset = adaption.PretrainingAdaptionDataModule(
            self.source_data, self.target_data, num_samples=10000, min_distance=2
        )
        mock_check_compat.assert_called_with(self.target_data)

    def test_error_on_same_fd(self):
        self.source_loader.fd = 1
        self.assertRaises(
            ValueError,
            adaption.PretrainingAdaptionDataModule,
            self.source_data,
            self.target_data,
            1000,
        )

    def test_error_on_target_data_failed(self):
        self.target_loader.percent_broken = None
        self.assertRaises(
            ValueError,
            adaption.PretrainingAdaptionDataModule,
            self.source_data,
            self.target_data,
            1000,
        )
        self.target_loader.percent_broken = 1.0
        self.assertRaises(
            ValueError,
            adaption.PretrainingAdaptionDataModule,
            self.source_data,
            self.target_data,
            1000,
        )

    def test_error_on_source_data_unfailed(self):
        self.source_loader.percent_broken = 0.8
        self.assertRaises(
            ValueError,
            adaption.PretrainingAdaptionDataModule,
            self.source_data,
            self.target_data,
            1000,
        )

    def test_warning_on_non_truncated_val_data(self):
        self.target_loader.truncate_val = False
        with pytest.warns(UserWarning):
            adaption.PretrainingAdaptionDataModule(
                self.source_data, self.target_data, 10
            )

    def test_hparams(self):
        expected_hparams = {
            "fd_source": 3,
            "fd_target": 1,
            "num_samples": 10000,
            "batch_size": 16,
            "window_size": 1,
            "max_rul": 125,
            "min_distance": 2,
            "percent_broken": 0.8,
            "percent_fail_runs": 0.8,
            "truncate_target_val": True,
            "distance_mode": "linear",
        }
        self.assertDictEqual(expected_hparams, self.dataset.hparams)


class TestPretrainingDataModuleLowData(
    PretrainingDataModuleTemplate, unittest.TestCase
):
    def setUp(self):
        source_mock_runs = [np.random.randn(16, 1, 14)] * 3, [np.random.rand(16)] * 3
        self.source_loader = mock.MagicMock(name="CMAPSSLoader")
        self.source_loader.fd = 3
        self.source_loader.percent_fail_runs = None
        self.source_loader.percent_broken = None
        self.source_loader.window_size = 1
        self.source_loader.max_rul = 125
        self.source_loader.hparams = {
            "fd": self.source_loader.fd,
            "window_size": self.source_loader.window_size,
        }
        self.source_loader.load_split.return_value = source_mock_runs
        self.source_data = core.RulDataModule(self.source_loader, batch_size=16)

        target_mock_runs = (
            [np.random.randn(3, 1, 14), np.random.randn(1, 1, 14)],
            [np.random.rand(3), np.random.rand(1)],
        )
        self.target_loader = mock.MagicMock(name="CMAPSSLoader")
        self.target_loader.fd = 1
        self.target_loader.percent_fail_runs = 0.8
        self.target_loader.percent_broken = 0.8
        self.target_loader.window_size = 1
        self.target_loader.max_rul = 125
        self.target_loader.truncate_val = True
        self.target_loader.hparams = {
            "fd": self.target_loader.fd,
            "window_size": self.target_loader.window_size,
        }
        self.target_loader.load_split.return_value = target_mock_runs
        self.target_data = core.RulDataModule(self.target_loader, batch_size=16)

        self.dataset = adaption.PretrainingAdaptionDataModule(
            self.source_data, self.target_data, num_samples=10000, min_distance=2
        )
        self.dataset.prepare_data()
        self.dataset.setup()

        self.expected_num_val_loaders = 3
        self.window_size = self.target_loader.window_size


@pytest.fixture()
def labeled():
    return TensorDataset(torch.arange(100), torch.arange(100))


@pytest.fixture(params=[1, 2])
def unlabeled(request):
    return [
        (TensorDataset(torch.arange(i * 50), torch.arange(i * 50)))
        for i in range(request.param, 0, -1)
    ]


@pytest.fixture()
def dataset(labeled, unlabeled):
    dataset = adaption.AdaptionDataset(labeled, *unlabeled)

    return dataset


class TestAdaptionDataset:
    @pytest.mark.parametrize("det", [True, False])
    def test_output_shape(self, det, labeled, unlabeled):
        dataset = adaption.AdaptionDataset(labeled, *unlabeled, deterministic=det)
        item = dataset[0]
        labeled_item = labeled[0]
        unlabeled_items = [ul[0] for ul in unlabeled]

        expected_length = len(labeled_item) + sum(len(ul) - 1 for ul in unlabeled_items)
        assert len(item) == expected_length

    def test_len(self, dataset, labeled):
        assert len(labeled) == len(dataset)

    def test_source_target_shuffeled(self, dataset):
        np.random.seed(42)
        source_one, label_one, *target_one = dataset[0]
        source_another, label_another, *target_another = dataset[0]
        assert source_one == source_another
        assert label_one == label_another
        assert not target_one == target_another

    def test_source_target_deterministic(self, labeled, unlabeled):
        dataset = adaption.AdaptionDataset(labeled, *unlabeled, deterministic=True)
        for i in range(len(dataset)):
            source_one, label_one, *target_one = dataset[i]
            source_another, label_another, *target_another = dataset[i]
            assert source_one == source_another
            assert label_one == label_another
            assert target_one == target_another

    def test_non_determinism(self, labeled, unlabeled):
        one = adaption.AdaptionDataset(labeled, *unlabeled)
        another = adaption.AdaptionDataset(labeled, *unlabeled)
        _, _, *target_one = one[0]
        _, _, *target_another = another[0]

        assert not target_one == target_another

    def test_source_sampled_completely(self, dataset):
        for i in range(len(dataset)):
            source, labels, *_ = dataset[i]
            assert i == source.item()
            assert i == labels.item()


@mock.patch(
    "rul_datasets.adaption.split_healthy",
    return_value=(TensorDataset(torch.zeros(1)),) * 2,
)
@pytest.mark.parametrize(["by_max_rul", "by_steps"], [(True, None), (False, 10)])
def test_latent_align_data_module(mock_split_healthy, by_max_rul, by_steps):
    source = DummyReader(1)
    target = source.get_compatible(2, percent_broken=0.8)
    dm = adaption.LatentAlignDataModule(
        core.RulDataModule(source, 32),
        core.RulDataModule(target, 32),
        split_by_max_rul=by_max_rul,
        split_by_steps=by_steps,
    )
    dm.setup()

    dm.train_dataloader()

    mock_split_healthy.assert_has_calls(
        [
            mock.call(mock.ANY, mock.ANY, by_max_rul=True),
            mock.call(mock.ANY, mock.ANY, by_max_rul, by_steps),
        ]
    )


def test_latent_align_with_dummy():
    source = DummyReader(1)
    target = source.get_compatible(2, percent_broken=0.8)
    dm = adaption.LatentAlignDataModule(
        core.RulDataModule(source, 32),
        core.RulDataModule(target, 32),
        split_by_max_rul=True,
    )
    dm.setup()

    for batch in dm.train_dataloader():
        assert len(batch) == 6


def test_split_healthy_max_rul():
    features = [np.random.randn(10, 100, 2)]
    targets = [np.minimum(np.arange(10)[::-1], 5)]

    healthy, degraded = adaption.split_healthy(features, targets, by_max_rul=True)

    assert len(healthy) == 5
    healthy_sample = healthy[0]
    assert len(healthy_sample) == 2  # features and labels
    assert healthy_sample[0].shape == (2, 100)  # features are channel first

    assert len(degraded) == 5
    for i, degraded_sample in enumerate(degraded):
        assert len(degraded_sample) == 3  # features, degradation steps, and labels
        assert degraded_sample[0].shape == (2, 100)  # features are channel first
        assert degraded_sample[1] == i  # degradation step is timestep since healthy
