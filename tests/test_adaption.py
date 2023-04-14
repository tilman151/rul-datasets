import unittest
import warnings
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, TensorDataset

import rul_datasets
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
        self.source_data = mock.MagicMock(rul_datasets.RulDataModule)
        self.source_data.reader = self.source_loader
        self.source_data.batch_size = 16
        self.source_data.to_dataset.return_value = TensorDataset(torch.zeros(1))

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
        self.target_data = mock.MagicMock(rul_datasets.RulDataModule)
        self.target_data.reader = self.target_loader
        self.target_data.batch_size = 16
        self.target_data.to_dataset.return_value = TensorDataset(torch.zeros(1))

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

    def test_inductive(self):
        self.dataset.inductive = False
        self.dataset.train_dataloader()
        self.dataset.target.to_dataset.assert_called_with("dev", alias="dev")

        self.dataset.inductive = True
        self.dataset.train_dataloader()
        self.dataset.target.to_dataset.assert_called_with("test", alias="dev")

    def test_train_source_target_order(self):
        train_dataloader = self.dataset.train_dataloader()
        self.source_data.to_dataset.assert_called_once_with("dev")
        self.target_data.to_dataset.assert_called_once_with("dev", alias="dev")
        self.assertIs(
            self.dataset.source.to_dataset.return_value,
            train_dataloader.dataset.labeled,
        )
        self.assertIs(
            self.dataset.target.to_dataset.return_value,
            train_dataloader.dataset.unlabeled[0],
        )

    def test_val_source_target_order(self):
        val_source_loader, val_target_loader = self.dataset.val_dataloader()
        self.assertIs(val_source_loader, self.dataset.source.val_dataloader())
        self.assertIs(val_target_loader, self.dataset.target.val_dataloader())

    def test_test_source_target_order(self):
        test_source_loader, test_target_loader = self.dataset.test_dataloader()
        self.assertIs(test_source_loader, self.dataset.source.test_dataloader())
        self.assertIs(test_target_loader, self.dataset.target.test_dataloader())

    @mock.patch(
        "rul_datasets.adaption.DomainAdaptionDataModule._get_training_dataset",
        return_value=TensorDataset(torch.zeros(1)),
    )
    def test_train_dataloader(self, mock_get_training_dataset):
        dataloader = self.dataset.train_dataloader()

        mock_get_training_dataset.assert_called_once_with()
        self.assertIs(mock_get_training_dataset.return_value, dataloader.dataset)
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
    source = mock.MagicMock(core.RulDataModule)
    source.batch_size = 32
    source.reader.window_size = 30
    source.load_split.return_value = ([torch.zeros(1)],) * 2
    target = mock.MagicMock(core.RulDataModule)
    target.load_split.return_value = ([torch.ones(1)],) * 2

    dm = adaption.LatentAlignDataModule(
        source, target, split_by_max_rul=by_max_rul, split_by_steps=by_steps
    )

    dm.train_dataloader()

    mock_split_healthy.assert_has_calls(
        [
            mock.call(
                source.load_split.return_value[0],
                source.load_split.return_value[1],
                by_max_rul=True,
            ),
            mock.call(
                target.load_split.return_value[0],
                target.load_split.return_value[1],
                by_max_rul,
                by_steps,
            ),
        ]
    )


@mock.patch(
    "rul_datasets.adaption.split_healthy",
    return_value=(TensorDataset(torch.zeros(1)),) * 2,
)
@pytest.mark.parametrize(["inductive", "exp_split"], [(True, "test"), (False, "dev")])
def test_latent_align_data_module_inductive(_, inductive, exp_split):
    source = mock.MagicMock(core.RulDataModule)
    source.batch_size = 32
    source.reader.window_size = 30
    source.load_split.return_value = ([torch.zeros(1)],) * 2
    target = mock.MagicMock(core.RulDataModule)
    target.load_split.return_value = ([torch.zeros(1)],) * 2

    dm = adaption.LatentAlignDataModule(
        source, target, inductive=inductive, split_by_max_rul=True
    )

    dm.train_dataloader()

    source.load_split.assert_called_once_with("dev")
    target.load_split.assert_called_once_with(exp_split, alias="dev")


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


@pytest.mark.parametrize(
    ["features", "targets"],
    [
        ([np.random.randn(11, 100, 2)], [np.minimum(np.arange(11)[::-1], 5)]),
        ([torch.randn(11, 2, 100)], [torch.clamp_max(torch.arange(11).flip(0), 5)]),
    ],
)
@pytest.mark.parametrize(["by_max_rul", "by_steps"], [(True, None), (False, 6)])
def test_split_healthy(features, targets, by_max_rul, by_steps):
    healthy, degraded = adaption.split_healthy(features, targets, by_max_rul, by_steps)

    assert len(healthy) == 6
    healthy_sample = healthy[0]
    assert len(healthy_sample) == 2  # features and labels
    assert healthy_sample[0].shape == (2, 100)  # features are channel first

    assert len(degraded) == 5
    for i, degraded_sample in enumerate(degraded, start=1):
        assert len(degraded_sample) == 3  # features, degradation steps, and labels
        assert degraded_sample[0].shape == (2, 100)  # features are channel first
        assert degraded_sample[1] == i  # degradation step is timestep since healthy


@pytest.mark.parametrize(["by_max_rul", "by_steps"], [(True, None), (False, 15)])
def test_split_healthy_no_degraded(by_steps, by_max_rul):
    features = [np.random.randn(11, 100, 2)]
    targets = [np.ones(11) * 125]

    healthy, degraded = adaption.split_healthy(features, targets, by_max_rul, by_steps)

    assert len(healthy) == 11
    assert len(degraded) == 0
