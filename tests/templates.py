import inspect
from typing import Type
from unittest import mock

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset

from rul_datasets.core import PairedRulDataset
from rul_datasets.loader import AbstractLoader


class PretrainingDataModuleTemplate:
    dataset: LightningDataModule
    window_size: int

    def test_data_structure(self):
        with self.subTest(split="dev"):
            dataloader = self.dataset.train_dataloader()
            self._check_paired_dataset(dataloader.dataset)

        with self.subTest(split="val"):
            loaders = self.dataset.val_dataloader()
            self.assertIsInstance(loaders, list)
            self.assertEqual(self.expected_num_val_loaders, len(loaders))
            self._check_paired_dataset(loaders[0].dataset)
            for dataloader in loaders[1:]:
                self._check_tensor_dataset(dataloader.dataset)

    def _check_paired_dataset(self, data):
        self.assertIsInstance(data, PairedRulDataset)
        self._check_paired_shapes(data)

    def _check_paired_shapes(self, data):
        for anchors, queries, distances, domain_labels in data:
            self.assertEqual(torch.Size((14, self.window_size)), anchors.shape)
            self.assertEqual(torch.Size((14, self.window_size)), queries.shape)
            self.assertEqual(torch.Size(()), distances.shape)
            self.assertEqual(torch.Size(()), domain_labels.shape)

    def _check_tensor_dataset(self, data):
        self.assertIsInstance(data, TensorDataset)
        self._check_cmapss_shapes(data)

    def _check_cmapss_shapes(self, data):
        for i in range(len(data)):
            features, labels = data[i]
            self.assertEqual(torch.Size((14, self.window_size)), features.shape)
            self.assertEqual(torch.Size(()), labels.shape)

    def test_distances(self):
        with self.subTest(split="dev"):
            _, _, distances, _ = self._run_epoch(self.dataset.train_dataloader())
            self.assertTrue(torch.all(distances >= 0))

        with self.subTest(split="val"):
            _, _, distances, _ = self._run_epoch(self.dataset.val_dataloader()[0])
            self.assertTrue(torch.all(distances >= 0))

    def test_determinism(self):
        with self.subTest(split="dev"):
            train_loader = self.dataset.train_dataloader()
            *one_train_data, one_domain_labels = self._run_epoch(train_loader)
            *another_train_data, another_domain_labels = self._run_epoch(train_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertNotEqual(0.0, torch.dist(one, another))

        with self.subTest(split="val"):
            paired_val_loader = self.dataset.val_dataloader()[0]
            one_train_data = self._run_epoch(paired_val_loader)
            another_train_data = self._run_epoch(paired_val_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertEqual(0.0, torch.dist(one, another))

    def _run_epoch(self, loader):
        anchors = torch.empty((len(loader.dataset), 14, self.window_size))
        queries = torch.empty((len(loader.dataset), 14, self.window_size))
        distances = torch.empty(len(loader.dataset))
        domain_labels = torch.empty(len(loader.dataset))

        start = 0
        end = loader.batch_size
        for anchor, query, dist, domain in loader:
            anchors[start:end] = anchor
            queries[start:end] = query
            distances[start:end] = dist
            domain_labels[start:end] = domain
            start = end
            end += anchor.shape[0]

        return anchors, queries, distances, domain_labels

    def test_min_distance(self):
        train_loader = self.dataset.train_dataloader()
        self.assertEqual(self.dataset.min_distance, train_loader.dataset.min_distance)
        val_loader = self.dataset.val_dataloader()[0]
        self.assertEqual(1, val_loader.dataset.min_distance)

    def test_train_dataloader(self):
        mock_get_paired_dataset = mock.MagicMock(
            name="_get_paired_dataset", wraps=self.dataset._get_paired_dataset
        )
        self.dataset._get_paired_dataset = mock_get_paired_dataset
        train_loader = self.dataset.train_dataloader()

        mock_get_paired_dataset.assert_called_with("dev")
        self.assertEqual(self.dataset.batch_size, train_loader.batch_size)
        self.assertFalse(train_loader.dataset.deterministic)
        self.assertTrue(train_loader.pin_memory)

    def test_val_dataloader(self):
        mock_get_paired_dataset = mock.MagicMock(
            name="_get_paired_dataset", wraps=self.dataset._get_paired_dataset
        )
        self.dataset._get_paired_dataset = mock_get_paired_dataset
        val_loaders = self.dataset.val_dataloader()

        mock_get_paired_dataset.assert_called_with("val")
        self.assertTrue(val_loaders[0].dataset.deterministic)
        for val_loader in val_loaders:
            self.assertEqual(self.dataset.batch_size, val_loader.batch_size)
            self.assertTrue(val_loader.pin_memory)


class LoaderInterfaceTemplate:
    loader_type: Type[AbstractLoader]

    def test_constructor(self):
        signature = inspect.signature(self.loader_type)
        self.assertListEqual(
            [
                "fd",
                "window_size",
                "max_rul",
                "percent_broken",
                "percent_fail_runs",
                "feature_select",
                "truncate_val",
            ],
            list(signature.parameters.keys()),
        )

        self.assertEqual(inspect.Signature.empty, signature.parameters["fd"].default)
        params_with_defaults = list(signature.parameters.values())[1:]
        self.assertTrue(
            all(
                [not p.default == inspect.Signature.empty for p in params_with_defaults]
            )
        )

    def test_attributes(self):
        tested_loader = self.loader_type(1)
        self.assertTrue(hasattr(tested_loader, "fd"))
        self.assertTrue(hasattr(tested_loader, "window_size"))
        self.assertTrue(hasattr(tested_loader, "max_rul"))
        self.assertTrue(hasattr(tested_loader, "percent_broken"))
        self.assertTrue(hasattr(tested_loader, "percent_fail_runs"))
        self.assertTrue(hasattr(tested_loader, "truncate_val"))
        self.assertTrue(hasattr(tested_loader, "_NUM_TRAIN_RUNS"))

    def test_hparams_is_property(self):
        self.assertTrue(hasattr(self.loader_type, "hparams"))
        self.assertIsInstance(self.loader_type.hparams, property)

    def test_functions(self):
        tested_loader = self.loader_type(1)
        self.assertTrue(hasattr(tested_loader, "prepare_data"))
        self.assertTrue(inspect.ismethod(tested_loader.prepare_data))
        self.assertTrue(hasattr(tested_loader, "load_split"))
        self.assertTrue(inspect.ismethod(tested_loader.load_split))
        self.assertTrue(hasattr(tested_loader, "check_compatibility"))
        self.assertTrue(inspect.ismethod(tested_loader.check_compatibility))
        self.assertTrue(hasattr(tested_loader, "_default_window_size"))
        self.assertTrue(inspect.ismethod(tested_loader._default_window_size))

    def test_load_split(self):
        tested_loader = self.loader_type(1)
        for split in ["dev", "test"]:
            with self.subTest(split):
                loaded_split = tested_loader.load_split(split)
                self._check_loaded_split(loaded_split)

        with self.subTest("val"):
            try:
                loaded_split = tested_loader.load_split("val")
            except ValueError:
                pass
            except Exception:
                self.fail("Loader has to throw a ValueError on missing val split.")
            else:
                self._check_loaded_split(loaded_split)

    def _check_loaded_split(self, loaded_split):
        self.assertIsInstance(loaded_split, tuple)
        for tensor_list in loaded_split:
            self.assertIsInstance(tensor_list, list)
            for tensor in tensor_list:
                self.assertIsInstance(tensor, torch.Tensor)
