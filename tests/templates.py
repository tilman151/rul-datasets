import os
import zipfile

import torch
from torch.utils.data import TensorDataset

from datasets.cmapss import PairedCMAPSS


class CmapssTestTemplate:
    @classmethod
    def setUpClass(cls):
        script_path = os.path.dirname(__file__)
        data_path = os.path.join(script_path, "..", "..", "data", "CMAPSS")
        if "train_FD001.txt" not in os.listdir(data_path):
            print("Extract CMAPSS data...")
            data_zip = os.path.join(data_path, "CMAPSSData.zip")
            with zipfile.ZipFile(data_zip) as zip_file:
                zip_file.extractall(data_path)


class FemtoTestTemplate:
    @classmethod
    def setUpClass(cls):
        script_path = os.path.dirname(__file__)
        data_path = os.path.join(script_path, "..", "..", "data", "FEMTOBearingDataSet")
        if "Test_set" not in os.listdir(data_path):
            print("Extract FEMTO data...")
            test_zip = os.path.join(data_path, "TestSet.zip")
            with zipfile.ZipFile(test_zip) as zip_file:
                zip_file.extractall(data_path)
            train_zip = os.path.join(data_path, "Training_set.zip")
            with zipfile.ZipFile(train_zip) as zip_file:
                zip_file.extractall(data_path)
            val_zip = os.path.join(data_path, "Validation_Set.zip")
            with zipfile.ZipFile(val_zip) as zip_file:
                zip_file.extractall(data_path)


class PretrainingDataModuleTemplate:
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
        self.assertIsInstance(data, PairedCMAPSS)
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
                self.assertNotEqual(0.0, torch.sum(one - another))

        with self.subTest(split="val"):
            paired_val_loader = self.dataset.val_dataloader()[0]
            one_train_data = self._run_epoch(paired_val_loader)
            another_train_data = self._run_epoch(paired_val_loader)

            for one, another in zip(one_train_data, another_train_data):
                self.assertEqual(0.0, torch.sum(one - another))

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
