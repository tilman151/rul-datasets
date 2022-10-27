import os
import unittest

import hydra

import rul_datasets

CONFIG_DIR = os.path.join(".", "assets", "configs")


class TestBaseline(unittest.TestCase):
    _hydra_init = None

    @classmethod
    def setUpClass(cls):
        cls._hydra_init = hydra.initialize(config_path=CONFIG_DIR)
        cls._hydra_init.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._hydra_init.__exit__(None, None, None)

    def test_dm(self):
        with self.subTest("cmapss"):
            cfg = hydra.compose(
                config_name="baseline", overrides=["+experiment=baseline_cmapss_4"]
            )
            cmapss_dm = hydra.utils.instantiate(cfg.dm)
            self.assertIsInstance(cmapss_dm, rul_datasets.BaselineDataModule)
            self.assertIsInstance(
                cmapss_dm.data_module.reader, rul_datasets.CmapssReader
            )

        with self.subTest("femto"):
            cfg = hydra.compose(
                config_name="baseline", overrides=["+experiment=baseline_femto_2"]
            )
            femto_dm = hydra.utils.instantiate(cfg.dm)
            self.assertIsInstance(femto_dm, rul_datasets.BaselineDataModule)
            self.assertIsInstance(femto_dm.data_module.reader, rul_datasets.FemtoReader)

    def test_dm_pre(self):
        with self.subTest("cmapss"):
            cfg = hydra.compose(
                config_name="baseline", overrides=["+experiment=baseline_pre_cmapss_4"]
            )
            cmapss_dm = hydra.utils.instantiate(cfg.dm_pre)
            self.assertIsInstance(cmapss_dm, rul_datasets.PretrainingBaselineDataModule)
            self.assertIsInstance(cmapss_dm.failed_loader, rul_datasets.CmapssReader)
            self.assertIsInstance(cmapss_dm.failed_loader, rul_datasets.CmapssReader)

        with self.subTest("femto"):
            cfg = hydra.compose(
                config_name="baseline", overrides=["+experiment=baseline_pre_femto_2"]
            )
            femto_dm = hydra.utils.instantiate(cfg.dm_pre)
            self.assertIsInstance(femto_dm, rul_datasets.PretrainingBaselineDataModule)
            self.assertIsInstance(femto_dm.failed_loader, rul_datasets.FemtoReader)
            self.assertIsInstance(femto_dm.failed_loader, rul_datasets.FemtoReader)


class TestAdaption(unittest.TestCase):
    _hydra_init = None

    @classmethod
    def setUpClass(cls):
        cls._hydra_init = hydra.initialize(config_path=CONFIG_DIR)
        cls._hydra_init.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._hydra_init.__exit__(None, None, None)

    def test_dm(self):
        with self.subTest("cmapss"):
            cfg = hydra.compose(
                config_name="adaption", overrides=["+experiment=adaption_cmapss_2to4"]
            )
            cmapss_dm = hydra.utils.instantiate(cfg.dm)
            self.assertIsInstance(cmapss_dm, rul_datasets.DomainAdaptionDataModule)
            self.assertIsInstance(cmapss_dm.source.reader, rul_datasets.CmapssReader)
            self.assertEqual(cmapss_dm.source.reader.fd, 2)
            self.assertIsInstance(cmapss_dm.target.reader, rul_datasets.CmapssReader)
            self.assertEqual(cmapss_dm.target.reader.fd, 4)

        with self.subTest("femto"):
            cfg = hydra.compose(
                config_name="adaption", overrides=["+experiment=adaption_femto_1to2"]
            )
            femto_dm = hydra.utils.instantiate(cfg.dm)
            self.assertIsInstance(femto_dm, rul_datasets.DomainAdaptionDataModule)
            self.assertIsInstance(femto_dm.source.reader, rul_datasets.FemtoReader)
            self.assertEqual(femto_dm.source.reader.fd, 1)
            self.assertIsInstance(femto_dm.target.reader, rul_datasets.FemtoReader)
            self.assertEqual(femto_dm.target.reader.fd, 2)

    def test_dm_pre(self):
        with self.subTest("cmapss"):
            cfg = hydra.compose(
                config_name="adaption",
                overrides=["+experiment=adaption_pre_cmapss_2to4"],
            )
            cmapss_dm = hydra.utils.instantiate(cfg.dm_pre)
            self.assertIsInstance(cmapss_dm, rul_datasets.PretrainingAdaptionDataModule)
            self.assertIsInstance(cmapss_dm.source.reader, rul_datasets.CmapssReader)
            self.assertEqual(cmapss_dm.source.reader.fd, 2)
            self.assertIsInstance(cmapss_dm.target.reader, rul_datasets.CmapssReader)
            self.assertEqual(cmapss_dm.target.reader.fd, 4)
            self.assertEqual(cmapss_dm.target.reader.percent_broken, 0.8)

        with self.subTest("femto"):
            cfg = hydra.compose(
                config_name="adaption",
                overrides=["+experiment=adaption_pre_femto_1to2"],
            )
            femto_dm = hydra.utils.instantiate(cfg.dm_pre)
            self.assertIsInstance(femto_dm, rul_datasets.PretrainingAdaptionDataModule)
            self.assertIsInstance(femto_dm.source.reader, rul_datasets.FemtoReader)
            self.assertEqual(femto_dm.source.reader.fd, 1)
            self.assertIsInstance(femto_dm.target.reader, rul_datasets.FemtoReader)
            self.assertEqual(femto_dm.target.reader.fd, 2)
            self.assertEqual(cmapss_dm.target.reader.percent_broken, 0.8)
