# RUL Datasets

This library contains a collection of common benchmark datasets for **remaining useful lifetime (RUL)** estimation.
They are provided as [LightningDataModules](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningDataModule.html#pytorch_lightning.core.LightningDataModule) to be readily used in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Currently, three datasets are supported:

* **C-MAPSS** Turbofan Degradation Dataset
* **FEMTO** (PRONOSTIA) Bearing Dataset
* **XJTU-SY** Bearing Dataset

All datasets share the same API, so they can be used as drop-in replacements for each other.
That means, if an experiment can be run with one of the datasets, it can be run with all of them.
No code changes needed.

Aside from the basic ones, this library contains data modules for advanced experiments concerning **transfer learning**, **unsupervised domain adaption** and **semi-supervised learning**.
These data modules are designed as **higher-order data modules**.
This means they take one or more of the basic data modules as inputs and adjust them to the desired use case.