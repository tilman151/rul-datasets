# RUL Datasets

[![Master](https://github.com/tilman151/rul-datasets/actions/workflows/on_push.yaml/badge.svg)](https://github.com/tilman151/rul-datasets/actions/workflows/on_push.yaml)
[![Release](https://github.com/tilman151/rul-datasets/actions/workflows/on_release.yaml/badge.svg)](https://github.com/tilman151/rul-datasets/actions/workflows/on_release.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This library contains a collection of common benchmark datasets for **remaining useful lifetime (RUL)** estimation.
They are provided as [LightningDataModules](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningDataModule.html#pytorch_lightning.core.LightningDataModule) to be readily used in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Currently, four datasets are supported:

* **C-MAPSS** Turbofan Degradation Dataset
* **FEMTO** (PRONOSTIA) Bearing Dataset
* **XJTU-SY** Bearing Dataset
* **Dummy** A tiny, simple dataset for debugging

All datasets share the same API, so they can be used as drop-in replacements for each other.
That means, if an experiment can be run with one of the datasets, it can be run with all of them.
No code changes needed.

Aside from the basic ones, this library contains data modules for advanced experiments concerning **transfer learning**, **unsupervised domain adaption** and **semi-supervised learning**.
These data modules are designed as **higher-order data modules**.
This means they take one or more of the basic data modules as inputs and adjust them to the desired use case.

## Installation

The library is pip-installable. Simply type:

```shell
pip install rul-datasets
```

## Contribution

Contributions are always welcome. Whether you want to fix a bug, add a feature or a new dataset, just open an issue and a PR.