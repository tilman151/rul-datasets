# RUL Datasets

This library contains a collection of common benchmark datasets for **remaining useful lifetime (RUL)** estimation.
They are provided as [LightningDataModules][lightning.pytorch.core.LightningDataModule] to be readily used in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Currently, five datasets are supported:

* **C-MAPSS** Turbofan Degradation Dataset
* **FEMTO** (PRONOSTIA) Bearing Dataset
* **XJTU-SY** Bearing Dataset
* **N-C-MAPSS** New Turbofan Degradation Dataset
* **Dummy** dataset for debugging

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

Datasets will be downloaded to a cache directory called `data_root` when used for the first time.
The default directory on all systems is `~/.rul-datasets`, where `~` is the users home folder.
You can customize the `data_root` by either setting the environment variable `RUL_DATASETS_DATA_ROOT` or by calling [rul_datasets.set_data_root][]. The manually set data root must be an already existing folder.

## Contribution

Contributions are always welcome. Whether you want to fix a bug, add a feature or a new dataset, just open an issue and a PR.