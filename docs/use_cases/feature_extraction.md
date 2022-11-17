Datasets like FEMTO or XJTU-SY provide raw vibration data.
It may be useful to extract hand-crafted features, i.e. RMS or P2P, from this vibration data to use in your network.
The [RulDataModule][rul_datasets.core.RulDataModule] provides the option to use a custom feature extractor on each window of data.

The feature extractor can be anything that can be called as a function.
It should take a numpy array with the shape `[num_windows, window_size, num_features]` and return an array with the shape `[num_windows, num_new_features]`.
An example would be taking the mean of the window with `lambda x: np.mean(x, axis=1)`.
After applying the feature extractor, the data module extracts new windows of extracted features:

```pycon
>>> import rul_datasets
>>> import numpy as np
>>> reader = rul_datasets.FemtoReader(fd=1)
>>> dm = rul_datasets.RulDataModule(
...     reader,
...     batch_size=32,
...     feature_extractor=lambda x: np.mean(x, axis=1),
...     window_size=10
... )
>>> dm.setup()
>>> features, _ = dm.to_dataset("dev")[0]
>>> features.shape
torch.Size([2, 10])
```

The new features have a new window size of 10 instead of 30, the readers default.
Each of the two features contains the mean of the corresponding feature in the original data window.

The number of samples will reduce by `num_runs * (window_size - 1)` due to the re-windowing of the data:

```pycon
>>> reader = rul_datasets.FemtoReader(fd=1)
>>> dm_org = rul_datasets.RulDataModule(reader, batch_size=32)
>>> dm_org.setup()
>>> dm_extracted = rul_datasets.RulDataModule(
...     reader,
...     batch_size=32,
...     feature_extractor=lambda x: np.mean(x, axis=1),
...     window_size=10
... )
>>> dm_extracted.setup()
>>> len(dm_org.to_dataset("dev"))
3674
>>> dm_extracted.to_dataset("dev")
3656
```