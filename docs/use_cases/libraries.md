## PyTorch Lightning

This library was developed to be used in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) first and foremost.
Lightning helps writing clean and reproducible deep learning code that can run on most common training hardware.
Datasets are represented by [LightningDataModules][pytorch_lightning.core.LightningDataModule] which give access to data loaders for each data split.
The RUL Datasets library implements several data modules that are 100% compatible with Lightning:

```python
import pytorch_lightning as pl
import rul_datasets

import rul_estimator


cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)

my_rul_estimator = rul_estimator.MyRulEstimator() # (1)!

trainer = pl.Trainer(max_epochs=100)
trainer.fit(my_rul_estimator, dm) # (2)!

trainer.test(my_rul_estimator, dm)
```

1. This should be a subclass of [LightningModule][pytorch_lightning.core.LightningModule].
2. The trainer calls the data module's `prepare_data` and `setup` functions automatically.

## PyTorch

If you do not want to work with PyTorch Lightning, you can still use the RUL Dataset library in plain PyTorch.
The data loaders provided by the data modules can be used as is:

```python
import torch
import rul_datasets

import rul_estimator

cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
dm.prepare_data() # (1)!
dm.setup() # (2)!

my_rul_estimator = rul_estimator.MyRulEstimator() # (3)!
optim = torch.optim.Adam(my_rul_estimator.parameters())

best_val_loss = torch.inf

for epoch in range(100):
    print(f"Train epoch {epoch}")
    my_rul_estimator.train()
    for features, targets in dm.train_dataloader():
        optim.zero_grad()
        
        predictions = my_rul_estimator(features)
        loss = torch.sqrt(torch.mean((targets - predictions)**2)) # (4)!
        loss.backward()
        print(f"Training loss: {loss}")
        
        optim.step()

    print(f"Validate epoch {epoch}")
    my_rul_estimator.eval()
    val_loss = 0
    num_samples = 0
    for features, targets in dm.val_dataloader():
        predictions = my_rul_estimator(features)
        loss = torch.sum((targets - predictions)**2)
        val_loss += loss.detach()
        num_samples += predictions.shape[0]
    val_loss = torch.sqrt(val_loss / num_samples) # (5)!
        
    if best_val_loss < val_loss:
        break
    else:
        best_val_loss = val_loss
        print(f"Validation loss: {best_val_loss}")

test_loss = 0
num_samples = 0
for features, targets in dm.test_dataloader():
    predictions = my_rul_estimator(features)
    loss = torch.sqrt(torch.dist(predictions, targets))
    test_loss += loss.detach()
    num_samples += predictions.shape[0]
test_loss = torch.sqrt(test_loss / num_samples) # (6)!

print(f"Test loss: {test_loss}")
```

1. You need to call `prepare_data` before using the reader. This downloads and pre-processes the dataset if not done already.
2. You need to call `setup` to load all splits into memory before using them.
3. This should be a subclass of a torch [Module][torch.nn.Module].
4. Calculates the RMSE loss.
5. Calculate the mean and square root after all squared sums are collected. This ensures a correct validation loss.
6. Calculate the mean and square root after all squared sums are collected. This ensures a correct test loss.

## Others

All datasets in this library can be used in any other library as well.
For this you need to create a [reader][rul_datasets.reader] for your desired dataset and call its `load_split` function.
Here is an example with [tslearn](https://tslearn.readthedocs.io/en/stable/):

```python
import numpy as np
import tslearn
import rul_datasets

cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
cmapss_fd1.prepare_data() # (1)!
dev_features, _ = cmapss_fd1.load_split("dev") # (2)!
dev_data = np.concatenate(dev_features) # (3)!

km = tslearn.clustering.TimeSeriesKMeans(n_clusters=5, metric="dtw")
km.fit(dev_data)
```

1. You need to call `prepare_data` before using the reader. This downloads and pre-processes the dataset if not done already.
2. This yields a list of numpy arrays with the shape `[len_time_series, window_size, num_features]`.
3. Concatenate to a single numpy array with the shape `[num_series, window_size, num_features]`.
