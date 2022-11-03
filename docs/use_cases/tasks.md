## Supervised Learning

For simple supervised learning experiments, the library provides the [RulDataModule][rul_datasets.core.RulDataModule].
It provides data loaders for each split of a single sub-dataset, e.g. FD1 of CMAPSS:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
>>> dm = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
>>> dm.prepare_data()
>>> dm.setup()
>>> features, targets = next(dm.train_dataloader())
>>> features.shape  # (1)!
torch.Size([32, 14, 30])
>>> targets.shape # (2)!
torch.Size([32])
```

1. Features with a shape of `[batch_size, num_features, window_size]`.
2. RUL targets with a shape of `[batch_size]`.

You can conduct experiments on all sub-datasets by simply switching the reader instance:

```python
import pytorch_lightning as pl
import rul_datasets

import rul_estimator

for fd in [1, 2, 3, 4]:
    cmapss = rul_datasets.CmapssReader(fd)
    dm = rul_datasets.RulDataModule(cmapss, batch_size=32)
    
    my_rul_estimator = rul_estimator.MyRulEstimator()
    
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(my_rul_estimator, dm)
    
    trainer.test(my_rul_estimator, dm)
```

In case you want to use supervised learning as a baseline to evaluate the generalization from one sub-dataset to all others, you can use the [BaselineDataModule][rul_datasets.baseline.BaselineDataModule].
It works similar to the [RulDataModule][rul_datasets.core.RulDataModule] but returns the validation and test data loaders for all sub-datasets:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
>>> dm = rul_datasets.BaselineDataModule(cmapss_fd1, batch_size=32)
>>> dm.prepare_data()
>>> dm.setup()
>>> val_fd1, val_fd2, val_fd3, val_fd4 = dm.val_dataloader()
>>> test_fd1, test_fd2, test_fd3, test_fd4 = dm.test_dataloader()
```

## Semi-Supervised Learning

In semi-supervised learning you try to incorporate additional unlabeled data into the learning process.
As we do not have additional data for our dataset, we need to split the existing data into a labeled and an unlabeled portion.
Each reader has a `percent_fail_runs` argument with which we can control which runs are included in our training data:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1, percent_fail_runs=[0, 1, 2])
>>> features, labels = cmapss_fd1.load_split("dev")
>>> len(features)
3
```

This way we can construct two readers that will include different runs.
One will be considered labeled and one unlabeled.
If you only want to create one reader and get a second reader containing the remaining runs, you can use the `get_complement` function:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1, percent_fail_runs=[0, 1, 2])
>>> complement = cmapss_fd1.get_complement()
>>> features, labels = complement.load_split("dev")
>>> len(features)
77
```

After constructing the readers, you can put them together in a [SemiSupervisedDataModule][rul_datasets.ssl.SemiSupervisedDataModule].
This data module constructs batches of training data in a way that you get a batch of random unlabeled data for each batch of labeled data.
Common semi-supervised approaches, like Ladder Networks, can be trained this way:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1, percent_fail_runs=[0, 1, 2])
>>> complement = cmapss_fd1.get_complement(percent_broken=0.8) # (1)!
>>> labeled = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
>>> unlabeled = rul_datasets.RulDataModule(complement, batch_size=32)
>>> dm = rul_datasets.SemiSupervisedDataModule(labeled, unlabeled)
>>> labeled_features, targets, unlabeled_features = next(dm.train_dataloader())
```

1. Using the `get_complement` function, we can override the `percent_broken` for the unlabeled data module to make sure that it does not contain data near failure.
   This way it behaves like real unlabeled data of machines that did not yet fail.

For validation and testing, the data module returns data loaders for the full labeled splits.

## Unsupervised Domain Adaption

Unsupervised domain adaption uses a labeled dataset form a source domain to train a model for a target domain for which only unlabeled data is available.
All included dataset consist of multiple sub-datasets that can be viewed as different domains.
As the sub-dataset still bear a sufficient similarity to each other, domain adaption between them should be possible.
The `get_compatible` function is useful to construct a reader for a different sub-dataset from an existing one:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
>>> cmapss_fd2 = cmapss_fd1.get_compatible(fd=2)
>>> cmapss_fd1.check_compatibility(cmapss_fd2)
>>> cmapss_fd2.fd
2
```

The [DomainAdaptionDataModule][rul_datasets.adaption.DomainAdaptionDataModule] can help with this use case.
It takes a source and a target data module which have to be different sub-datasets of the same dataset or different datasets entirely.
As for semi-supervised learning, this data module constructs batches of training data in a way that you get a batch of random unlabeled data for each batch of labeled data.
Common unsupervised domain adaption approaches, like Domain Adversarial Neural Networks, can be trained this way:

```pycon
>>> cmapss_fd1 = rul_datasets.CmapssReader(fd=1)
>>> cmapss_fd2 = cmapss_fd1.get_compatible(fd=2, percent_broken=0.8) # (1)!
>>> source = rul_datasets.RulDataModule(cmapss_fd1, batch_size=32)
>>> target = rul_datasets.RulDataModule(cmapss_fd2, batch_size=32)
>>> dm = rul_datasets.DomainAdaptionDataModule(source, target)
>>> source_features, source_targets, target_features = next(dm.train_dataloader())
```

1. Using the `get_compatible` function, we can override the `percent_broken` for the target data module to make sure that it does not contain data near failure.
   This way it behaves like real unlabeled data of machines that did not yet fail.

For validation and testing, the data module returns both source and target data loaders.