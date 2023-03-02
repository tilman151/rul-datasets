"""A module for dataset readers. Currently supported datasets are:

* **C-MAPSS** Turbofan Degradation Dataset
* **FEMTO** (PRONOSTIA) Bearing Dataset
* **XJTU-SY** Bearing Dataset

Readers are the foundation of the RUL Datasets library. They provide access to the
data on disk and convert them into a common format so that other parts of the library
can interact with it. The common format is as follows:

1. Each dataset consists of multiple sub-datasets. The indices of these sub-datasets
are called `FD`, following CMAPSS convention.

1. Each sub-dataset contains a development (`dev`), a validation (`val`) and test split
(`test`).

1. Each split contains one or multiple time series of features and RUL targets that
represent run-to-failure experiments.

1. At each time step of a time series we have a window of features and a target RUL
value. The target is the RUL value of the last time step of the window.

A reader class, e.g. the [CmapssReader][rul_datasets.reader.cmapss.CmapssReader]
represents a dataset and can manipulate it to your liking. A reader object has access
to one sub-dataset of the dataset:

```pycon
>>> reader = CmapssReader(fd=1)
```

The reader object can load the features and targets of each split into memory:

```pycon
>>> dev_features, dev_targets = reader.load_split("dev")
>>> val_features, val_targets = reader.load_split("val")
>>> test_features, test_targets = reader.load_split("test")
```

The features are a list of [numpy arrays][numpy.ndarray] where each array has a shape of
`[num_windows, window_size, num_channels]`:

```pycon
>>> type(dev_features)
<class 'list'>
>>> dev_features[0].shape
(163, 30, 14)
```

The targets are a list of [numpy arrays][numpy.ndarray], too, where each array has a
shape of `[num_windows]`:

```pycon
>>> type(dev_targets)
<class 'list'>
>>> dev_targets[0].shape
(163,)
```

Each reader defines a default window size for its data. This can be overridden by the
`window_size` argument:

```pycon
>>> fd1 = CmapssReader(fd=1, window_size=15)
>>> features, _ = fd1.load_split("dev")
>>> features[0].shape
(163, 15, 14)
```

Some datasets, i.e. CMAPSS, use a piece-wise linear RUL function, where a maximum RUL
value is defined. The maximum RUL value for a reader can be set via the `max_rul`
argument:

```pycon
>>> fd1 = CmapssReader(fd=1, max_rul=100)
>>> targets = fd1.load_split("dev")
>>> max(np.max(t) for t in targets)
100.0
```

If you want to use a sub-dataset as unlabeled data, e.g. for unsupervised domain
adaption, it should not contain features from the point of failure. If the data
contains these features, there would be no reason for it to be unlabeled. The
`percent_broken` argument controls how much data near failure is available. A
`percent_broken` of `0.8` for example means that only the first 80% of each time
series are available:

```pycon
>>> fd1 = CmapssReader(fd=1, percent_broken=0.8)
>>> features, targets = fd1.load_split("dev")
>>> features[0].shape
(130, 30, 14])
>>> np.min(targets[0])
34.0
```

You may want to apply the same `percent_broken` from your training data to your
validation data. This is sensible if you do not expect that your algorithm has access
to labeled validation data in real-life. You can achieve this, by setting
`truncate_val` to `True`:

```pycon
>>> fd1 = CmapssReader(fd=1, percent_broken=0.8, truncate_val=True)
>>> features, targets = fd1.load_split("val")
>>> np.min(targets[0])
44.0
```

Data-driven RUL estimation algorithms are often sensitive to the overall amount of
training data. The more data is available, the more of its variance is covered. If
you want to investigate how an algorithm performs in a low-data setting, you can use
`percent_fail_runs`. This argument controls how many runs are used for training. A
`percent_fail_runs` of `0.8` means that 80% of the available training runs are used.
If you need more controll over which runs are used, you can pass a list of indices to
use only these runs. This is useful for conducting semi-supervised learning where you
consider one part of a sub-dataset labeled and the other part unlabeled:

```pycon
>>> fd1 = CmapssReader(fd=1, percent_fail_runs=0.8)
>>> features, targets = fd1.load_split("dev")
>>> len(features)
64
>>> fd1 = CmapssReader(fd=1, percent_fail_runs=[0, 5, 40])
>>> features, targets = fd1.load_split("dev")
>>> len(features)
3
```

If you have constructed a reader with a certain `percent_fail_runs`, you can get a
reader containing all other runs by using the `get_complement` function:

```pycon
>>> fd1 = CmapssReader(fd=1, percent_fail_runs=0.8)
>>> fd1_complement = fd1.get_complement()
>>> features, targets = fd1_complement.load_split("dev")
>>> len(features)
16
```

The effects of `percent_broken` and `percent_fail_runs` are summarized under the term
**truncation** as they effectively truncate the dataset in two dimensions.

The readers for the FEMTO and XJTU-SY datasets have two additional constructor
arguments. The `first_time_to_predict` lets you set an individual maximum RUL value
per run in the dataset. As both are bearing datasets, the first-time-to-predict is
defined as the time step where the degradation of the bearing is first noticeable.
The RUL value before this time step is assumed to be constant. Setting `norm_rul`
scales the RUL between [0, 1] per run, as it is best practice when using
first-time-to-predict.

```pycon
>>> fttp = [10, 20, 30, 40, 50]
>>> fd1 = rul_datasets.reader.XjtuSyReader(
...     fd=1, first_time_to_predict=fttp, norm_rul=True
... )
>>> fd1.prepare_data()
>>> features, labels = fd1.load_split("dev")
>>> labels[0][:15]
array([1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 0.99115044, 0.98230088, 0.97345133, 0.96460177])
```

Readers can be used as is if you just want access to the dataset. If you plan to use
them with PyTorch or PyTorch Lightning, it is recommended to combine them with a
[RulDataModule][rul_datasets.core.RulDataModule]:

```pycon
>>> fd1 = CmapssReader(fd=1)
>>> dm = RulDataModule(fd1, batch_size=32)
```

For more information, see [core][rul_datasets.core] module page or the
[Libraries](/rul-datasets/use_cases/libraries) page.

 """

from .abstract import AbstractReader
from .data_root import _DATA_ROOT
from .cmapss import CmapssReader
from .femto import FemtoReader, FemtoPreparator
from .xjtu_sy import XjtuSyReader, XjtuSyPreparator
from .dummy import DummyReader
