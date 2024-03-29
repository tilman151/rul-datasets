"""Basic data modules for experiments involving only a single subset of any RUL
dataset. """

from typing import Dict, List, Optional, Tuple, Any, Callable, cast, Union, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    get_worker_info,
    Dataset,
)

from rul_datasets import utils
from rul_datasets.reader import AbstractReader


class RulDataModule(pl.LightningDataModule):
    """
    A [data module][lightning.pytorch.core.LightningDataModule] to provide windowed
    time series features with RUL targets. It exposes the splits of the underlying
    dataset for easy usage with PyTorch and PyTorch Lightning.

    The data module implements the `hparams` property used by PyTorch Lightning to
    save hyperparameters to checkpoints. It retrieves the hyperparameters of its
    underlying reader and adds the batch size to them.

    If you want to extract features from the windows, you can pass the
    `feature_extractor` and `window_size` arguments to the constructor. The
    `feature_extractor` is a callable that takes a windowed time series as a numpy
    array with the shape `[num_windows, window_size, num_features]` and returns
    another numpy array. Depending on `window_size`, the expected output shapes for
    the `feature_extractor` are:

    * `window_size is None`: `[num_new_windows, new_window_size, features]`
    * `window_size is not None`: `[num_windows, features]`

    If `window_size` is set, the extracted features are re-windowed.

    Examples:
        Default

        >>> import rul_datasets
        >>> cmapss = rul_datasets.reader.CmapssReader(fd=1)
        >>> dm = rul_datasets.RulDataModule(cmapss, batch_size=32)

        With Feature Extractor

        >>> import rul_datasets
        >>> import numpy as np
        >>> cmapss = rul_datasets.reader.CmapssReader(fd=1)
        >>> dm = rul_datasets.RulDataModule(
        ...     cmapss,
        ...     batch_size=32,
        ...     feature_extractor=lambda x: np.mean(x, axis=1),
        ...     window_size=10
        ... )

        Only Degraded Validation and Test Samples

        >>> import rul_datasets
        >>> cmapss = rul_datasets.reader.CmapssReader(fd=1)
        >>> dm = rul_datasets.RulDataModule(cmapss, 32, degraded_only=["val", "test"])
    """

    _data: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]

    def __init__(
        self,
        reader: AbstractReader,
        batch_size: int,
        feature_extractor: Optional[Callable] = None,
        window_size: Optional[int] = None,
        degraded_only: Optional[List[Literal["dev", "val", "test"]]] = None,
    ):
        """
        Create a new RUL data module from a reader.

        This data module exposes a training, validation and test data loader for the
        underlying dataset. First, `prepare_data` is called to download and
        pre-process the dataset. Afterward, `setup_data` is called to load all
        splits into memory.

        If a `feature_extractor` is supplied, the data module extracts new features
        from each window of the time series. If `window_size` is `None`,
        it is assumed that the extracted features form a new windows themselves. If
        `window_size` is an int, it is assumed that the extracted features are a
        single feature vectors and should be re-windowed. The expected output shapes
        for the `feature_extractor` are:

        * `window_size is None`: `[num_new_windows, new_window_size, features]`
        * `window_size is not None`: `[num_windows, features]`

        The expected input shape for the `feature_extractor` is always
        `[num_windows, window_size, features]`.

        Args:
            reader: The dataset reader for the desired dataset, e.g., CmapssLoader.
            batch_size: The size of the batches built by the data loaders.
            feature_extractor: A feature extractor that extracts feature vectors from
                               windows.
            window_size: The new window size to apply after the feature extractor.
            degraded_only: Whether to load only degraded samples for the `dev`, 'val'
                           or 'test' split.
        """
        super().__init__()

        self._reader: AbstractReader = reader
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.degraded_only = degraded_only

        if (self.feature_extractor is None) and (self.window_size is not None):
            raise ValueError(
                "A feature extractor has to be supplied "
                "to set a window size for re-windowing."
            )

        hparams = {
            "reader": self.reader.hparams,
            "batch_size": self.batch_size,
            "feature_extractor": (
                str(self.feature_extractor) if self.feature_extractor else None
            ),
            "window_size": self.window_size,
            "degraded_only": self.degraded_only,
        }
        self.save_hyperparameters(hparams)

    @property
    def data(self) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        A dictionary of the training, validation and test splits.

        Each split is a tuple of feature and target tensors.
        The keys are `dev` (training split), `val` (validation split) and `test`
        (test split).
        """
        return self._data

    @property
    def reader(self) -> AbstractReader:
        """The underlying dataset reader."""
        return self._reader

    @property
    def fds(self):
        """Index list of the available subsets of the underlying dataset, i.e.
        `[1, 2, 3, 4]` for `CMAPSS`."""
        return self._reader.fds

    def check_compatibility(self, other: "RulDataModule") -> None:
        """
        Check if another RulDataModule is compatible to be used together with this one.

        RulDataModules can be used together in higher-order data modules,
        e.g. AdaptionDataModule. This function checks if `other` is compatible to
        this data module to do so. It checks the underlying dataset readers, matching
        batch size, feature extractor and window size. If anything is incompatible,
        this function will raise a ValueError.

        Args:
            other: The RulDataModule to check compatibility with.
        """
        try:
            self.reader.check_compatibility(other.reader)
        except ValueError:
            raise ValueError("RulDataModules incompatible on reader level.")

        if not self.batch_size == other.batch_size:
            raise ValueError(
                f"The batch size of both data modules has to be the same, "
                f"{self.batch_size} vs. {other.batch_size}."
            )

        if not self.window_size == other.window_size:
            raise ValueError(
                f"The window size of both data modules has to be the same, "
                f"{self.window_size} vs. {other.window_size}."
            )

    def is_mutually_exclusive(self, other: "RulDataModule") -> bool:
        """
        Check if the other data module is mutually exclusive to this one. See
        [AbstractReader.is_mutually_exclusive]
        [rul_datasets.reader.abstract.AbstractReader.is_mutually_exclusive].

        Args:
            other: Data module to check exclusivity against.

        Returns:
            Whether both data modules are mutually exclusive.
        """
        return self.reader.is_mutually_exclusive(other.reader)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Download and pre-process the underlying data.

        This calls the `prepare_data` function of the underlying reader. All
        previously completed preparation steps are skipped. It is called
        automatically by `pytorch_lightning` and executed on the first GPU in
        distributed mode.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.
        """
        self.reader.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load all splits as tensors into memory and optionally apply feature extractor.

        The splits are placed inside the [data][rul_datasets.core.RulDataModule.data]
        property. If a split is empty, a tuple of empty tensors with the correct
        number of dimensions is created as a placeholder. This ensures compatibility
        with higher-order data modules.

        If the data module was constructed with a `feature_extractor` argument,
        the feature windows are passed to the feature extractor. The resulting,
        new features may be re-windowed.

        Args:
            stage: Ignored. Only for adhering to parent class interface.
        """
        self._data = {
            "dev": self.load_split("dev"),
            "val": self.load_split("val"),
            "test": self.load_split("test"),
        }

    def load_split(
        self,
        split: str,
        alias: Optional[str] = None,
        degraded_only: Optional[bool] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load a split from the underlying reader and apply the feature extractor.

        By setting alias, it is possible to load a split aliased as another split,
        e.g., load the test split and treat it as the dev split. The data of the split
        is loaded, but all pre-processing steps of alias are carried out.

        If `degraded_only` is set, only degraded samples are loaded. This is only
        possible if the underlying reader has a `max_rul` set or `norm_rul` is set to
        `True`. The `degraded_only` argument takes precedence over the `degraded_only`
        of the data module.

        Args:
            split: The desired split to load.
            alias: The split as which the loaded data should be treated.
            degraded_only: Whether to only load degraded samples.

        Returns:
            The feature and target tensors of the split's runs.
        """
        features, targets = self.reader.load_split(split, alias)
        features, targets = self._apply_feature_extractor_per_run(features, targets)
        if degraded_only is None:
            degraded_only = (
                self.degraded_only is not None
                and (alias or split) in self.degraded_only
            )
        if degraded_only:
            self._filter_out_healthy(features, targets)

        return features, targets

    def _apply_feature_extractor_per_run(
        self, features: List[np.ndarray], targets: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        extracted = [self._extract_and_window(f, t) for f, t in zip(features, targets)]
        features, targets = zip(*extracted) if extracted else ((), ())

        return list(features), list(targets)

    def _extract_and_window(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.feature_extractor is not None:
            features, targets = self.feature_extractor(features, targets)
        if self.window_size is not None:
            cutoff = self.window_size - 1
            features = utils.extract_windows(features, self.window_size)
            targets = targets[cutoff:]

        return features, targets

    def _filter_out_healthy(self, features, targets):
        if self.reader.max_rul is not None:
            thresh = self.reader.max_rul
        elif hasattr(self.reader, "norm_rul") and self.reader.norm_rul:
            thresh = 1.0
        else:
            raise ValueError(
                "Cannot filter degraded samples if no max_rul is set and "
                "norm_rul is False."
            )
        for i in range(len(targets)):
            degraded = targets[i] < thresh
            features[i] = features[i][degraded]
            targets[i] = targets[i][degraded]

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a [data loader][torch.utils.data.DataLoader] for the training split.

        The data loader is configured to shuffle the data. The `pin_memory` option is
        activated to achieve maximum transfer speed to the GPU. The data loader is also
        configured to drop the last batch of the data if it would only contain one
        sample.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.

        Returns:
            The training data loader
        """
        dataset = self.to_dataset("dev")
        drop_last = len(dataset) % self.batch_size == 1
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a [data loader][torch.utils.data.DataLoader] for the validation split.

        The data loader is configured to leave the data unshuffled. The `pin_memory`
        option is activated to achieve maximum transfer speed to the GPU.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.

        Returns:
            The validation data loader
        """
        return DataLoader(
            self.to_dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a [data loader][torch.utils.data.DataLoader] for the test split.

        The data loader is configured to leave the data unshuffled. The `pin_memory`
        option is activated to achieve maximum transfer speed to the GPU.

        The whole split is held in memory. Therefore, the `num_workers` are set to
        zero which uses the main process for creating batches.

        Args:
            *args: Ignored. Only for adhering to parent class interface.
            **kwargs: Ignored. Only for adhering to parent class interface.

        Returns:
            The test data loader
        """
        return DataLoader(
            self.to_dataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def to_dataset(self, split: str, alias: Optional[str] = None) -> "RulDataset":
        """
        Create a dataset of a split.

        This convenience function creates a plain [tensor dataset]
        [torch.utils.data.TensorDataset] to use outside the `rul_datasets` library.

        The data placed inside the dataset will be from the specified `split`. If
        `alias` is set, the loaded data will be treated as if from the `alias` split.
        For example, one could load the test data and treat them as if it was the
        training data. This may be useful for inductive domain adaption.

        Args:
            split: The split to place inside the dataset.
            alias: The split the loaded data should be treated as.

        Returns:
            A dataset containing the requested split.
        """
        if (alias is None) or (split == alias):
            features, targets = self._data[split]
        else:
            features, targets = self.load_split(split, alias)
        split_dataset = RulDataset(features, targets)

        return split_dataset


class RulDataset(Dataset):
    """Internal dataset to hold multiple runs.

    Its length is the sum of all runs' lengths.
    """

    def __init__(
        self,
        features: List[np.ndarray],
        *targets: List[np.ndarray],
        copy_tensors: bool = False,
    ) -> None:
        """
        Create a new dataset from multiple runs.

        If `copy_tensors` is true, the tensors are copied to avoid side effects when
        modifying them. Otherwise, the tensors use the same memory as the original
        Numpy arrays to save space.

        Args:
            features: The features of each run.
            targets: The targets of each run.
            copy_tensors: Whether to copy the tensors or not.
        """
        super().__init__()

        self.features = features
        self.targets = targets
        self.copy_tensors = copy_tensors

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported by this dataset.")
        for i in range(len(self.features)):
            if index < len(self.features[i]):
                tensor_feat = utils.feature_to_tensor(
                    self.features[i][index], copy=self.copy_tensors
                )
                tensor_tar = tuple(torch.as_tensor(t[i][index]) for t in self.targets)
                return tensor_feat, *tensor_tar
            else:
                index -= len(self.features[i])

        raise IndexError(f"Index {index} out of range.")

    def __len__(self) -> int:
        return sum(len(f) for f in self.features)


class PairedRulDataset(IterableDataset):
    """A dataset of sample pairs drawn from the same time series.

    The dataset uses the runs exactly as loaded by the passed data modules. Options
    like `degraded_only` need to be set there."""

    def __init__(
        self,
        dms: List[RulDataModule],
        split: str,
        num_samples: int,
        min_distance: int,
        deterministic: bool = False,
        mode: str = "linear",
    ):
        super().__init__()

        self.dms = dms
        self.split = split
        self.min_distance = min_distance
        self.num_samples = num_samples
        self.deterministic = deterministic
        self.mode = mode

        for dm in self.dms:
            dm.check_compatibility(self.dms[0])

        self._run_domain_idx: np.ndarray
        self._features: List[np.ndarray]
        self._labels: List[np.ndarray]
        self._prepare_datasets()

        self._max_rul = self._get_max_rul()
        self._curr_iter = 0
        self._rng = self._reset_rng()
        if mode == "linear":
            self._get_pair_func = self._get_pair_idx
        elif mode == "piecewise":
            self._get_pair_func = self._get_pair_idx_piecewise
        elif mode == "labeled":
            self._get_pair_func = self._get_labeled_pair_idx

    def _get_max_rul(self) -> Optional[int]:
        max_ruls = [dm.reader.max_rul for dm in self.dms]
        if all(m is None for m in max_ruls):
            max_rul = None
        elif any(m is None for m in max_ruls):
            raise ValueError(
                "PairedRulDataset needs a set max_rul for all or none of the readers "
                "but at least one and not all of them has None."
            )
        else:
            max_rul = max(cast(List[int], max_ruls))

        return max_rul

    def _prepare_datasets(self):
        run_domain_idx = []
        features = []
        labels = []
        for domain_idx, dm in enumerate(self.dms):
            run_features, run_labels = dm.data[self.split]
            for feat, lab in zip(run_features, run_labels):
                if len(feat) > self.min_distance:
                    run_domain_idx.append(domain_idx)
                    features.append(feat)
                    labels.append(lab)

        self._run_domain_idx = np.array(run_domain_idx)
        self._features = features
        self._labels = labels

    def _reset_rng(self, seed=42) -> np.random.Generator:
        return np.random.default_rng(seed=seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        self._curr_iter = 0
        worker_info = get_worker_info()
        if self.deterministic and worker_info is not None:
            raise RuntimeError(
                "PairedDataset cannot run deterministic in multiprocessing"
            )
        elif self.deterministic:
            self._rng = self._reset_rng()
        elif worker_info is not None:
            self._rng = self._reset_rng(worker_info.seed)

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._curr_iter < self.num_samples:
            run_idx, anchor_idx, query_idx, dist, domain_label = self._get_pair_func()
            self._curr_iter += 1
            run = self._features[run_idx]
            return self._build_pair(run, anchor_idx, query_idx, dist, domain_label)
        else:
            raise StopIteration

    def _get_pair_idx(self) -> Tuple[int, int, int, Union[int, float], int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = cast(int, self._run_domain_idx[chosen_run_idx])
        chosen_run = self._features[chosen_run_idx]

        run_length = chosen_run.shape[0]
        anchor_idx = self._rng.integers(
            low=0,
            high=run_length - self.min_distance,
        )
        end_idx = min(run_length, anchor_idx + (self._max_rul or 999999))
        query_idx = self._rng.integers(
            low=anchor_idx + self.min_distance,
            high=end_idx,
        )
        distance = query_idx - anchor_idx

        return chosen_run_idx, anchor_idx, query_idx, distance, domain_label

    def _get_pair_idx_piecewise(self) -> Tuple[int, int, int, Union[int, float], int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = cast(int, self._run_domain_idx[chosen_run_idx])
        chosen_run = self._features[chosen_run_idx]

        run_length = chosen_run.shape[0]
        middle_idx = run_length // 2
        anchor_idx = self._rng.integers(
            low=0,
            high=run_length - self.min_distance,
        )
        end_idx = (
            middle_idx if anchor_idx < (middle_idx - self.min_distance) else run_length
        )
        query_idx = self._rng.integers(
            low=anchor_idx + self.min_distance,
            high=end_idx,
        )
        distance = query_idx - anchor_idx if anchor_idx > middle_idx else 0

        return chosen_run_idx, anchor_idx, query_idx, distance, domain_label

    def _get_labeled_pair_idx(self) -> Tuple[int, int, int, Union[int, float], int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = cast(int, self._run_domain_idx[chosen_run_idx])
        chosen_run = self._features[chosen_run_idx]
        chosen_labels = self._labels[chosen_run_idx]

        run_length = chosen_run.shape[0]
        anchor_idx = self._rng.integers(
            low=0,
            high=run_length - self.min_distance,
        )
        query_idx = self._rng.integers(
            low=anchor_idx + self.min_distance,
            high=run_length,
        )
        # RUL label difference is negative time step difference
        distance = (chosen_labels[anchor_idx] - chosen_labels[query_idx]).item()

        return chosen_run_idx, anchor_idx, query_idx, distance, domain_label

    def _build_pair(
        self,
        run: np.ndarray,
        anchor_idx: int,
        query_idx: int,
        distance: Union[int, float],
        domain_label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = utils.feature_to_tensor(run[anchor_idx], torch.float32)
        queries = utils.feature_to_tensor(run[query_idx], torch.float32)
        domain_tensor = torch.tensor(domain_label, dtype=torch.float)
        distances = torch.tensor(distance, dtype=torch.float)
        if self._max_rul is not None:  # normalize only if max_rul is set
            distances /= self._max_rul
            distances = torch.clamp_max(distances, max=1)

        return anchors, queries, distances, domain_tensor
