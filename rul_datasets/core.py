"""Basic data modules for experiments involving only a single subset of any RUL
dataset. """

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset, get_worker_info

from rul_datasets.reader import AbstractReader


class RulDataModule(pl.LightningDataModule):
    """
    A [data module][pytorch_lightning.core.LightningDataModule] to provide windowed
    time series features with RUL targets. It exposes the splits of the underlying
    dataset for easy usage with PyTorch and PyTorch Lightning.

    The data module implements the `hparams` property used by PyTorch Lightning to
    save hyperparameters to checkpoints. It retrieves the hyperparameters of its
    underlying reader and adds the batch size to them.

    Examples:
        >>> import rul_datasets
        >>> cmapss = rul_datasets.reader.CmapssReader(fd=1)
        >>> dm = rul_datasets.RulDataModule(cmapss, batch_size=32)
    """

    _data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]

    def __init__(self, reader: AbstractReader, batch_size: int):
        """
        Create a new RUL data module from a reader.

        This data module exposes a training, validation and test data loader for the
        underlying dataset. First, `prepare_data` is called to download and
        pre-process the dataset. Afterwards, `setup_data` is called to load all
        splits into memory.

        Args:
            reader: The dataset reader for the desired dataset, e.g. CmapssLoader.
            batch_size: The size of the batches build by the data loaders
        """
        super().__init__()

        self._reader: AbstractReader = reader
        self.batch_size: int = batch_size

        hparams = deepcopy(self.reader.hparams)
        hparams["batch_size"] = self.batch_size
        self.save_hyperparameters(hparams)

    @property
    def data(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
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
        this data module to do so. It checks the underlying dataset loaders and for
        matching batch size. If anything is incompatible, this function will raise a
        ValueError.

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
                f"{self.batch_size} vs. {other.batch_size}"
            )

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
        Load all splits as tensors into memory.

        The splits are placed inside the [data][rul_datasets.core.RulDataModule.data]
        property. If a split is empty, a tuple of empty tensors with the correct
        number of dimensions is created as a placeholder. This ensures compatibility
        with higher-order data modules.

        Args:
            stage: Ignored. Only for adhering to parent class interface.
        """
        self._data = {
            "dev": self._setup_split("dev"),
            "val": self._setup_split("val"),
            "test": self._setup_split("test"),
        }

    def _setup_split(self, split):
        features, targets = self.reader.load_split(split)
        if features:
            features = torch.cat(features)
            targets = torch.cat(targets)
        else:
            features = torch.empty(0, 0, 0)
            targets = torch.empty(0)

        return features, targets

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """
        Create a [data loader][torch.utils.data.DataLoader] for the training split.

        The data loader is configured to shuffle the data. The `pin_memory` option is
        activated to achieve maximum transfer speed to the GPU. The data loader is also
        configured to drop the last batch of the data if it would only contain one
        sample.

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

    def to_dataset(self, split: str) -> TensorDataset:
        """
        Create a dataset of a split.

        This convenience function creates a plain [tensor dataset]
        [torch.utils.data.TensorDataset] to use outside the `rul_datasets` library.

        Args:
            split: The split to place inside the dataset.
        Returns:
            A dataset containing the requested split.
        """
        features, targets = self._data[split]
        split_dataset = TensorDataset(features, targets)

        return split_dataset


class PairedRulDataset(IterableDataset):
    """TODO."""

    def __init__(
        self,
        readers: List[AbstractReader],
        split: str,
        num_samples: int,
        min_distance: int,
        deterministic: bool = False,
        mode: str = "linear",
    ):
        super().__init__()

        self.readers = readers
        self.split = split
        self.min_distance = min_distance
        self.num_samples = num_samples
        self.deterministic = deterministic
        self.mode = mode

        for reader in self.readers:
            reader.check_compatibility(self.readers[0])

        self._run_domain_idx: np.ndarray
        self._features: List[torch.Tensor]
        self._labels: List[torch.Tensor]
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

    def _get_max_rul(self):
        max_ruls = [reader.max_rul for reader in self.readers]
        if any(m is None for m in max_ruls):
            raise ValueError(
                "PairedRulDataset needs a set max_rul for all readers "
                "but at least one of them has is None."
            )
        max_rul = max(max_ruls)

        return max_rul

    def _prepare_datasets(self):
        run_domain_idx = []
        features = []
        labels = []
        for domain_idx, reader in enumerate(self.readers):
            run_features, run_labels = reader.load_split(self.split)
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

    def _get_pair_idx(self) -> Tuple[int, int, int, int, int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = self._run_domain_idx[chosen_run_idx]
        chosen_run = self._features[chosen_run_idx]

        run_length = chosen_run.shape[0]
        anchor_idx = self._rng.integers(
            low=0,
            high=run_length - self.min_distance,
        )
        end_idx = min(run_length, anchor_idx + self._max_rul)
        query_idx = self._rng.integers(
            low=anchor_idx + self.min_distance,
            high=end_idx,
        )
        distance = query_idx - anchor_idx

        return chosen_run_idx, anchor_idx, query_idx, distance, domain_label

    def _get_pair_idx_piecewise(self) -> Tuple[int, int, int, int, int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = self._run_domain_idx[chosen_run_idx]
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

    def _get_labeled_pair_idx(self) -> Tuple[int, int, int, int, int]:
        chosen_run_idx = self._rng.integers(0, len(self._features))
        domain_label = self._run_domain_idx[chosen_run_idx]
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
        distance = int(chosen_labels[anchor_idx] - chosen_labels[query_idx])

        return chosen_run_idx, anchor_idx, query_idx, distance, domain_label

    def _build_pair(
        self,
        run: torch.Tensor,
        anchor_idx: int,
        query_idx: int,
        distance: int,
        domain_label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = run[anchor_idx]
        queries = run[query_idx]
        domain_tensor = torch.tensor(domain_label, dtype=torch.float)
        distances = torch.tensor(distance, dtype=torch.float) / self._max_rul
        distances = torch.clamp_max(distances, max=1)  # max distance is max_rul

        return anchors, queries, distances, domain_tensor
