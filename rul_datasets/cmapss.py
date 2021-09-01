from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from rul_datasets.loader import CMAPSSLoader


class CMAPSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fd: int,
        batch_size: int,
        window_size: int = None,
        max_rul: int = 125,
        percent_broken: float = None,
        percent_fail_runs: Union[float, List[int]] = None,
        feature_select: List[int] = None,
        truncate_val: bool = False,
    ):
        super().__init__()

        self._loader = CMAPSSLoader(
            fd,
            window_size,
            max_rul,
            percent_broken,
            percent_fail_runs,
            feature_select,
            truncate_val,
        )

        self.batch_size = batch_size
        self.fd = self._loader.fd
        self.window_size = self._loader.window_size
        self.max_rul = self._loader.max_rul
        self.percent_broken = self._loader.percent_broken
        self.percent_fail_runs = self._loader.percent_fail_runs
        self.feature_select = self._loader.feature_select
        self.truncate_val = self._loader.truncate_val

        self.save_hyperparameters(
            {
                "fd": self.fd,
                "batch_size": self.batch_size,
                "window_size": self.window_size,
                "max_rul": self.max_rul,
                "percent_broken": self.percent_broken,
                "percent_fail_runs": self.percent_fail_runs,
                "truncate_val": self.truncate_val,
            }
        )

        self.data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        super().transfer_batch_to_device(batch, device)

    @classmethod
    def from_loader(cls, loader: CMAPSSLoader, batch_size: int):
        return cls(
            loader.fd,
            batch_size,
            loader.window_size,
            loader.max_rul,
            loader.percent_broken,
            loader.percent_fail_runs,
            loader.feature_select,
            loader.truncate_val,
        )

    def prepare_data(self, *args, **kwargs):
        self._loader.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.data["dev"] = self._setup_split("dev")
        self.data["val"] = self._setup_split("val")
        self.data["test"] = self._setup_split("test")

    def _setup_split(self, split):
        features, targets = self._loader.load_split(split)
        if features:
            features = torch.cat(features)
            targets = torch.cat(targets)
        else:
            features = torch.empty(
                0, len(self._loader.feature_select), self.window_size
            )
            targets = torch.empty(0)

        return features, targets

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
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

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.to_dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.to_dataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def to_dataset(self, split):
        features, targets = self.data[split]
        split_dataset = TensorDataset(features, targets)

        return split_dataset


class PairedCMAPSS(IterableDataset):
    def __init__(
        self,
        loaders: List[CMAPSSLoader],
        split: str,
        num_samples: int,
        min_distance: int,
        deterministic: bool = False,
        mode: str = "linear",
    ):
        super().__init__()

        self.loaders = loaders
        self.split = split
        self.min_distance = min_distance
        self.num_samples = num_samples
        self.deterministic = deterministic
        self.mode = mode

        if not all(d.window_size == self.loaders[0].window_size for d in self.loaders):
            window_sizes = [d.window_size for d in self.loaders]
            raise ValueError(
                f"Datasets to be paired do not have "
                f"the same window size, but {window_sizes}"
            )

        self._run_domain_idx: np.ndarray
        self._features: List[torch.Tensor]
        self._labels: List[torch.Tensor]
        self._prepare_datasets()

        self._max_rul = max(loader.max_rul for loader in self.loaders)
        self._current_iteration = 0
        self._rng = self._reset_rng()
        if mode == "linear":
            self._get_pair_func = self._get_pair_idx
        elif mode == "piecewise":
            self._get_pair_func = self._get_pair_idx_piecewise
        elif mode == "labeled":
            self._get_pair_func = self._get_labeled_pair_idx

    def _prepare_datasets(self):
        run_domain_idx = []
        features = []
        labels = []
        for domain_idx, loader in enumerate(self.loaders):
            run_features, run_labels = loader.load_split(self.split)
            for feat, lab in zip(run_features, run_labels):
                if len(feat) > self.min_distance:
                    run_domain_idx.append(domain_idx)
                    features.append(feat)
                    labels.append(lab)

        self._run_domain_idx = np.array(run_domain_idx)
        self._features = features
        self._labels = labels

    def _reset_rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        self._current_iteration = 0
        if self.deterministic:
            self._rng = self._reset_rng()

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._current_iteration < self.num_samples:
            self._current_iteration += 1
            idx = self._get_pair_func()
            return self._build_pair(*idx)
        else:
            raise StopIteration

    def _get_pair_idx(self) -> Tuple[torch.Tensor, int, int, int, int]:
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

        return chosen_run, anchor_idx, query_idx, distance, domain_label

    def _get_pair_idx_piecewise(self) -> Tuple[torch.Tensor, int, int, int, int]:
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

        return chosen_run, anchor_idx, query_idx, distance, domain_label

    def _get_labeled_pair_idx(self) -> Tuple[torch.Tensor, int, int, int, int]:
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

        return chosen_run, anchor_idx, query_idx, distance, domain_label

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
