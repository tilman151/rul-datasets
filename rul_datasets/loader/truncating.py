from typing import List, Tuple, Iterable, Union

import numpy as np


def truncate_runs(
    features: List[np.ndarray],
    targets: List[np.ndarray],
    percent_broken: float = None,
    included_runs: Union[float, Iterable[int]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Truncate the number of runs
    if included_runs is not None:
        features, targets = _truncate_included(features, targets, included_runs)

    # Truncate the number of samples per run, starting at failure
    if percent_broken is not None and percent_broken < 1:
        features, targets = _truncate_broken(features, targets, percent_broken)

    return features, targets


def _truncate_included(
    features: List[np.ndarray],
    targets: List[np.ndarray],
    included_runs: Union[float, Iterable[int]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if isinstance(included_runs, float):
        features, targets = _truncate_included_by_percentage(
            features, targets, included_runs
        )
    elif isinstance(included_runs, Iterable):
        features, targets = _truncate_included_by_index(
            features, targets, included_runs
        )

    return features, targets


def _truncate_broken(
    features: List[np.ndarray], targets: List[np.ndarray], percent_broken: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    for i, run in enumerate(features):
        num_cycles = int(percent_broken * len(run))
        features[i] = run[:num_cycles]
        targets[i] = targets[i][:num_cycles]

    return features, targets


def _truncate_included_by_index(
    features: List[np.ndarray], targets: List[np.ndarray], included_idx: Iterable[int]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    features = [features[i] for i in included_idx]
    targets = [targets[i] for i in included_idx]

    return features, targets


def _truncate_included_by_percentage(
    features: List[np.ndarray], targets: List[np.ndarray], percent_included: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_runs = int(percent_included * len(features))
    features = features[:num_runs]
    targets = targets[:num_runs]

    return features, targets
