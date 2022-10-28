"""A module with functions for efficient saving and loading of RUL features and
targets. """

import os.path
from typing import Tuple, List, Dict, Literal, Optional

import numpy as np
from tqdm import tqdm  # type: ignore


def save(save_path: str, features: np.ndarray, targets: np.ndarray) -> None:
    """
    Save features and targets of a run to .npy files.

    The arrays are saved to separate .npy files to enable memmap mode in case RAM is
    short. The files are saved as <save_path>_features.npy and
    <save_path>_targets.npy. To load the files, supply the same `save_path` to the
    [load][rul_datasets.reader.saving.load] function. If the `save_path` does not have
    the .npy file extension .npy will be appended.

    Args:
          save_path: The path including file name to save the arrays to.
          features: The feature array to save.
          targets: The targets array to save.
    """
    feature_path = _get_feature_path(save_path)
    np.save(feature_path, features, allow_pickle=False)
    target_path = _get_target_path(save_path)
    np.save(target_path, targets, allow_pickle=False)


def load(save_path: str, memmap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and targets of a run from .npy files.

    This method is used to restore runs that were saved with the [save]
    [rul_datasets.reader.saving.save] function. If the runs are too large for the RAM,
    `memmap` can be set to True to avoid reading them completely to memory. This
    results in slower processing, though.

    Args:
        save_path: Path that was supplied to the
                   [save][rul_datasets.reader.saving.save] function.
        memmap: whether to use memmap to avoid loading the whole run into memory
    Returns:
        features: The feature array saved in `save_path`
        targets: The target array saved in `save_path`
    """
    memmap_mode: Optional[Literal["r"]] = "r" if memmap else None
    feature_path = _get_feature_path(save_path)
    features = np.load(feature_path, memmap_mode, allow_pickle=False)
    target_path = _get_target_path(save_path)
    targets = np.load(target_path, memmap_mode, allow_pickle=False)

    return features, targets


def load_multiple(
    save_paths: List[str], memmap: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load multiple runs with the [load][rul_datasets.reader.saving.load] function.

    Args:
        save_paths: The list of run files to load.
        memmap: See [load][rul_datasets.reader.saving.load]
    Returns:
        features: The feature arrays saved in `save_paths`
        targets: The target arrays saved in `save_paths`
    """
    runs = [load(save_path, memmap) for save_path in save_paths]
    features, targets = [list(x) for x in zip(*runs)]

    return features, targets


def exists(save_path: str) -> bool:
    """
    Return if the files resulting from a `save` call with `save_path` exist.

    Args:
        save_path: the `save_path` the [save][rul_datasets.reader.saving.save]
                   function was called with
    Returns:
        Whether the files exist
    """
    feature_path = _get_feature_path(save_path)
    target_path = _get_target_path(save_path)

    return os.path.exists(feature_path) and os.path.exists(target_path)


def _get_feature_path(save_path):
    if save_path.endswith(".npy"):
        save_path = save_path[:-4]
    feature_path = f"{save_path}_features.npy"

    return feature_path


def _get_target_path(save_path):
    if save_path.endswith(".npy"):
        save_path = save_path[:-4]
    target_path = f"{save_path}_targets.npy"

    return target_path


def load_raw(
    file_paths: Dict[int, List[str]],
    window_size: int,
    columns: List[int],
    skip_rows: int = 0,
) -> Dict[int, np.ndarray]:
    features = {}
    for run_idx, run_files in tqdm(file_paths.items(), desc="Runs"):
        run_features = np.empty((len(run_files), window_size, len(columns)))
        for i, file_path in enumerate(tqdm(run_files, desc="Files", leave=False)):
            run_features[i] = _load_raw_features(file_path, skip_rows, columns)
        features[run_idx] = run_features

    return features


def _load_raw_features(
    file_path: str, skip_rows: int, columns: List[int]
) -> np.ndarray:
    try:
        features = _load_raw_unsafe(file_path, skip_rows)
    except ValueError:
        _replace_delimiters(file_path)
        features = _load_raw_unsafe(file_path, skip_rows)
    features = features[:, columns]

    return features


def _load_raw_unsafe(file_path: str, skip_rows: int) -> np.ndarray:
    return np.loadtxt(file_path, skiprows=skip_rows, delimiter=",")


def _replace_delimiters(file_path: str) -> None:
    with open(file_path, mode="r+t") as f:
        content = f.read()
        f.seek(0)
        content = content.replace(";", ",")
        f.write(content)
        f.truncate()
