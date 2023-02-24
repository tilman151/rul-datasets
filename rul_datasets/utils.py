import os
from typing import List, Optional, Callable, Dict, Tuple

import numpy as np
import requests  # type: ignore
import torch
from tqdm import tqdm  # type: ignore


def get_files_in_path(path: str, condition: Optional[Callable] = None) -> List[str]:
    """
    Return the paths of all files in a path that satisfy a condition in alphabetical
    order.

    If the condition is `None` all files are returned.

    Args:
        path: the path to look into
        condition: the include-condition for files
    Returns:
        all files that satisfy the condition in alphabetical order
    """
    if condition is None:
        feature_files = [f for f in os.listdir(path)]
    else:
        feature_files = [f for f in os.listdir(path) if condition(f)]
    feature_files = sorted(os.path.join(path, f) for f in feature_files)

    return feature_files


def get_targets_from_file_paths(
    file_paths: Dict[int, List[str]], timestep_from_file_path: Callable
) -> Dict[int, np.ndarray]:
    """
    Create the RUL targets based on the file paths of the feature files.

    The function extracts the feature file path from each path. The supplied
    conversion function extracts the time step from it. Afterwards the RUL is
    calculated by subtracting each time step from the maximum time step plus 1.

    Args:
        file_paths: runs represented as dict of feature file paths
        timestep_from_file_path: Function to convert a feature file path to a time step

    Returns:
        A list of RUL target arrays for each run
    """
    targets = {}
    for run_idx, run_files in file_paths.items():
        run_targets = np.empty(len(run_files))
        for i, file_path in enumerate(run_files):
            run_targets[i] = timestep_from_file_path(file_path)
        run_targets = np.max(run_targets) - run_targets + 1
        targets[run_idx] = run_targets

    return targets


def extract_windows(seq: np.ndarray, window_size: int) -> np.ndarray:
    """
    Extract sliding windows from a sequence.

    The step size is considered to be one, which results in `len(seq) - window_size +
    1` extracted windows. The resulting array has the shape [num_windows, window_size,
    num_channels].

    Args:
        seq: sequence to extract windows from
        window_size: length of the sliding window
    Returns:
        array of sliding windows
    """
    if window_size > len(seq):
        raise ValueError(
            f"Cannot extract windows of size {window_size} "
            f"from a sequence of length {len(seq)}."
        )

    num_frames = seq.shape[0] - window_size + 1
    window_idx = np.arange(window_size)[None, :] + np.arange(num_frames)[:, None]
    windows = seq[window_idx]

    return windows


def download_file(url: str, save_path: str) -> None:
    response = requests.get(url, stream=True)
    if not response.status_code == 200:
        raise RuntimeError(f"Download failed. Server returned {response.status_code}")
    content_len = int(response.headers["Content-Length"]) // 1024
    with open(save_path, mode="wb") as f:
        for data in tqdm(response.iter_content(chunk_size=1024), total=content_len):
            f.write(data)


def to_tensor(
    features: List[np.ndarray], *targets: List[np.ndarray]
) -> Tuple[List[torch.Tensor], ...]:
    dtype = torch.float32
    tensor_feats = [feature_to_tensor(f, dtype) for f in features]
    tensor_targets = [
        [torch.tensor(t, dtype=dtype) for t in target] for target in targets
    ]

    return tensor_feats, *tensor_targets


def feature_to_tensor(features: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    if len(features.shape) == 2:
        return torch.tensor(features, dtype=dtype).permute(1, 0)
    else:
        return torch.tensor(features, dtype=dtype).permute(0, 2, 1)
