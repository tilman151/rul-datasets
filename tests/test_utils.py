import os
import random

import numpy as np
import pytest

from rul_datasets import utils


def dummy_file_name_to_timestep(file_path):
    file_name = os.path.basename(file_path)
    time_step = int(file_name.replace(".csv", ""))

    return time_step


def ordered_file_paths():
    file_paths = []
    time_steps = []
    for _ in range(2):
        run_steps = list(range(1, random.randint(10, 20)))
        time_steps.append(run_steps)
        file_paths.append([f"foo/bar/{i}.csv" for i in run_steps])

    return file_paths, time_steps


def shuffled_file_paths():
    file_paths = []
    time_steps = []
    for _ in range(2):
        run_steps = list(range(1, random.randint(10, 20)))
        random.shuffle(run_steps)
        time_steps.append(run_steps)
        file_paths.append([f"foo/bar/{i}.csv" for i in run_steps])

    return file_paths, time_steps


@pytest.mark.parametrize("file_path_func", [ordered_file_paths, shuffled_file_paths])
def test_get_targets_from_file_paths(file_path_func):
    file_paths, time_steps = file_path_func()
    targets = utils.get_targets_from_file_paths(file_paths, dummy_file_name_to_timestep)

    for run_targets, run_steps in zip(targets, time_steps):
        sorted_idx = np.argsort(run_targets)
        sorted_targets = run_targets[sorted_idx]
        sorted_steps = np.array(run_steps)[sorted_idx[::-1]]  # reverse sorted
        assert np.all(sorted_targets == sorted_steps)
