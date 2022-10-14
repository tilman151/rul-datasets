from typing import List, Optional, Tuple, Union
from unittest import mock

import numpy as np
import pytest

from rul_datasets import loader


class DummyLoader(loader.AbstractLoader):
    fd: int
    window_size: int
    max_rul: int
    percent_broken: Optional[float] = None
    percent_fail_runs: Optional[Union[float, List[int]]] = None
    truncate_val: bool = True

    _NUM_TRAIN_RUNS = {1: 100}

    def _default_window_size(self, fd):
        return 15

    def prepare_data(self):
        pass

    def _load_complete_split(
        self, split: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return [], []


class TestAbstractLoader:
    @mock.patch("rul_datasets.loader.truncating.truncate_runs", return_value=([], []))
    def test_truncation_dev_split(self, mock_truncate_runs):
        this = DummyLoader(1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8)
        this.load_split("dev")
        mock_truncate_runs.assert_called_with([], [], 0.2, 0.8)

    @mock.patch("rul_datasets.loader.truncating.truncate_runs", return_value=([], []))
    def test_truncation_val_split(self, mock_truncate_runs):
        this = DummyLoader(1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8)
        this.load_split("val")
        mock_truncate_runs.assert_not_called()

        this = DummyLoader(
            1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8, truncate_val=True
        )
        this.load_split("val")
        mock_truncate_runs.assert_called_with([], [], 0.2, 0.8)

    @mock.patch("rul_datasets.loader.truncating.truncate_runs", return_value=([], []))
    def test_truncation_test_split(self, mock_truncate_runs):
        this = DummyLoader(1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8)
        this.load_split("val")
        mock_truncate_runs.assert_not_called()

    def test_check_compatibility(self):
        this = DummyLoader(1, 30, 125)
        this.check_compatibility(DummyLoader(1, 30, 125))
        with pytest.raises(ValueError):
            this.check_compatibility(DummyLoader(1, 20, 125))
        with pytest.raises(ValueError):
            this.check_compatibility(DummyLoader(1, 30, 120))

    def test_get_compatible_same(self):
        this = DummyLoader(1, 30, 125)
        other = this.get_compatible()
        this.check_compatibility(other)
        assert other is not this
        assert this.fd == other.fd
        assert 30 == other.window_size == this.window_size
        assert this.max_rul == other.max_rul
        assert this.percent_broken == other.percent_broken
        assert this.percent_fail_runs == other.percent_fail_runs
        assert this.truncate_val == other.truncate_val

    def test_get_compatible_different(self):
        this = DummyLoader(1, 30, 125)
        other = this.get_compatible(2, 0.2, 0.8, False)
        this.check_compatibility(other)
        assert other is not this
        assert 2 == other.fd
        assert 15 == other.window_size
        assert 15 == this.window_size  # original loader is made compatible
        assert this.max_rul == other.max_rul
        assert 0.2 == other.percent_broken
        assert 0.8 == other.percent_fail_runs
        assert False == other.truncate_val

    def test_get_complement_percentage(self):
        this = DummyLoader(1, 30, 125, percent_fail_runs=0.8)
        other = this.get_complement(0.8, False)
        assert other.percent_fail_runs == list(range(80, 100))
        assert 0.8 == other.percent_broken
        assert not other.truncate_val

    def test_get_complement_idx(self):
        this = DummyLoader(1, 30, 125, percent_fail_runs=list(range(80)))
        other = this.get_complement(0.8, False)
        assert other.percent_fail_runs == list(range(80, 100))
        assert 0.8 == other.percent_broken
        assert not other.truncate_val

    def test_get_complement_empty(self):
        this = DummyLoader(1, 30, 125)  # Uses all runs
        other = this.get_complement(0.8, False)
        assert not other.percent_fail_runs  # Complement is empty
        assert 0.8 == other.percent_broken
        assert not other.truncate_val
