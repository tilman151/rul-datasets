from typing import List, Optional, Tuple, Union
from unittest import mock

import numpy as np
import pytest

from rul_datasets import reader


class DummyAbstractReader(reader.AbstractReader):
    fd: int
    window_size: int
    max_rul: int
    percent_broken: Optional[float] = None
    percent_fail_runs: Optional[Union[float, List[int]]] = None
    truncate_val: bool = True

    _NUM_TRAIN_RUNS = {1: 100}

    @property
    def dataset_name(self) -> str:
        return "dummy_abstract"

    @property
    def fds(self):
        return [1]

    def default_window_size(self, fd):
        return 15

    def prepare_data(self):
        pass

    def load_complete_split(
        self, split: str, alias: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return [], []


class TestAbstractLoader:
    @mock.patch("rul_datasets.reader.truncating.truncate_runs", return_value=([], []))
    def test_truncation_dev_split(self, mock_truncate_runs):
        this = DummyAbstractReader(
            1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8
        )
        this.load_split("dev")
        mock_truncate_runs.assert_called_with([], [], 0.2, 0.8, False)

    @mock.patch("rul_datasets.reader.truncating.truncate_runs", return_value=([], []))
    def test_truncation_val_split(self, mock_truncate_runs):
        this = DummyAbstractReader(
            1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8
        )
        this.load_split("val")
        mock_truncate_runs.assert_not_called()

        this = DummyAbstractReader(
            1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8, truncate_val=True
        )
        this.load_split("val")
        mock_truncate_runs.assert_called_with([], [], 0.2, degraded_only=False)

    @mock.patch("rul_datasets.reader.truncating.truncate_runs", return_value=([], []))
    def test_truncation_test_split(self, mock_truncate_runs):
        this = DummyAbstractReader(
            1, 30, 125, percent_broken=0.2, percent_fail_runs=0.8
        )
        this.load_split("val")
        mock_truncate_runs.assert_not_called()

    def test_check_compatibility(self):
        this = DummyAbstractReader(1, 30, 125)
        this.check_compatibility(DummyAbstractReader(1, 30, 125))
        with pytest.raises(ValueError):
            this.check_compatibility(DummyAbstractReader(1, 20, 125))
        with pytest.raises(ValueError):
            this.check_compatibility(DummyAbstractReader(1, 30, 120))

    def test_get_compatible_same(self):
        this = DummyAbstractReader(1, 30, 125)
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
        this = DummyAbstractReader(1, 30, 125)
        other = this.get_compatible(2, 0.2, 0.8, False)
        this.check_compatibility(other)
        assert other is not this
        assert 2 == other.fd
        assert 30 == other.window_size
        assert 30 == this.window_size  # original reader overrides window size
        assert this.max_rul == other.max_rul
        assert 0.2 == other.percent_broken
        assert 0.8 == other.percent_fail_runs
        assert not other.truncate_val

    def test_get_complement_percentage(self):
        this = DummyAbstractReader(1, 30, 125, percent_fail_runs=0.8)
        other = this.get_complement(0.8, False)
        assert other.percent_fail_runs == list(range(80, 100))
        assert 0.8 == other.percent_broken
        assert not other.truncate_val

    def test_get_complement_idx(self):
        this = DummyAbstractReader(1, 30, 125, percent_fail_runs=list(range(80)))
        other = this.get_complement(0.8, False)
        assert other.percent_fail_runs == list(range(80, 100))
        assert 0.8 == other.percent_broken
        assert not other.truncate_val

    def test_get_complement_empty(self):
        this = DummyAbstractReader(1, 30, 125)  # Uses all runs
        other = this.get_complement(0.8, False)
        assert not other.percent_fail_runs  # Complement is empty
        assert 0.8 == other.percent_broken
        assert not other.truncate_val

    @pytest.mark.parametrize(
        ["runs_this", "runs_other", "success"],
        [
            (None, None, False),
            (None, [], True),
            (None, [1], False),
            ([1, 2], [1, 2, 3], False),
            ([1, 2], [3, 4], True),
            (0.8, [90], True),
            (0.8, [1], False),
        ],
    )
    def test_is_mutually_exclusive(self, runs_this, runs_other, success):
        this = DummyAbstractReader(1, percent_fail_runs=runs_this)
        other = DummyAbstractReader(1, percent_fail_runs=runs_other)

        assert this.is_mutually_exclusive(other) == success
        assert other.is_mutually_exclusive(this) == success

    @pytest.mark.parametrize(
        ["mode", "expected_this", "expected_other"],
        [("override", 30, 30), ("min", 15, 15), ("none", 30, 15)],
    )
    def test_consolidate_window_size(self, mode, expected_this, expected_other):
        this = DummyAbstractReader(1, window_size=30)
        other = this.get_compatible(2, consolidate_window_size=mode)

        assert this.window_size == expected_this
        assert other.window_size == expected_other

    @pytest.mark.parametrize(
        ["split", "alias", "truncate_val", "exp_truncated"],
        [
            ("dev", None, False, True),
            ("dev", "dev", False, True),
            ("val", None, False, False),
            ("val", None, True, True),
            ("val", "dev", False, True),
            ("val", "test", False, False),
            ("test", "dev", False, True),
            ("test", None, False, False),
        ],
    )
    @mock.patch("rul_datasets.reader.truncating.truncate_runs", return_value=([], []))
    def test_alias(self, mock_truncate_runs, split, alias, truncate_val, exp_truncated):
        this = DummyAbstractReader(1, truncate_val=truncate_val)
        this.load_complete_split = mock.Mock(wraps=this.load_complete_split)

        this.load_split(split, alias)

        this.load_complete_split.assert_called_with(split, alias or split)
        if exp_truncated:
            mock_truncate_runs.assert_called()
        else:
            mock_truncate_runs.assert_not_called()
