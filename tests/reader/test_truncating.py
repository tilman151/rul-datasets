import numpy as np
import numpy.testing as npt
import pytest

from rul_datasets.reader import truncating


@pytest.fixture
def rul_data():
    targets = [np.arange(np.random.randint(10, 50), 0, -1) for _ in range(10)]
    features = [np.random.randn(t.shape[0], 10, 2) for t in targets]

    return features, targets


class TestIncludedRuns:
    @pytest.mark.parametrize("included_percentage", [0.1, 0.5, 1.0])
    def test_included_percentage(self, rul_data, included_percentage):
        features, targets = rul_data
        trunc_features, trunc_targets = truncating.truncate_runs(
            features, targets, included_runs=included_percentage
        )

        expected_trunc_length = int(included_percentage * len(features))
        assert len(trunc_features) == len(trunc_targets)
        assert len(trunc_features) == expected_trunc_length
        assert trunc_features == features[:expected_trunc_length]
        assert trunc_targets == targets[:expected_trunc_length]

    @pytest.mark.parametrize("included_idx", [[0], [1, 2], [5, 1, 8]])
    def test_included_idx(self, rul_data, included_idx):
        features, targets = rul_data
        trunc_features, trunc_targets = truncating.truncate_runs(
            features, targets, included_runs=included_idx
        )

        expected_trunc_length = len(included_idx)
        assert len(trunc_features) == len(trunc_targets)
        assert len(trunc_features) == expected_trunc_length
        assert trunc_features == [features[i] for i in included_idx]
        assert trunc_targets == [targets[i] for i in included_idx]


@pytest.mark.parametrize("percent_broken", [0.1, 0.5, 1.0])
def test_percent_broken(rul_data, percent_broken):
    features, targets = rul_data
    trunc_features, trunc_targets = truncating.truncate_runs(
        features, targets, percent_broken=percent_broken
    )
    _check_truncation_broken(features, trunc_features, percent_broken)
    _check_truncation_broken(targets, trunc_targets, percent_broken)


def _check_truncation_broken(runs, trunc_runs, percent_broken):
    for run, trunc_run in zip(runs, trunc_runs):
        expected_run_length = int(percent_broken * len(run))
        assert len(trunc_run) == expected_run_length
        npt.assert_equal(trunc_run, run[:expected_run_length])
