import warnings

import pytest

from rul_datasets import ssl, CmapssReader, RulDataModule


def test_check_compatibility():
    labeled = CmapssReader(fd=1, percent_fail_runs=0.8)
    unlabeled = labeled.get_complement(percent_broken=0.6)
    ssl.SemiSupervisedDataModule(
        RulDataModule(labeled, 32), RulDataModule(unlabeled, 32)
    )


def test_check_compatibility_different_fds():
    labeled = CmapssReader(fd=1, percent_fail_runs=0.8)
    unlabeled = CmapssReader(fd=3, percent_fail_runs=0.8, percent_broken=0.6)
    with pytest.raises(ValueError):
        ssl.SemiSupervisedDataModule(
            RulDataModule(labeled, 32), RulDataModule(unlabeled, 32)
        )


def test_check_compatibility_warn_overlap():
    labeled = CmapssReader(fd=1, percent_fail_runs=0.8)
    unlabeled = CmapssReader(fd=1, percent_fail_runs=0.8, percent_broken=0.6)
    with warnings.catch_warnings(record=True) as caught_warnings:
        ssl.SemiSupervisedDataModule(
            RulDataModule(labeled, 32), RulDataModule(unlabeled, 32)
        )

    assert len(caught_warnings) == 1


def test_check_compatibility_warn_unlabeled_untruncated():
    labeled = CmapssReader(fd=1, percent_fail_runs=0.8)
    unlabeled = labeled.get_complement()
    with warnings.catch_warnings(record=True) as caught_warnings:
        ssl.SemiSupervisedDataModule(
            RulDataModule(labeled, 32), RulDataModule(unlabeled, 32)
        )

    assert len(caught_warnings) == 1
