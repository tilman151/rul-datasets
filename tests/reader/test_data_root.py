import importlib
import os.path

import pytest

from rul_datasets.reader import data_root


def test_default_data_root():
    data_root = _reimport_package()

    exp_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )
    assert data_root.get_data_root() == exp_data_root


@pytest.mark.parametrize(
    "root", [os.getcwd(), pytest.param("/bogus", marks=pytest.mark.xfail)]
)
def test_env_var_data_root(monkeypatch, root):
    monkeypatch.setenv("RUL_DATASETS_DATA_ROOT", root)
    data_root = _reimport_package()

    assert data_root.get_data_root() == root


@pytest.mark.parametrize(
    "root", [os.getcwd(), pytest.param("/bogus", marks=pytest.mark.xfail)]
)
def test_manual_data_root(root):
    data_root = _reimport_package()

    data_root.set_data_root(root)

    assert data_root.get_data_root() == root


def _reimport_package():
    return importlib.reload(data_root)
