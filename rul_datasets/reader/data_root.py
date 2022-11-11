"""A module for working with the data root directory."""

import os


def _init_data_root() -> str:
    if "RUL_DATASETS_DATA_ROOT" in os.environ:
        root = os.environ["RUL_DATASETS_DATA_ROOT"]
        if not os.path.exists(root):
            raise ValueError(
                f"Data root '{root}' set by 'RUL_DATASETS_DATA_ROOT' "
                "env var does not exist."
            )
    else:
        root = os.path.expanduser(os.path.join("~", ".rul-datasets"))
        os.makedirs(root, exist_ok=True)

    return root


_DATA_ROOT = _init_data_root()


def get_data_root() -> str:
    """
    Return the path to the data root directory.

    The default data root is located at `~/.rul-datasets`. You can customize the
    location by setting the environment variable `RUL_DATASETS_DATA_ROOT` or by
    calling [set_data_root][rul_datasets.reader.data_root.set_data_root].

    A manually set data root must be an already existing directory.

    Returns:
        The data root path.
    """
    return _DATA_ROOT


def set_data_root(data_root: str) -> None:
    """
    Set the data root to the specified location.

    A manually set data root must be an already existing directory.

    Args:
        data_root: The new data root location.
    """
    global _DATA_ROOT
    if not os.path.exists(data_root):
        raise ValueError(f"Data root '{data_root}' does not exist.")
    _DATA_ROOT = data_root
