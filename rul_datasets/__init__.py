__version__ = "0.0.0"

import warnings

from .adaption import (
    DomainAdaptionDataModule,
    PretrainingAdaptionDataModule,
    LatentAlignDataModule,
)
from .baseline import BaselineDataModule, PretrainingBaselineDataModule
from .core import RulDataModule
from .reader import CmapssReader, FemtoReader, XjtuSyReader, NCmapssReader
from .reader.data_root import get_data_root, set_data_root
from .ssl import SemiSupervisedDataModule

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
