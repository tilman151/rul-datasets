import os

from rul_datasets import loader


def process_femto():
    for fd in range(1, 4):
        femto_root = os.path.join(loader.DATA_ROOT, "FEMTOBearingDataSet")
        preparator = loader.FemtoPreparator(fd, femto_root)
        for split in ["dev", "test"]:
            preparator.prepare_split(split)


if __name__ == "__main__":
    process_femto()
