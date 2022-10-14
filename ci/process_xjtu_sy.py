import os

from rul_datasets import loader


def process_xjtu_sy():
    for fd in range(1, 4):
        print(f"Prepare FD{fd}")
        xjtu_sy_root = os.path.join(loader.DATA_ROOT, "XJTU-SY")
        preparator = loader.XjtuSyPreparator(fd, xjtu_sy_root)
        preparator.prepare_split("dev")


if __name__ == "__main__":
    process_xjtu_sy()
