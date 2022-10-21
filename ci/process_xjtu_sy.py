import os

from rul_datasets import loader


def process_xjtu_sy():
    for fd in range(1, 4):
        print(f"Prepare FD{fd}")
        preparator = loader.XjtuSyLoader(fd)
        preparator.prepare_data()


if __name__ == "__main__":
    process_xjtu_sy()
