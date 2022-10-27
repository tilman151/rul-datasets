import os

from tqdm import tqdm

from rul_datasets import reader


def process_femto():
    for fd in range(1, 4):
        preparator = reader.FemtoReader(fd)
        preparator.prepare_data()


if __name__ == "__main__":
    process_femto()
