import os

from tqdm import tqdm

from rul_datasets import loader


def process_femto():
    for fd in range(1, 4):
        preparator = loader.FemtoLoader(fd)
        preparator.prepare_data()


if __name__ == "__main__":
    process_femto()
