from rul_datasets import loader


def process_femto():
    for fd in range(1, 4):
        preparator = loader.FemtoPreparator(fd, loader.FemtoLoader.DATA_ROOT)
        for split in ["train", "test"]:
            preparator.prepare_split(split)


if __name__ == "__main__":
    process_femto()
