#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script converts all *.txt files in the data directory into chunks of one hot encoded .npy files that then can
be used to train & test the models. Additionally an alphabet will be saved that contains all unique characters present
in the dataset
"""
import argparse
import os
from pathlib import Path
import numpy as np

from src.utils.helper_functions import char_to_one_hot


def txt_to_npy(path: str) -> None:
    """
    Creates an alphabet and sequences of all lines from a given .txt file and
    writes the one-hot vector sequences to separate files.
    Args:
        path (str): The path to the .txt file containing the raw text
    """
    dataset_name = Path(path).stem
    # Create folder if it doesn't exist already
    os.makedirs(dataset_name, exist_ok=True)

    with open(path, "r", encoding='utf-8') as file:
        # 1. Determine the alphabet of the text
        chars = []
        for line in file:
            if len(line) <= 1:
                continue
            for char in line:
                # Only consider lower-case characters
                chars.append(char.lower())

        alphabet = np.unique(np.array(chars))
        np.save(os.path.join(dataset_name, "alphabet.npy"), alphabet)

        # 2. Create one-hot vector sequences
        file.seek(0, 0)
        data_idx = 0

        for line in file:
            if len(line) <= 1:
                continue
            sample = np.zeros((len(line), len(alphabet)))
            for i, char in enumerate(line):
                sample[i] = char_to_one_hot(char.lower(), alphabet)
            data_idx += 1
            np.save(os.path.join(dataset_name, "sample_{num:04d}.npy".format(num=data_idx)), sample)


def prepare_data(data_path: str) -> None:
    # convert every dataset (*.txt file) in /data into one hot encoded samples
    datasets = [file for file in os.listdir(data_path) if file.endswith(".txt")]
    print("Starting dataset preparation")
    for dataset in datasets:
        txt_to_npy(dataset)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".")
    args = parser.parse_args()
    prepare_data(args.path)
