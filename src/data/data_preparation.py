#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script converts all *.txt files in the data directory into chunks of one hot encoded .npy files that then can
be used to train & test the models. Additionally an alphabet will be saved that contains all unique characters present
in the dataset
"""
import os
from pathlib import Path
import numpy as np

from src.utils.helper_functions import char_to_one_hot

DATA_PATH = "."


def txt_to_npy(path, dataset_name):
    """
    Creates an alphabet and sequences of all lines from a given .txt file and
    writes the one-hot vector sequences to separate files.
    :param path: The path to the .txt file containing the raw text
    :param dataset_name: The name of the dataset we are parsing
    """
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


def prepare_data(data_path):
    # convert every dataset (*.txt file) in /data into one hot encoded samples
    datasets = [file for file in os.listdir(data_path) if file.endswith(".txt")]
    print("Starting dataset preparation")
    for dataset in datasets:
        dataset_name = Path(dataset).stem
        print(f"Parsing dataset: {dataset_name}")
        txt_to_npy(dataset, dataset_name)
    print("Done.")


if __name__ == "__main__":
    prepare_data(DATA_PATH)
