# -*- coding: utf-8 -*-
"""
This file contains the organization of the custom dataset such that it can be
read efficiently in combination with the DataLoader from PyTorch to prevent that
data reading and preparing becomes the bottleneck.

Originally inspired by
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import glob
import os

import numpy as np
from torch.utils.data import Dataset


class TolkienDataset(Dataset):
    """
    The custom Tolkien dataset class which can be used with PyTorch's
    DataLoader.

    Attributes:
        data_paths: list of paths to npy files that belong to this dataset
        alphabet: contains all unique characters that appear in the dataset
    """

    def __init__(self, dataset_name: str):
        """
        Constructor class setting up the data loader
        Args:
            dataset_name (str): The name of the dataset (e.g. "chapter1")
        """

        data_root_path = os.path.join(
            os.path.abspath("../.."), "data", dataset_name
        )
        self.data_paths = np.sort(
            glob.glob(os.path.join(data_root_path, "sample*.npy"))
        )
        self.alphabet: np.ndarray = np.load(os.path.join(data_root_path, "alphabet.npy"))

    def __len__(self) -> int:
        """
        Denotes the total number of samples that exist
        Returns:
            int: The number of samples
        """
        return len(self.data_paths)

    def __getitem__(self, index: int) -> tuple[np.array, np.array]:
        """
        Generates a sample batch in the form [batch_size, time, dim],
        where x and y are the sizes of the data and dim is the number of
        features.
        Args:
            index (int): The index of the sample in the path array
        Returns:
             np.array, np.array: One batch of data in the form of input-label pairs
        """
        # Load a sample from file and divide it in input and label. The label is the input shifted one time-step to
        # train one step ahead prediction.
        sample = np.float32(np.load(self.data_paths[index]))
        net_input = np.copy(sample[:-1])
        net_label = np.copy(sample[1:])

        return net_input, net_label
