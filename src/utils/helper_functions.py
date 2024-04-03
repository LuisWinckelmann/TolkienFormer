# -*- coding: utf-8 -*-
"""
Several helper functions that are used multiple times throughout the project
"""

import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset

from src.utils.configuration import Configuration
from src.utils.datasets import TolkienDataset
from src.models.lstm.modules import Model as LSTM
from src.models.transformer.modules import Model as Transformer


def determine_device() -> torch.device:
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    Returns:
        The device where tensor calculations shall be made on
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print(f"\t Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB")
        print(f"\t Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB\n")
    return device


def build_dataloader(cfg: Configuration, batch_size: int) -> tuple[Dataset | TolkienDataset, DataLoader]:
    """
    This function creates a dataset and return the appropriate dataloader to
    iterate over this dataset
    Arguments:
        cfg: The general configurations of the model
        batch_size: The number of samples per batch
    Returns:
        DataLoader: A PyTorch dataloader object
    """
    # Set up a dataset and dataloader
    dataset = TolkienDataset(dataset_name=cfg.dataset.name)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=2 # Was disabled in newer pytorch variants,
        # see https://github.com/pytorch/pytorch/issues/68576
    )
    return dataset, dataloader


def save_model_to_file(model_src_path: str, cfg: Configuration, epoch: int, epoch_errors_train: list,
                       model: LSTM | Transformer) -> None:
    """
    This function writes the model weights along with the network configuration
    and current performance to file.
    Arguments:
        model_src_path (str): The source path where the model will be saved to
        cfg (Configuration): The configurations of the model
        epoch (int): The current epoch
        epoch_errors_train (list): The training epoch errors
        model(LSTM | Transformer): The actual model
    """
    model_save_path = os.path.join(model_src_path, "checkpoints", cfg.model.name)
    os.makedirs(model_save_path, exist_ok=True)

    # Save model weights to file
    torch.save(model.state_dict(), os.path.join(model_save_path, cfg.model.name + "_epoch_" + str(epoch) + ".pt"))

    # Copy the configurations and add a results entry
    cfg["results"] = {
        "current_epoch": epoch + 1,
        "current_training_error": epoch_errors_train[-1],
        "lowest_train_error": min(epoch_errors_train),
    }
    # Save the configuration and current performance to file
    cfg.save(model_save_path)


def one_hot_to_char(one_hot_vector: np.array, alphabet: np.array) -> str:
    """
    Converts a one hot vector into a character given an alphabet.
    Arguments:
        one_hot_vector (np.array): The one-hot vector
        alphabet (np.array): The alphabet
    Returns:
         str: The character which the one_hot_vector represents
    """
    idx = np.where(one_hot_vector == 1)[0][0]
    char = alphabet[idx]
    return char


def char_to_one_hot(char: str, alphabet: np.array) -> np.array:
    """
    Converts a character into a one-hot vector given an alphabet.
    Arguments:
        char (str): The character
        alphabet (np.array): The alphabet
    Returns:
        np.array: The one-hot vector for the given character
    """
    one_hot_vector = np.zeros(len(alphabet))
    try:
        idx = np.where(alphabet == char)[0][0]
        one_hot_vector[idx] = 1
    except BaseException as error:
        print(error)
        raise Exception("Character not in the Alphabet")
        # FIXME: better handling
    return one_hot_vector


def softmax_to_one_hot(soft: np.array) -> np.array:
    """
    Converts a softmax output into a discretized one-hot vector, where the
    largest value of the softmax is encoded as one.
    Arguments:
        soft (np.array): The softmax vector
    Returns:
        np.array: The discretized one-hot vector
    """
    one_hot_vector = np.zeros(len(soft), dtype=np.int8)
    one_hot_vector[soft.data.topk(1)[1][0].detach().cpu().numpy()] = 1
    return one_hot_vector
