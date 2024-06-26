#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script trains a Transformer-like model based on the configuration provided
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from threading import Thread

from modules import Model
from src.utils.configuration import Configuration
import src.utils.helper_functions as helpers


def run_training(arguments: argparse.Namespace) -> None:
    """
    Performs the training of the Transformer-like model with the settings specified in the configuration file
    Arguments:
        arguments (argparse.Namespace): Custom arguments for the training, specifically the config path + name
    """
    # Load the user configurations
    cfg = Configuration(os.path.join(arguments.cfg_path, arguments.cfg_name))

    # Print some information to console
    print("Model name:", cfg.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
    if cfg.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()

    # Initialize and set up the model
    model = Model(
        n_heads=cfg.model.n_heads,
        linear_layer_size=cfg.model.linear_layer_size,
        d_model=cfg.model.d_model,
        d_one_hot=cfg.model.d_one_hot,
        dropout=0.1
    ).to(device=device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Count number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable model parameters:", pytorch_total_params)

    # Set up an optimizer and the criterion (loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    # criterion = nn.NLLLoss(reduction="mean")
    criterion = nn.CrossEntropyLoss()

    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty

    # Set up the dataloader
    dataset, dataloader = helpers.build_dataloader(cfg=cfg, batch_size=1)

    a = time.time()
    np.random.seed(1)

    # Start the training and iterate over all epochs
    for epoch in range(cfg.training.epochs):

        epoch_start_time = time.time()

        # List to store the errors for each sequence
        sequence_errors = []

        # Iterate over the training batches
        for batch_idx, (net_input, net_label) in enumerate(dataloader):
            # Move data to the desired device and convert from
            # [batch_size, time, dim] to [time, batch_size, dim]
            net_input = net_input.to(device=device).transpose(0, 1)
            net_label = net_label.to(device=device).transpose(0, 1)

            # Reset optimizer to clear the previous batch
            optimizer.zero_grad()

            trg_mask = np.triu(np.ones((1, net_label.shape[0], net_label.shape[0])), k=1).astype('uint8')
            trg_mask = torch.from_numpy(trg_mask) == 0

            # Calculate the error
            y_hat = model(net_input, mask=trg_mask.to(device=device))
            loss = criterion(y_hat.squeeze(1), net_label.squeeze(1).argmax(dim=1))

            # Compute the gradients
            loss.backward()

            # Perform weight update
            optimizer.step()

            sequence_errors.append(loss.item())

        epoch_errors_train.append(np.mean(sequence_errors))

        # Save the model to file (if desired)
        if cfg.training.save_model and epoch % cfg.training.save_every_nth_epoch == 0:
            print(f'\nSaving model @ epoch {epoch + 1}')
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=os.path.abspath(""),
                cfg=cfg,
                epoch=epoch,
                epoch_errors_train=epoch_errors_train,
                model=model))
            thread.start()

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]

        # Print progress to the console
        print(
            f"Epoch {str(epoch + 1).zfill(int(np.log10(cfg.training.epochs)) + 1)}"
            f"/{str(cfg.training.epochs)} took "
            f"{str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} "
            f"seconds. \t\tAverage epoch training error: "
            f"{train_sign}"
            f"{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}"
        )

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default=".")
    parser.add_argument("--cfg_name", default="config.json")
    args = parser.parse_args()
    run_training(arguments=args)
    print("Done.")
