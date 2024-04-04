#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script trains an LSTM based on the configuration provided by config.json
"""


import os
import time
import numpy as np
import torch
from torch import nn
from threading import Thread

from modules import Model
from src.utils.configuration import Configuration
import src.utils.helper_functions as helpers


def run_training() -> None:
    """
    Performs the training of the LSTM with the settings specified in the configuration.json file
    """
    # Load the user configurations
    cfg = Configuration("config.json")

    # Print some information to console
    print("Model name:", cfg.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
    if cfg.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("USING CPU")
    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()

    # Initialize and set up the model
    model = Model(
        d_one_hot=cfg.model.d_one_hot,
        d_lstm=cfg.model.d_lstm,
        num_lstm_layers=cfg.model.num_lstm_layers
    ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    # Set up an optimizer and the criterion (loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Set up a list to save and store the epoch errors
    epoch_errors = []
    best_error = np.infty

    # Set up the training dataloader
    dataset, dataloader = helpers.build_dataloader(cfg=cfg, batch_size=1)

    a = time.time()

    # Start the training and iterate over all epochs
    for epoch in range(cfg.training.epochs):

        epoch_start_time = time.time()

        # List to store the errors for each sequence
        sequence_errors = []

        # Iterate over the training batches
        for batch_idx, (net_input, net_label) in enumerate(dataloader):
            net_input = net_input.to(device=device)
            net_label = net_label.to(device=device)

            # Reset optimizer to clear the previous batch
            optimizer.zero_grad()

            # Generate a model prediction and compute the cross-entropy loss
            y_hat, (h, c) = model(net_input)

            target = net_label.squeeze()  # type(torch.cuda.LongTensor)
            target = torch.argmax(target, dim=1)
            loss = criterion(y_hat, target)

            # Compute gradients
            loss.backward()

            # Perform weight update
            optimizer.step()
            sequence_errors.append(loss.item())

        epoch_errors.append(np.mean(sequence_errors))

        # Save the model to file (if desired)
        if cfg.training.save_model and epoch % cfg.training.save_every_nth_epoch == 0:  # and np.mean(
            # sequence_errors) < best_train:
            print('\nSaving model')
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=os.path.abspath(""),
                cfg=cfg,
                epoch=epoch,
                epoch_errors_train=epoch_errors,
                model=model))
            thread.start()

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors[-1] < best_error:
            train_sign = "(+)"
            best_error = epoch_errors[-1]

        # Print progress to the console
        print(
            f"Epoch {str(epoch + 1).zfill(int(np.log10(cfg.training.epochs)) + 1)}"
            f"/{str(cfg.training.epochs)} took "
            f"{str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} "
            f"seconds. \t\tAverage epoch training error: "
            f"{train_sign}"
            f"{str(np.round(epoch_errors[-1], 10)).ljust(12, ' ')}"
        )

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_training()
    print("Done.")
