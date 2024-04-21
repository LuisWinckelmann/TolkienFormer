#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script enables testing of previously trained LSTM models. Depending on the config, a random index of the dataset is picked and for
config.teacher_forcing_steps fed to the LSTM. Afterwards, the model predicts the next characters
cfg.testing.closed_loop_steps-times before the prediction as well as the ground truth get printed to the console.
"""

import argparse
import os
import torch
import numpy as np
from torch import autograd, nn

from modules import Model
from src.utils.configuration import Configuration
import src.utils.helper_functions as helpers


def run_testing(arguments: argparse.Namespace) -> None:
    """
    Does multiple forward passes on the selected model with the specified teacher forcing from the configuration. The
    real and the predicted sentences then get printed to the console. Arguments: arguments (argparse.Namespace):
    Specified information about the amount of tests, the chosen model and the corresponding config.
    """
    # Unpack args:
    num_predicted_sentences = arguments.num_sentences
    model_epoch = arguments.model_epoch

    # Load the user configurations
    cfg = Configuration(os.path.join(arguments.cfg_path, arguments.cfg_name))

    # Print some information to console
    print("Model name:", cfg.model.name + "_epoch_" + str(model_epoch))

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
    if cfg.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU
    device = helpers.determine_device()

    # Initialize and set up the model
    model = Model(d_one_hot=cfg.model.d_one_hot, d_lstm=cfg.model.d_lstm, num_lstm_layers=cfg.model.num_lstm_layers
                  ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model.load_state_dict(torch.load(os.path.join(os.path.abspath(""),
                                                  "checkpoints",
                                                  cfg.model.name,
                                                  cfg.model.name + "_epoch_" + str(model_epoch) + ".pt"),
                                     map_location=device))
    model.eval()

    # Set up the dataloader for the testing
    dataset, dataloader = helpers.build_dataloader(
        cfg=cfg, batch_size=1
    )

    # Extract num of teacher forcing step and number of prediction steps from the config
    tf_steps = cfg.testing.teacher_forcing_steps
    cl_steps = cfg.testing.closed_loop_steps

    # Iterate over the batches
    for batch_idx, (net_input, _) in enumerate(dataloader):

        # Move data to the desired device and swap batch_size and dim:
        # [batch_size, time, dim] --> [time, batch_size, dim]
        net_input = net_input.to(device=device).transpose(0, 1)

        # Start with the first teacher forcing characters and let the model continue
        x = net_input[:tf_steps]
        h = autograd.Variable(torch.zeros(model.num_layers, 1, model.hidden_size, device=device))  # hidden state
        c = autograd.Variable(torch.zeros(model.num_layers, 1, model.hidden_size, device=device))  # internal state
        y_hat_last = None
        for i in x:
            i = torch.unsqueeze(i, 0)
            y_hat, (h, c) = model(x=i, state=(h, c))
            y_hat = nn.functional.softmax(y_hat, dim=-1)

            y_hat = torch.unsqueeze(y_hat, 0)
            y_hat_last = torch.tensor(np.array([[helpers.softmax_to_one_hot(soft=y_hat[-1, 0])]])).to(device=device)

        # Append the model output to the input and continue
        x = torch.cat((x, y_hat_last), dim=0)

        for t in range(cl_steps):
            # Generate a prediction and apply the softmax
            y_hat, (h, c) = model(x=y_hat_last.float(), state=(h, c))
            y_hat = nn.functional.softmax(y_hat, dim=-1)
            y_hat = torch.unsqueeze(y_hat, 0)

            # Convert the last softmax output to a one-hot vector
            y_hat_last = torch.tensor(np.array([[helpers.softmax_to_one_hot(soft=y_hat[-1, 0])]])).to(device=device)

            # Append the model output to the input and continue
            x = torch.cat((x, y_hat_last), dim=0)

        net_input = net_input.detach().cpu().numpy()
        tolkien_text = []

        for t in range(len(net_input[:tf_steps + cl_steps])):
            y_t = helpers.one_hot_to_char(one_hot_vector=net_input[t, 0], alphabet=dataset.alphabet)
            tolkien_text.append(y_t[0])

        model_text = []
        for t in range(len(x)):
            x_t = helpers.one_hot_to_char(
                one_hot_vector=helpers.softmax_to_one_hot(soft=x[t, 0]),
                alphabet=dataset.alphabet
            )
            model_text.append(x_t[0])

        tolkien_text = "".join(tolkien_text)
        model_text = "".join(model_text)

        print(f"Initialization: {tolkien_text[:tf_steps]}...")
        print(f"\nTolkien: \n --- {tolkien_text}")
        print(f"\nLSTM model: \n --- {model_text}")

        if batch_idx > num_predicted_sentences:
            exit()
        print("\n\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sentences", default=5, type=int)
    parser.add_argument("--model_epoch", default=125, type=int)
    parser.add_argument("--cfg_path", default=".")
    parser.add_argument("--cfg_name", default="config.json")
    args = parser.parse_args()
    run_testing(arguments=args)
    print("Done.")
