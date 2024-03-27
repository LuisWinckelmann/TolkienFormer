#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: Description
"""

import os
import numpy as np
import torch as th
import torch.nn as nn

import src.utils.helper_functions as helpers
from modules import Model
from src.utils.configuration import Configuration

# TODO: Turn this into flags
# Number of Predicted sentences
NUM_PREDICTED_SENTENCES = 5
# Epoch of the model to load
LOADING_MODEL_EPOCH = 1


def run_testing():
    # Load the user configurations
    cfg = Configuration("config.json")

    # Print some information to console
    print("Model name:", cfg.model.name + "_epoch_" + str(LOADING_MODEL_EPOCH))

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
        d_one_hot=cfg.model.d_one_hot
    ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    model.load_state_dict(
        th.load(os.path.join(os.path.abspath(""),
                             "checkpoints",
                             cfg.model.name,
                             cfg.model.name + "_epoch_" + str(LOADING_MODEL_EPOCH) + ".pt"),
                map_location=device))
    model.eval()

    # Set up the dataloader for the testing
    dataset, dataloader = helpers.build_dataloader(cfg=cfg, batch_size=1)

    tf_steps = cfg.testing.teacher_forcing_steps
    cl_steps = cfg.testing.closed_loop_steps

    # Iterate over the training batches
    for batch_idx, (net_input, _) in enumerate(dataloader):
        # Move data to the desired device and swap batch_size and time
        # [batch_size, time, dim] --> [time, batch_size, dim]
        net_input = net_input.to(device=device).transpose(0, 1)

        # Start with the first teacher forcing characters and let the model continue
        x = net_input[:tf_steps]

        trg_mask = np.triu(np.ones((1, x.shape[0], x.shape[0])), k=1).astype('uint8')
        mask = (th.from_numpy(trg_mask) == 0).cuda()

        # Generate predictions
        y_hat = model(x=x, mask=mask)
        y_hat = nn.functional.softmax(y_hat, dim=-1)

        y_hat_last = th.tensor(np.array([[helpers.softmax_to_one_hot(soft=y_hat[-1, 0])]])).to(device=device)
        x = th.cat((x, y_hat_last.float()), dim=0)

        for t in range(cl_steps):
            trg_mask = np.triu(np.ones((1, x.shape[0], x.shape[0])), k=1).astype('uint8')
            mask = (th.from_numpy(trg_mask) == 0).cuda()

            y_hat = model(x=x, mask=mask)
            y_hat = nn.functional.softmax(y_hat, dim=-1)
            y_hat_last = th.tensor(np.array([[helpers.softmax_to_one_hot(soft=y_hat[-1, 0])]])).to(device=device)

            # Append the model output to the input and continue
            x = th.cat((x, y_hat_last.float()), dim=0)

        net_input = net_input.detach().cpu().numpy()
        tolkien_text = []

        for t in range(len(net_input[:tf_steps + cl_steps])):
            y_t = helpers.one_hot_to_char(one_hot_vector=net_input[t, 0], alphabet=dataset.alphabet)
            tolkien_text.append(y_t[0])

        model_text = []
        for t in range(len(x)):
            x_t = helpers.one_hot_to_char(one_hot_vector=helpers.softmax_to_one_hot(soft=x[t, 0]),
                                          alphabet=dataset.alphabet)
            model_text.append(x_t[0])

        tolkien_text = "".join(tolkien_text)
        model_text = "".join(model_text)

        print(f"Initialization: {tolkien_text[:tf_steps]}...")
        print(f"\nTolkien:\n --- {tolkien_text}")
        print(f"\nTransformer model:\n --- {model_text}")
        print()

        if batch_idx > NUM_PREDICTED_SENTENCES:
            exit()
        print("\n\n")


if __name__ == "__main__":
    th.set_num_threads(1)
    run_testing()

    print("Done.")
