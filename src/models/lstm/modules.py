# -*- coding: utf-8 -*-
"""
Implementation of an LSTM to predict the most probable next character based on the previous given/produced text
"""

from typing import Optional

import torch
from torch import nn, autograd


class Model(nn.Module):
    """
    The actual model consisting of some feed forward and recurrent layers.
    Attributes:
        num_layers (int): number of LSTM layers
        hidden_size (int): hidden size of the LSTM
        lstm nn.LSTM: torch LSTM object
        linear nn.Linear: torch Linear Layer for the model's output
    """

    def __init__(self, d_one_hot: int, d_lstm: int, num_lstm_layers: int, dropout: float = 0.1,
                 bias: bool = True) -> None:
        """
        Constructor method of the Model module.
        Arguments:
            d_one_hot (int): The size of the input and output vector
            d_lstm (int): The hidden size of the lstm layers
            num_lstm_layers (int): The number of sequential lstm layers
            dropout (float): Probability of dropping out certain neurons
            bias (bool): Whether to use bias neurons
        """
        self.num_layers = num_lstm_layers
        self.hidden_size = d_lstm
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_one_hot, hidden_size=d_lstm, num_layers=num_lstm_layers, bias=bias,
                            dropout=dropout, batch_first=True)
        self.linear = nn.Linear(d_lstm, d_one_hot)

    def forward(self, x: torch.tensor, state: Optional[tuple[autograd.Variable, autograd.Variable]] = None) \
            -> tuple[torch.tensor, tuple[[autograd.Variable, autograd.Variable]]]:
        """
        The forward pass function of the module.
        Arguments:
            x (torch.tensor): The input to the module
            state: The previous model state, None if no previous state is passed to the model
        Returns:
             torch.tensor: The module's output
             autograd.Variable, autograd.Variable: the model's current state
        """
        if state is None:
            device = x.device
            h_0 = autograd.Variable(
                torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))  # hidden state
            c_0 = autograd.Variable(
                torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))  # internal state
        else:
            (h_0, c_0) = state

        output, (h, c) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        output_flat = output.view(-1, self.hidden_size)
        out = self.linear(output_flat)
        return out, (h, c)
