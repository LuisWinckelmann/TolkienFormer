# -*- coding: utf-8 -*-
"""
Implementation of an LSTM to predict the most probably next character based on the previous given/produced input
"""

import torch as th
import torch.nn as nn
import torch.autograd as atgr


class Model(nn.Module):
    """
    The actual model consisting of some feed forward and recurrent layers.
    """

    def __init__(self, d_one_hot, d_lstm, num_lstm_layers, dropout=0.1, bias=True):
        """
        Constructor method of the Model module.
        :param d_one_hot: The size of the input and output vector
        :param d_lstm: The hidden size of the lstm layers
        :param num_lstm_layers: The number of sequential lstm layers
        :param dropout: Probability of dropping out certain neurons
        :param bias: Whether to use bias neurons
        """
        self.num_layers = num_lstm_layers
        self.hidden_size = d_lstm
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_one_hot, hidden_size=d_lstm, num_layers=num_lstm_layers, bias=bias,
                            dropout=dropout, batch_first=True)
        self.linear = nn.Linear(d_lstm, d_one_hot)

    def forward(self, x, state=None):
        """
        The forward pass function of the module.
        :param x: The input to the module
        :param state: The previous model state, None if no previous state is passed to the model
        :return: The module's output
        """
        if state is None:
            device = x.device
            h_0 = atgr.Variable(th.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))  # hidden state
            c_0 = atgr.Variable(th.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))  # internal state
        else:
            (h_0, c_0) = state

        output, (h, c) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        output_flat = output.view(-1, self.hidden_size)
        out = self.linear(output_flat)
        return out, (h, c)
