# -*- coding: utf-8 -*-
"""
TODO: Description
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    A specific feed forward module that consists of a ReLU layer followed by a dropout and a linear layer.
    """

    def __init__(self, d_model, linear_layer_size, dropout=0.1, bias=True):
        """
        Constructor method of the feed forward module.
        :param linear_layer_size: The internal size of the feed forward module
        :param dropout: Probability of dropping out certain neurons
        :param bias: Whether to use bias neurons
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, linear_layer_size, bias=bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(linear_layer_size, d_model, bias=bias)

    def forward(self, x):
        """
        The forward pass function of the module.
        :param x: The input to the module
        :return: The module's output
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    The core component of the transformer realizing the attention mechanism.
    """

    def __init__(self, n_heads, d_model, dropout=0.1, bias=True):
        """
        Constructor method of the attention module.
        :param n_heads: The number of attention heads
        :param d_model: The size of the K, V, Q and output vectors
        :param dropout: Probability of dropping out certain neurons
        :param bias: Whether to use bias neurons
        """
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.d_per_h = d_model // n_heads
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Forward pass of the multi head attention module.
        # TODO document
        :return: The attention weighted output as linear combination of v
        """
        seq_len, batch_size, _ = x.shape
        k = self.k(x).view(batch_size, seq_len, self.h, self.d_per_h).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.h, self.d_per_h).transpose(1, 2)
        q = self.q(x).view(batch_size, seq_len, self.h, self.d_per_h).transpose(1, 2)
        o, _ = self.attention(k, v, q, mask)
        c = o.transpose(1, 2).contiguous().view(seq_len, batch_size, self.d)
        out = self.out(c)
        return out

    def attention(self, k, v, q, mask=None):
        """
        The attention mechanism computing a weighted linear combination of v
        based on the similarity of the according k and v entries.
        :param k: Key vector
        :param v: Value vector
        :param q: Query vector
        :param mask: Mask to hide future entries
        :return: Weighted linear combination v_hat and the attention weights
        """
        s = th.matmul(q, k.transpose(-2, -1)) / self.d_per_h ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(0)
            s = s.masked_fill(mask == 0, -1e9)

        s = F.softmax(s, dim=-1)
        if self.dropout is not None:
            s = self.dropout(s)
        out = th.matmul(s, v)
        return out, s


class DecoderLayer(nn.Module):
    """
    A decoder layer part (of a Transformer) which predicts next observations
    based on the previous inputs.
    """
    def __init__(self, n_heads, d_model, linear_layer_size, dropout=0.1, bias=True):
        """
        Constructor method of the attention module.
        :param n_heads: The number of attention heads
        :param d_model: The size of the K, V, Q and output vectors
        :param linear_layer_size: The internal size of the feed forward module
        :param dropout: Probability of dropping out certain neurons
        :param bias: Whether to use bias neurons
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.att1 = MultiHeadAttention(n_heads, d_model, dropout, bias)
        self.att2 = MultiHeadAttention(n_heads, d_model, dropout, bias)

        self.feedforward = FeedForward(d_model, linear_layer_size, dropout, bias)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        The forward pass function of the module.
        :param x: The input to the module
        :param mask: Mask to hide future entries
        :return: The module's output
        """
        y = self.norm1(x)
        x = x + self.dropout1(self.att1.forward(y, mask))
        y = self.norm2(x)
        x = x + self.dropout2(self.att2.forward(y, mask))
        y = self.norm3(x)
        x = x + self.dropout3(self.feedforward.forward(y))
        y = self.norm4(x)
        return y


class Model(nn.Module):
    """
    The actual model consisting of a selected number of sequential decoder layers.
    """
    def __init__(self, n_heads, d_model, linear_layer_size, d_one_hot, dropout=0.1, bias=True):
        """
        Constructor method of the Model module.
        :param n_heads: The number of attention heads
        :param d_model: The size of the K, V, Q and output vectors
        :param linear_layer_size: The internal size of the feed forward module
        :param d_one_hot: The size of the input and output vector
        :param dropout: Probability of dropping out certain neurons
        :param bias: Whether to use bias neurons
        """
        # TODO: Cleanup
        super().__init__()
        self.enc = nn.Linear(d_one_hot, d_model, bias=bias)
        # self.enc = nn.Embedding(d_one_hot, d_model)
        # self.dec1 = DecoderLayer(n_heads, d_model, linear_layer_size, dropout, bias)
        self.dec2 = DecoderLayer(n_heads, d_model, linear_layer_size, dropout, bias)
        self.out = nn.Linear(d_model, d_one_hot, bias=bias)

    def forward(self, x, mask):
        """
        The forward pass function of the module.
        :param x: The input to the module
        :param mask: Mask to hide future entries
        :return: The module's output
        """
        # x = self.enc(th.argmax(x, dim=-1))
        x = self.enc(x)
        # x = self.dec1(x, mask)
        x = self.dec2.forward(x, mask)
        x = self.out(x)
        return x
