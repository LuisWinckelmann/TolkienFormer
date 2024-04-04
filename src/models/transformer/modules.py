# -*- coding: utf-8 -*-
"""
Implementation of a Transformer-like architecture to predict the most probable next character based on previous
given/produced text. The implementation includes MultiHeadAttention, Encoding and Decoding.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class FeedForward(nn.Module):
    """
    A specific feed forward module that consists of a ReLU layer followed by a dropout and a linear layer.
    Attributes:
        net (nn.Sequential): The sequential layers of the network
    """

    def __init__(self, d_model: int, linear_layer_size: int, dropout: float = 0.1, bias: bool = True) -> None:
        """
        Constructor method of the feed forward module.
        Arguments:
            d_model (int): The size of input and output dimensions.
            linear_layer_size (int): The size of the hidden layer.
            dropout (float): The dropout rate.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, linear_layer_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_layer_size, d_model, bias=bias)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        The forward pass function of the module.
        Arguments:
            x (torch.tensor): The input tensor.
        Returns:
            torch.tensor: The output of the network
        """
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    The core component of the transformer realizing the attention mechanism.
    Attributes:
        d_model (int): The size of the key, value, query, and output vectors.
        n_heads (int): The number of attention heads.
        d_per_head (int): The dimension of each head, calculated as `d_model // n_heads`.
        key (nn.Linear): Linear layer to transform input into key vector.
        value (nn.Linear): Linear layer to transform input into value vector.
        query (nn.Linear): Linear layer to transform input into query vector.
        dropout (nn.Dropout): Dropout layer applied to the attention scores.
        out (nn.Linear): Output linear layer that transforms the concatenated head outputs.
    """

    def __init__(self, n_heads, d_model, dropout=0.1, bias=True) -> None:
        """
        Initializes the MultiHeadAttention module.
        Arguments:
            n_heads (int): The number of attention heads.
            d_model (int): The total dimension of the key, value, and query vectors.
            dropout (float): The dropout rate applied to attention scores.
            bias (bool): Whether bias should be included in linear layers.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_per_head = d_model // n_heads
        # Ensure the division is clean
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None) -> torch.tensor:
        """
        Forward pass for MultiHeadAttention.
        Arguments:
            x (torch.tensor): Input to the MultiHeadAttention Module
            mask (torch.tensor optional): Potential Mask to declare whether to hide part of the input
        Returns:
            torch.tensor: The attention weighted output as linear combination of v
        """
        seq_len, batch_size, _ = x.shape
        # Transform and reshape input to (batch_size, n_heads, seq_len, d_per_head)
        key = self.key(x).view(batch_size, seq_len, self.n_heads, self.d_per_head).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.n_heads, self.d_per_head).transpose(1, 2)
        query = self.query(x).view(batch_size, seq_len, self.n_heads, self.d_per_head).transpose(1, 2)

        # Compute attention (batch_size, n_heads, seq_len, d_per_head)
        attn_out, _ = self.attention(key, value, query, mask)

        # Concatenate heads and put through final linear layer
        out = attn_out.transpose(1, 2).contiguous().view(seq_len, batch_size, self.d_model)
        return self.out(out)

    def attention(self, k: torch.tensor, v: torch.tensor, q: torch.tensor, mask: Optional[torch.tensor] = None) \
            -> tuple[torch.tensor, torch.tensor]:
        """
        The attention mechanism computing a weighted linear combination of v
        based on the similarity of the according k and v entries, aka. "Scaled Dot Product Attention".
        Arguments:
            k (torch.tensor): Key vector
            v (torch.tensor): Value vector
            q (torch.tensor): Query vector
            mask (torch.tensor, optional): Mask to hide future entries
        Returns:
            torch.Tensor: The attention-weighted output tensor.
            torch.Tensor: The attention weights tensor.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_per_head ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = softmax(scores, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class DecoderLayer(nn.Module):
    """
    Represents a single layer within the decoder part of a Transformer model.

    This layer performs two sequences of multi-head self-attention and position-wise feedforward network operations,
    each followed by normalization and dropout for regularization. The layer is designed to process sequences in the
    context of the entire sequence and previous decoder layers' outputs, enabling effective sequence-to-sequence
    transformations.

    Attributes:
        norm1 (nn.LayerNorm): First layer normalization applied before the first multi-head attention operation.
        dropout1 (nn.Dropout): Dropout applied after the first multi-head attention operation.
        norm2 (nn.LayerNorm): Second layer normalization applied before the second multi-head attention operation.
        dropout2 (nn.Dropout): Dropout applied after the second multi-head attention operation.
        norm3 (nn.LayerNorm): Third layer normalization applied before the feedforward network operation.
        dropout3 (nn.Dropout): Dropout applied after the feedforward network operation.
        att1 (MultiHeadAttention): First multi-head attention mechanism, focusing on self-attention within the input sequence.
        att2 (MultiHeadAttention): Second multi-head attention mechanism, typically focusing on attention towards the encoder's output.
        feedforward (FeedForward): A feedforward neural network applied after the second attention mechanism.
        norm4 (nn.LayerNorm): Final layer normalization applied after the feedforward network operation.
    """

    def __init__(self, n_heads: int, d_model: int, linear_layer_size: int, dropout: float = 0.1, bias: bool = True) \
            -> None:
        """
        Initializes the DecoderLayer with specified parameters for multi-head attention and feedforward network.

        Parameters:
            n_heads (int): Number of attention heads in the multi-head attention mechanisms.
            d_model (int): The dimensionality of the input and output vectors for this layer.
            linear_layer_size (int): The size of the internal layer within the feedforward network.
            dropout (float): The dropout rate applied after attention operations and feedforward network.
            bias (bool): Whether to include bias terms in the linear transformations within the attention mechanisms and feedforward network.
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

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor]) -> torch.tensor:
        """
        Defines the forward pass of the DecoderLayer.

        Parameters:
            x (torch.Tensor): The input tensor to the decoder layer with shape [seq_len,bs,d_model].
            mask (torch.Tensor, optional): An optional tensor used for masking the attention mechanism to
        prevent it from attending to certain positions.

        Returns:
            torch.Tensor: The output tensor of the decoder layer with the same shape as the input tensor.
        """
        # First attention block
        x1 = self.norm1(x)
        x = x + self.dropout1(self.att1(x1, mask))

        # Second attention block
        x2 = self.norm2(x)
        x = x + self.dropout2(self.att2(x2, mask))

        # Feedforward block
        x3 = self.norm3(x)
        x = x + self.dropout3(self.feedforward(x3))

        # Final normalization
        x = self.norm4(x)
        return x


class Model(nn.Module):
    """
    The model consisting of a linear encoding layer, a decoder layer, and a linear output layer.
    Attributes:
        enc (nn.Linear): Encoding Layer
        dec (DecoderLayer): Decoding Layer
        out (nn.Linear): Output Layer
    """

    def __init__(self, n_heads: int, d_model: int, linear_layer_size: int, d_one_hot: int, dropout: float = 0.1,
                 bias: bool = True) -> None:
        """
        Constructor method of the Model module.
        Arguments:
            n_heads (int): The number of attention heads.
            d_model (int): The size of the K, V, Q and output vectors.
            linear_layer_size (int): The size of the feedforward network within the decoder.
            d_one_hot (int): The size of the input and output vector.
            dropout (float): Dropout rate.
            bias (bool): Whether to use bias in linear layers.
        """
        super().__init__()
        self.enc = nn.Linear(d_one_hot, d_model, bias=bias)
        self.dec = DecoderLayer(n_heads, d_model, linear_layer_size, dropout, bias)
        self.out = nn.Linear(d_model, d_one_hot, bias=bias)

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor]) -> torch.tensor:
        """
        Defines the computation performed at every call.
        Arguments:
            x (torch.tensor): The input to the module
            mask (torch.tensor, optional): Mask to hide future entries
        Returns:
            torch.tensor: The module's output
        """
        x = self.enc(x)
        x = self.dec(x, mask)
        return self.out(x)
