"""
This code is to implement the IndRNN (only the recurrent part). The code is based on the implementation from 
https://github.com/StefOe/indrnn-pytorch/blob/master/indrnn.py.
Since this only contains the recurrent part of IndRNN, fully connected layers or convolutional layers are needed before it.
Please cite the following paper if you find it useful.
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN," 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5457-5466. 2018.
@inproceedings{li2018independently,
  title={Independently recurrent neural network (indrnn): Building A longer and deeper RNN},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5457--5466},
  year={2018}
}
"""


import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class IndRNNCell_onlyrecurrent(nn.Module):
    r"""An IndRNN cell with ReLU non-linearity. This is only the recurrent part where the input is already processed with w_{ih} * x + b_{ih}.

    .. math::
        input=w_{ih} * x + b_{ih}
        h' = \relu(input +  w_{hh} (*) h)
    With (*) being element-wise vector multiplication.

    Args:
        hidden_size: The number of features in the hidden state h

    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    """

    def __init__(self, hidden_size, 
                 hidden_max_abs=None, recurrent_init=None):
        super(IndRNNCell_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.weight_hh = Parameter(torch.Tensor(hidden_size))            
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "weight_hh" in name:
                if self.recurrent_init is None:
                    nn.init.uniform(weight, a=0, b=1)
                else:
                    self.recurrent_init(weight)

    def forward(self, input, hx):
        return F.relu(input + hx * self.weight_hh.unsqueeze(0).expand(hx.size(0), len(self.weight_hh)))


class IndRNN(nn.Module):
    r"""Applies an IndRNN with `ReLU` non-linearity to an input sequence. 
    This is the recurrent part where the input is already processed with w_{ih} * x + b_{ih}.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h.
        n_layer: Number of recurrent layers.
        batch_norm: If ``True``, then apply batch normalization to the output of each layer.
        bidirectional: If ``True``, becomes a bidirectional IndRNN. Default: ``False``
        recurrent_init: If not None, use this function to initialize the recurrent weight. Default: ``None``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. 
        - **h_0** of shape `(n_layer, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the IndRNN,
          for each `k`.
        - **h_n** of shape `(n_layer, batch, hidden_size * num_directions)`: tensor
          containing the hidden state for `k = seq_len`.
    """
    def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False, bidirectional=False, recurrent_init=None):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional
        # Create multiple IndRNNCell_onlyrecurrent layers
        self.cells = nn.ModuleList([
            IndRNNCell_onlyrecurrent(hidden_size, recurrent_init=recurrent_init)
            for _ in range(n_layer)
        ])
        if self.batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(n_layer)])

    
    def forward(self, input, h0=None):
        assert input.dim() == 2 or input.dim() == 3        
        num_directions = 2 if self.bidirectional else 1
        if h0 is None:
            h0 = input.data.new(self.n_layer * num_directions, input.size(-2), self.hidden_size).zero_()
        elif (h0.size(-1) != self.hidden_size) or (h0.size(-2) != input.size(-2)) or (h0.size(0) != self.n_layer * num_directions):
            raise RuntimeError(
                'The initial hidden size must be equal to input_size. Expected {}, got {}'.format(
                    (self.n_layer * num_directions, input.size(-2), self.hidden_size), h0.size()))
        
        # Handle packed sequence
        is_packed = isinstance(input, nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        
        # For the time being, this module only supports non-bidirectional RNNs.
        if self.bidirectional:
            raise NotImplementedError()  # Bidirectional IndRNN not implemented in this example

        # Loop through layers
        for i in range(self.n_layer):
            layer_input = input
            hx_cell = h0[i] 
            layer_outputs = []
            for seq_idx in range(input.size(0)):
                input_t = layer_input[seq_idx]
                if batch_sizes is not None:
                    input_t = input_t[:batch_sizes[seq_idx]]
                hx_cell = self.cells[i](input_t, hx_cell)
                if self.batch_norm:
                    hx_cell = self.bns[i](hx_cell)
                layer_outputs.append(hx_cell)
            output = torch.stack(layer_outputs, 0)  # stack the outputs from the cells for each layer 
            
            if batch_sizes is not None:
                output = nn.utils.rnn.PackedSequence(output, batch_sizes)
            
            input = output  # Prepare output for the next layer
        if is_packed:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        return output  # Only return the final output from the final layer
