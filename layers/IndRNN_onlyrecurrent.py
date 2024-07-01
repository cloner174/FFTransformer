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
import torch.nn as nn
import torch.nn.functional as F
import math

class IndRNNCell(nn.Module):
    
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
    def __init__(self, input_size, hidden_size, recurrent_clip_min=-1, recurrent_clip_max=-1, bias=True, activation='relu', num_layers=1, bidirectional=False):
        
        super(IndRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_clip_min = recurrent_clip_min
        self.recurrent_clip_max = recurrent_clip_max
        self.activation = getattr(F, activation)  # Allow any activation function from F
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList([nn.Linear(input_size if i == 0 else hidden_size * (2 if bidirectional else 1), hidden_size) for i in range(num_layers)])
        self.recurrent_kernels = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_size)) for _ in range(num_layers)])  # Separate recurrent kernel for each layer
        
        # Initialize recurrent kernels
        for recurrent_kernel in self.recurrent_kernels:
            nn.init.uniform_(recurrent_kernel, -1, 1) 
        if bias:
            self.biases = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_size)) for _ in range(num_layers)])
            for bias in self.biases:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layers[0].weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
        else:
            self.register_parameter('biases', None)
    
    def forward(self, input, hx):
        if self.bidirectional:
            hx_forward, hx_backward = hx[0], hx[1]  # Split hidden states
            outputs_forward, outputs_backward = [], []
        for i in range(self.num_layers):
            h = self.layers[i](input)
            h += hx * self.recurrent_kernels[i]
            if self.biases is not None:
                h += self.biases[i]
            h = torch.clamp(h, self.recurrent_clip_min, self.recurrent_clip_max)
            h = self.activation(h)
            input = h 
            if self.bidirectional:
                outputs_forward.append(h.unsqueeze(1))  # Store forward output
                h_backward = self.layers[i](input.flip(dims=[1]))
                h_backward += hx_backward * self.recurrent_kernels[i]
                if self.biases is not None:
                    h_backward += self.biases[i]
                h_backward = torch.clamp(h_backward, self.recurrent_clip_min, self.recurrent_clip_max)
                h_backward = self.activation(h_backward)
                input = h_backward
                outputs_backward.append(h_backward.unsqueeze(1))
        if self.bidirectional:
            return torch.cat(outputs_forward, dim=1), torch.cat(outputs_backward.flip(dims=[1]), dim=1)  
        else:
            return h


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
    
    def __init__(self, input_size, hidden_size, recurrent_clip_min=-1, recurrent_clip_max=-1, bias=True, activation='relu', return_sequences=False):
        super(IndRNN, self).__init__()
        self.cell = IndRNNCell(input_size, hidden_size, recurrent_clip_min, recurrent_clip_max, bias, activation)
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
    
    def forward(self, input):
        batch_size, seq_len, _ = input.size()
        hx = torch.zeros(batch_size, self.hidden_size, device=input.device)
        outputs = []
        for t in range(seq_len):
            hx = self.cell(input[:, t, :], hx)
            if self.return_sequences:
                outputs.append(hx.unsqueeze(1))
        if self.return_sequences:
            return torch.cat(outputs, dim=1), None
        else:
            return hx, None