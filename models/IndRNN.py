import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class IndRNNCell(nn.Module):
    
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