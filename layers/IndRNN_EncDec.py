import torch
import torch.nn as nn
from layers.IndRNN_onlyrecurrent import IndRNN 

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, dropout=0., output_hidden=True, recurrent_weight_init=None):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_hidden = output_hidden
        self.input_layer = nn.Linear(d_model, d_model) # 
        self.indrnn = IndRNN(d_model, d_model, n_layer=num_layers, batch_norm=True, bidirectional=False, recurrent_init=recurrent_weight_init) # Pass hidden_size here
    
    def forward(self, x, **_):
        x = self.input_layer(x)
        output = self.indrnn(x)  
        if self.output_hidden:
            return output, None  
        else:
            return output  


class Decoder(nn.Module):
    def __init__(self, output_size, d_model, num_layers, dropout=0., recurrent_weight_init=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_layer = nn.Linear(output_size, d_model)  # Add a fully connected layer before IndRNN
        self.indrnn = IndRNN(d_model, d_model, n_layer=num_layers, batch_norm=True,
                             bidirectional=False, recurrent_init=recurrent_weight_init)
        self.linear = nn.Linear(d_model, output_size)
    
    def forward(self, x, encoder_hidden_states, **_):
        x = self.input_layer(x)
        output = self.indrnn(x.unsqueeze(1))  # Use h0 for initial hidden state
        output = self.linear(output.squeeze(1))
        return output, None
