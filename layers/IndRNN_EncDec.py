import torch.nn as nn
from .IndRNN_onlyrecurrent import IndRNN

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, dropout=0., output_hidden=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_hidden = output_hidden
        self.rnn = IndRNN(input_size=d_model, hidden_size=d_model)
    def forward(self, x, **_):
        rnn_out, self.hidden = self.rnn(x)    # Assumes input as [B, L, d]
        if self.output_hidden:
            return rnn_out, self.hidden
        else:
            return rnn_out, None

class Decoder(nn.Module):
    def __init__(self, output_size, d_model, num_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.rnn = IndRNN(input_size=output_size, hidden_size=d_model)
        self.linear = nn.Linear(d_model, output_size)
    def forward(self, x, encoder_hidden_states, **_):
        rnn_out, self.hidden = self.rnn(x.unsqueeze(1))
        output = self.linear(rnn_out.squeeze(1))
        return output, self.hidden