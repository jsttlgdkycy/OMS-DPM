import torch
import torch.nn as nn
import torch.nn.functional as F

class ms_encoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_ratio = config["dropout_ratio"]

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True, dropout=self.dropout_ratio)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = F.normalize(out, 2, dim=-1)
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        return out