import torch
import torch.nn as nn
import torch.nn.functional as F

class regression_head(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()

        self.out_dims = config["out_dims"]

        self.MLP = nn.ModuleList()
        in_dim = input_dim
        for i in range(len(self.out_dims)+1):
            if i<len(self.out_dims):
                self.layer = nn.Sequential(
                    nn.Linear(in_dim, self.out_dims[i]),
                    nn.ReLU(),
                )
                in_dim = self.out_dims[i]
            else:
                self.layer = nn.Linear(in_dim, 1)

            self.MLP.append(self.layer)
            
    def forward(self, x):
        x = F.normalize(x, 2, dim=-1)
        for i in range(len(self.out_dims)+1):
            x = self.MLP[i](x)
        x = x.view(-1)
        return x