import torch
import torch.nn as nn

class model_embedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = nn.Embedding(config["model_zoo_size"]+1, config["embedding_dim"]) # +1 is because of "the null model"
        
    def forward(self, ms):
        return self.embedder(ms)