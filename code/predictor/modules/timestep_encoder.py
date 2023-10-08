import torch
import torch.nn as nn
import math

class timestep_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_temb_dim = config["input_temb_dim"]
        output_emb_dim = config["output_temb_dim"] 
        
        self.input_temb_dim = input_temb_dim

        self.fc1 = nn.Linear(input_temb_dim, output_emb_dim)
        self.fc2 = nn.Linear(output_emb_dim, output_emb_dim)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def nonlinearity(self, x):
        return x*torch.sigmoid(x)

    def forward(self, t):
        if len(t.shape)==1:
            t = self.get_timestep_embedding(t, self.input_temb_dim)
        elif len(t.shape)==2:
            temb = torch.zeros(t.size(0), t.size(1), self.input_temb_dim).to(t.device)
            for b in range(t.size(0)):
                temb[b, :, :] = self.get_timestep_embedding(t[b], self.input_temb_dim)
            t = temb
        else:
            raise ValueError(f"The shape of timesteps is {t.shape}")
        
        t = self.fc1(t)
        t = self.nonlinearity(t)
        t = self.fc2(t)

        return t