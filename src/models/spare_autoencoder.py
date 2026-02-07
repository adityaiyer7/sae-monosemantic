import torch
import torch.nn as nn
import numpy as np


class SparseAutoEncoder(nn.Module):
    def __init__(self, 
            d_model : int,
            expansion_factor : int,
            tie_weights : bool = False
            ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_hidden = self.d_model * expansion_factor

        self.W_enc = nn.Parameter(nn.randn(self.d_model, self.d_hidden)/np.sqrt(self.d_model))
        self.b_enc = nn.Parameter(nn.zeros(self.d_hidden))

        if tie_weights:
            self.tie_weights = True
        else:
            self.tie_weights = False
            self.W_dec = nn.Parameter(nn.randn(self.d_hidden, self.d_model)/np.sqrt(self.d_hidden))
        self.b_dec = nn.Parameter(nn.zeros(self.d_model))
    
    def encode(self, x : torch.Tensor) -> torch.Tensor:
        return torch.relu(
            torch.matmul(x, self.W_enc) + self.b_enc 
        )

    def decode(self, f : torch.Tensor) -> torch.Tensor:
        W = self.W_enc.T if self.tie_weights else self.W_dec
        return torch.matmul(
            f, W
        ) + self.b_dec 
    
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        features = self.encode(x)
        results = self.decode(features)
        return features, results