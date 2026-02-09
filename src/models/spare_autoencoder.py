import torch
import torch.nn as nn
import numpy as np


class SparseAutoEncoder(nn.Module):
    """
    A sparse autoencoder that learns sparse feature representations.

    This autoencoder expands the input dimension to a larger hidden dimension,
    applies ReLU activation to encourage sparsity, and can optionally tie the
    encoder and decoder weights.
    """

    def __init__(self,
            d_model : int,
            expansion_factor : int,
            tie_weights : bool = False
            ) -> None:
        """
        Initialize the sparse autoencoder.

        Args:
            d_model: Dimension of the input/output space.
            expansion_factor: Factor by which to expand the hidden dimension
                (d_hidden = d_model * expansion_factor).
            tie_weights: If True, the decoder uses the transpose of the encoder weights.
                If False, the decoder has its own separate weight matrix.
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = self.d_model * expansion_factor

        self.W_enc = nn.Parameter(torch.randn(self.d_model, self.d_hidden)/np.sqrt(self.d_model))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))

        if tie_weights:
            self.tie_weights = True
        else:
            self.tie_weights = False
            self.W_dec = nn.Parameter(torch.randn(self.d_hidden, self.d_model)/np.sqrt(self.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
    
    def encode(self, x : torch.Tensor) -> torch.Tensor:
        """
        Encode input into sparse feature representation.

        Applies a linear transformation followed by ReLU activation to produce
        a sparse representation in the expanded hidden space.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Encoded features of shape (..., d_hidden) with ReLU activation applied.
        """
        return torch.relu(
            torch.matmul(x, self.W_enc) + self.b_enc
        )

    def decode(self, f : torch.Tensor) -> torch.Tensor:
        """
        Decode feature representation back to original input space.

        Applies a linear transformation to map from the hidden feature space
        back to the original input dimension. Uses either tied weights
        (transpose of encoder) or separate decoder weights.

        Args:
            f: Feature tensor of shape (..., d_hidden).

        Returns:
            Reconstructed output of shape (..., d_model).
        """
        W = self.W_enc.T if self.tie_weights else self.W_dec
        return torch.matmul(
            f, W
        ) + self.b_dec 
    
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Encodes the input into sparse features and then decodes back to the
        original space, returning both the intermediate features and the
        reconstruction.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            A tuple containing:
                - features: Sparse encoded features of shape (..., d_hidden).
                - results: Reconstructed output of shape (..., d_model).
        """
        features = self.encode(x)
        results = self.decode(features)
        return features, results