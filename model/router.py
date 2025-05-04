import torch
import torch.nn as nn
from torch import Tensor
from config import Config


class Router(nn.Module):
    def __init__(self, config: Config):
        """
        Router MLP that outputs logits for node vs. edge prediction.

        Args:
            config: An object with attributes:
                - d_model (int): input feature dimension
                - router_hidden_dim (int): hidden dimension for the MLP
        """
        super().__init__()
        model_config = config.model_config
        d_model: int = model_config.d_model
        hidden_dim: int = model_config.router_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # Two logits: node vs edge
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, 2), the routing logits.
        """
        return self.mlp(hidden_states)
