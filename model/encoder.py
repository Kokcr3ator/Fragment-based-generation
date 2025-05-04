import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from config import Config
from .blocks import RMSNorm, TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model: int = config.model_config.d_model
        self.n_layers: int = config.model_config.n_encoder_layers
        self.dropout: float = config.model_config.dropout
        self.n_heads: int = config.model_config.n_heads

        self.layers = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.final_norm = RMSNorm(self.d_model)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply a stack of Transformer blocks with final normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Optional attention mask of shape (seq_len, seq_len)
            key_padding_mask: Optional mask of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model): transformed output
        """
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
        x = self.final_norm(x)
        return x
