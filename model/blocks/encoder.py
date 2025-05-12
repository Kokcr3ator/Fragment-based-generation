import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from .norms import RMSNorm
from .transformer_block import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model: int = d_model
        self.n_layers: int = n_layers
        self.dropout: float = dropout
        self.n_heads: int = n_heads

        self.layers = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        self.final_norm = RMSNorm(self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
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
                query = query,
                key = key,
                value= value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
        x = self.final_norm(x)
        return x
