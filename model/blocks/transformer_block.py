import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .norms import RMSNorm
from .ffn import GEGLUFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = GEGLUFeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d)
            attn_mask: Optional tensor of shape (T, T)
            key_padding_mask: Optional tensor of shape (B, T)

        Returns:
            Tensor of the same shape (B, T, d) after attention and FFN
        """
        q_norm = self.norm1(query)
        k_norm = self.norm1(key)
        v_norm = self.norm1(value)
        attn_out, _ = self.attn(
            q_norm,        
            k_norm,        
            v_norm,        
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        query = query + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(query))
        query = query + self.dropout(ffn_out)
        return query
