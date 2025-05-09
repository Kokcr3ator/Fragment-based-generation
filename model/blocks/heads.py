import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from config import Config
from .norms import RMSNorm
from .transformer_block import TransformerBlock


class NodeHead(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, max_fragment: int, max_node_id: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.max_fragment = max_fragment
        self.max_node_id = max_node_id

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # 2 heads: one for node id and one for the node label
        self.idx_head = nn.Linear(d_model, max_node_id)
        self.label_head = nn.Linear(d_model, max_fragment)

    def forward(self, x: Tensor, attn_mask = None, key_padding_mask = None) -> Dict[str, Tensor]:
        """
        Args:
            x: Input tensor of shape (N_node, d_model)

        Returns:
            Dict with:
                - "node_id": (N_node, max_node_id)
                - "node_label": (N_node, vocab_size)
        """
        for layer in self.layers:
            x = layer(x, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        x = self.final_norm(x)

        outputs = {
            "node_id": self.idx_head(x),
            "node_label": self.label_head(x)
        }
        return outputs


class EdgeHead(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, max_node_id: int, max_rank: int, num_edge_types: int):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.max_node_id = max_node_id
        self.max_rank = max_rank
        self.num_edge_types = num_edge_types

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # 5 heads for source node id, source connection site, dest node id, dest connection site, and edge type
        self.source_id_head = nn.Linear(d_model, max_node_id)
        self.source_site_head = nn.Linear(d_model, max_rank)
        self.dest_id_head = nn.Linear(d_model, max_node_id)
        self.dest_site_head = nn.Linear(d_model, max_rank)
        self.edge_type_head = nn.Linear(d_model, num_edge_types)

    def forward(self, x: Tensor, attn_mask = None, key_padding_mask = None) -> Dict[str, Tensor]:
        """
        Args:
            x: Input tensor of shape (N_edge, d_model)

        Returns:
            Dict with:
                - "source_id": (N_edge, max_node_id)
                - "source_site_number": (N_edge, max_site_number)
                - "dest_id": (N_edge, max_node_id)
                - "dest_site_number": (N_edge, max_site_number)
                - "edge_type": (N_edge, num_edge_types)
        """
        for layer in self.layers:
            x = layer(x, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        x = self.final_norm(x)

        outputs = {
            "source_id": self.source_id_head(x),
            "source_site_number": self.source_site_head(x),
            "dest_id": self.dest_id_head(x),
            "dest_site_number": self.dest_site_head(x),
            "edge_type": self.edge_type_head(x)
        }
        return outputs
