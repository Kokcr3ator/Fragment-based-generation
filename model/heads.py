import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from config import Config
from .blocks import RMSNorm, TransformerBlock


class NodeHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        model_config = config.model_config
        dataset_config = config.dataset_config
        
        d_model: int = model_config.d_model
        n_layers: int = model_config.n_layers
        n_heads: int = model_config.n_heads
        dropout: float = model_config.dropout
        vocab_size: int = dataset_config.node_vocab_size
        max_node_id: int = dataset_config.max_node_id

        # Shared mini-transformer backbone
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Two separate classification heads
        self.idx_head = nn.Linear(d_model, max_node_id)
        self.label_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Input tensor of shape (N_node, d_model)

        Returns:
            Dict with:
                - "node_id": (N_node, max_node_id)
                - "node_label": (N_node, vocab_size)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        outputs = {
            "node_id": self.idx_head(x),
            "node_label": self.label_head(x)
        }
        return outputs


class EdgeHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        model_config = config.model_config
        dataset_config = config.dataset_config

        d_model: int = model_config.d_model
        n_layers: int = model_config.n_layers
        n_heads: int = model_config.n_heads
        dropout: float = model_config.dropout
        max_node_id: int = dataset_config.max_node_id
        max_rank: int = dataset_config.max_rank
        num_edge_types: int = dataset_config.num_edge_types

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Separate output heads
        self.source_id_head = nn.Linear(d_model, max_node_id)
        self.source_site_head = nn.Linear(d_model, max_rank)
        self.dest_id_head = nn.Linear(d_model, max_node_id)
        self.dest_site_head = nn.Linear(d_model, max_rank)
        self.edge_type_head = nn.Linear(d_model, num_edge_types)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
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
            x = layer(x)
        x = self.final_norm(x)

        outputs = {
            "source_id": self.source_id_head(x),
            "source_site_number": self.source_site_head(x),
            "dest_id": self.dest_id_head(x),
            "dest_site_number": self.dest_site_head(x),
            "edge_type": self.edge_type_head(x)
        }
        return outputs
