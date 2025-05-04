import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, Union

from .router import Router
from .heads import NodeHead, EdgeHead
from .encoder import TransformerEncoder

from config import Config


class GraphGenerator(nn.Module):
    def __init__(self, config: Config):
        """
        Full graph generation model including encoder, router, and heads.

        Args:
            config: An object with attributes such as d_model, dropout, etc.
        """
        super().__init__()
        model_config = config.model_config
        dataset_config = config.dataset_config

        self.d_model: int = model_config.d_model
        self.vocab_size = dataset_config.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.encoder = TransformerEncoder(config)
        self.router = Router(config)
        self.node_head = NodeHead(config)
        self.edge_head = EdgeHead(config)
        self.pos_embedding = nn.Embedding(config.dataset_config.max_seq_len, self.d_model)

    def forward(
        self,
        input: Tensor,
        routing_labels: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None
    ) -> Dict[str, Union[Tensor, Optional[Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]]]]]:
        """
        Forward pass of the graph generator.

        Args:
            input: input sequence, Tensor of shape (B, T, d_model)
            routing_labels: Optional tensor of shape (B, T) with values 0 (node) or 1 (edge)
            attn_mask: Optional attention mask of shape (T, T)
            key_padding_mask: Optional padding mask of shape (B, T)

        Returns:
            Dictionary with:
                - 'router_logits': (B, T, 2) tensor of routing logits
                - 'node_outputs': WIP
                - 'edge_outputs': WIP
        """


