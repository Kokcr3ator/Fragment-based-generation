import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, Union
from torchinfo import summary
from .blocks.router import Router
from .blocks.heads import NodeHead, EdgeHead
from .blocks.encoder import TransformerEncoder
from .utils import PositionalEncoding, interleave_nodes_edges, generate_sample_batch

from config import Config
config = Config()


class GraphGenerator(nn.Module):
    def __init__(self, config: Config = config):
        super().__init__()
        self._parse_config(config)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.node_encoder = TransformerEncoder(self.d_model, self.n_encoder_layers, self.n_heads, self.dropout)
        self.edge_encoder = TransformerEncoder(self.d_model, self.n_encoder_layers, self.n_heads, self.dropout)
        self.router = Router(self.d_model, self.router_hidden_dim, self.n_router_layers)
        self.node_head = NodeHead(self.d_model, self.n_decoder_layers, self.n_heads, self.dropout,
                                   self.max_fragment, self.max_node_id)
        self.edge_head = EdgeHead(self.d_model, self.n_decoder_layers, self.n_heads, self.dropout,
                                  self.max_node_id, self.max_rank, self.num_edge_types)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            batch_first=True,
        )
        self.positional_encoding = PositionalEncoding(self.max_num_nodes + self.max_num_edges, self.d_model)
    
    def summary(self, batch_size=2, depth=3):
        sample_input = generate_sample_batch(batch_size=batch_size)
        print("Model structure:\n")
        print(summary(self, input_data=(sample_input,), depth=depth, col_names=["input_size", "output_size", "num_params", "trainable"]))
        print("\nParameter count:")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,}")

    def _parse_config(self, config: Config) -> None:
        """
        Parse the configuration object to set up the model parameters.
        This method is called in the constructor to initialize the model.
        """
        self.d_model = config.model_config.d_model
        self.dropout = config.model_config.dropout
        self.n_heads = config.model_config.n_heads
        self.n_router_layers = config.model_config.n_router_layers
        self.router_hidden_dim = config.model_config.router_hidden_dim
        self.n_encoder_layers = config.model_config.n_encoder_layers
        self.n_decoder_layers = config.model_config.n_decoder_layers
        self.vocab_size = config.dataset_config.vocab_size
        self.max_num_nodes = config.dataset_config.max_num_nodes
        self.max_num_edges = config.dataset_config.max_num_edges
        self.max_rank = config.dataset_config.max_rank
        self.max_fragment = config.dataset_config.max_fragment
        self.max_node_id = config.dataset_config.max_node_id
        self.num_edge_types = config.dataset_config.num_edge_types
        self.rout_PAD_token = config.model_config.rout_PAD_token

    def forward(
        self,
        batch: Tensor,
    ) :            
        nodes, edges, routing, routing_next, node_mask, edge_mask, rout_mask, node_att_mask, \
        edge_att_mask, seq_att_mask, seq_pad_mask = batch.values()
        B, max_nodes = nodes.shape[:2]
        _, max_edges = edges.shape[:2]
        nodes = nodes.view(B, -1)
        edges = edges.view(B, -1)
        # embed nodes and edges
        nodes = self.embedding(nodes) # (B, 2*max_nodes, d_model)
        edges = self.embedding(edges) # (B, 5*max_edges, d_model)

        # add positional encoding
        node_pe, edge_pe = self.positional_encoding(routing, max_nodes, max_edges)
        nodes = nodes + node_pe
        edges = edges + edge_pe

        padding_mask_nodes = node_mask.view(B, -1)
        nodes = self.node_encoder(nodes,
                                   attn_mask = node_att_mask, 
                                key_padding_mask = padding_mask_nodes
                                   )
        padding_mask_edges = edge_mask.view(B, -1)
        edges = self.edge_encoder(edges,
                                   attn_mask = edge_att_mask,
                                 key_padding_mask = padding_mask_edges
                                   )

        # concatenate nodes and edges
        x, token_to_routing_idx = interleave_nodes_edges(nodes, edges, routing, self.rout_PAD_token)

        expanded_mask = seq_att_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        expanded_mask = expanded_mask.reshape(B * self.n_heads, x.shape[1], x.shape[1])
        x = self.attn(x,x,x, attn_mask = expanded_mask, key_padding_mask = seq_pad_mask )[0]
        router_logits = self.router(x)

        next_labels = torch.gather(routing_next, dim=1, index=token_to_routing_idx)  # (B, 2·max_nodes + 5·max_edges)
        # WIP
        # this still needs some work the idea is that for the node head the queries are going to be the tokens before the node
        # keys and values are the tokens before that 
        # same goes for the edge head
        # change the node and edge head architecture?

        return x
