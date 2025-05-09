import torch
from torch import Tensor
import torch.nn as nn
from config import config

class PositionalEncoding(nn.Module):
    def __init__(self, max_routing: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_routing, d_model)

    def forward(self, routing: torch.Tensor, max_nodes: int, max_edges: int):
        B, _ = routing.shape
        node_pos, edge_pos = get_logical_routing_positions(routing, max_nodes, max_edges)

        # Expand to token level
        # nodes: each index in node_pos is repeated twice
        # edges: each index in edge_pos is repeated five times

        node_pos_expanded = node_pos.unsqueeze(-1).repeat(1, 1, 2)       # (B, max_nodes, 2)
        edge_pos_expanded = edge_pos.unsqueeze(-1).repeat(1, 1, 5)       # (B, max_edges, 5)

        # Flatten back to token-level position: (B, max_nodes*2), (B, max_edges*5)
        node_token_positions = node_pos_expanded.view(B, max_nodes * 2)
        edge_token_positions = edge_pos_expanded.view(B, max_edges * 5)


        node_pe = self.pos_embed(node_token_positions.clamp(min=0))
        edge_pe = self.pos_embed(edge_token_positions.clamp(min=0))
        return node_pe, edge_pe
    
def get_logical_routing_positions(routing, max_nodes, max_edges):
    B, max_routing = routing.shape
    node_positions = torch.full((B, max_nodes), -1, dtype=torch.long, device=routing.device)
    edge_positions = torch.full((B, max_edges), -1, dtype=torch.long, device=routing.device)

    for b in range(B):
        n_idx = 0
        e_idx = 0
        for t in range(max_routing):
            if routing[b, t] == 0 and n_idx < max_nodes:
                node_positions[b, n_idx] = t
                n_idx += 1
            elif routing[b, t] == 1 and e_idx < max_edges:
                edge_positions[b, e_idx] = t
                e_idx += 1

    return node_positions, edge_positions

def interleave_nodes_edges(nodes, edges, routing, rout_PAD_token):
    """
    nodes: (B, max_nodes*2, D)
    edges: (B, max_edges*5, D)
    routing: (B, max_nodes + max_edges)
    Returns: (B, max_nodes*2 + max_edges*5, D)
    """

    B, N_tokens, D = nodes.shape
    _, E_tokens, _ = edges.shape
    total_tokens = N_tokens + E_tokens

    result = torch.zeros((B, total_tokens, D), dtype=nodes.dtype, device=nodes.device)
    token_to_routing_idx = torch.full((B, total_tokens), rout_PAD_token, dtype=torch.long, device=nodes.device)

    for b in range(B):
        node_ptr = 0
        edge_ptr = 0
        t_idx = 0
        for r_idx, r in enumerate(routing[b]):
            if r == rout_PAD_token:
                continue  # skip padded routing entries
            elif r == 0:
                for _ in range(2):
                    if node_ptr < N_tokens:
                        result[b, t_idx] = nodes[b, node_ptr]
                        token_to_routing_idx[b, t_idx] = r_idx
                        node_ptr += 1
                        t_idx += 1
            elif r == 1:
                for _ in range(5):
                    if edge_ptr < E_tokens:
                        result[b, t_idx] = edges[b, edge_ptr]
                        token_to_routing_idx[b, t_idx] = r_idx
                        edge_ptr += 1
                        t_idx += 1

    return result, token_to_routing_idx

def generate_sample_batch(
    batch_size=2,
    max_nodes=config.dataset_config.max_num_nodes,
    max_edges=config.dataset_config.max_num_edges,
    vocab_size =config.dataset_config.vocab_size,
):
    B = batch_size
    max_routing_length = max_nodes + max_edges
    nodes = torch.zeros((B, max_nodes, 2), dtype=torch.long)
    node_mask = torch.zeros((B, max_nodes, 2), dtype=torch.bool)

    for b in range(B):
        num_nodes = torch.randint(5, max_nodes - 2, (1,)).item()
        nodes[b, :num_nodes, 0] = torch.randint(1, vocab_size, (num_nodes,))
        for i in range(num_nodes):
            nodes[b, i, 1] = i  # position
        node_mask[b, :num_nodes, :] = True

    # Random edge tokens [B, max_edges, 5]
    edges = torch.zeros((B, max_edges, 5), dtype=torch.long)
    edge_mask = torch.zeros((B, max_edges, 5), dtype=torch.bool)
    for b in range(B):
        num_edges = torch.randint(3, max_edges - 5, (1,)).item()
        edges[b, :num_edges] = torch.randint(1, vocab_size, (num_edges, 5))
        edge_mask[b, :num_edges, :] = True

    # Routing sequence [B, max_routing_length] with 0 (node), 1 (edge), 2 (PAD)
    routing = torch.full((B, max_routing_length), 2, dtype=torch.long)
    routing_mask = torch.zeros((B, max_routing_length), dtype=torch.bool)
    for b in range(B):
        sequence = []
        n_used = 0
        e_used = 0
        while len(sequence) < max_routing_length:
            if n_used < max_nodes and torch.rand(1).item() < 0.6:
                sequence.append(0)
                n_used += 1
            elif e_used < max_edges:
                sequence.append(1)
                e_used += 1
            else:
                break
        routing[b, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        routing_mask[b, :len(sequence)] = True

    # routing_next = shift left + fill end
    routing_next = torch.roll(routing, shifts=-1, dims=1)
    routing_next[:, -1] = 2 

    return {
        "nodes": nodes,
        "edges": edges,
        "routing": routing,
        "routing_next": routing_next,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "routing_mask": routing_mask,
    }