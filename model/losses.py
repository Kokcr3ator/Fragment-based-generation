import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional

def router_loss(router_logits: Tensor, routing_labels: Tensor) -> Tensor:
    """
    Binary classification loss to route tokens to node or edge head.

    Args:
        router_logits: Tensor of shape (B, T, 2), logits for node vs edge.
        routing_labels: Tensor of shape (B, T), with 0=node, 1=edge labels.

    Returns:
        Scalar loss tensor.
    """
    node_logit = router_logits[..., 0]           # (B, T)
    node_target = (routing_labels == 0).float()  # (B, T)
    return F.binary_cross_entropy_with_logits(node_logit, node_target)


def node_head_loss(
    node_outputs: Optional[Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]]],
    node_labels: Dict[str, Tensor]
) -> Tensor:
    """
    Cross-entropy loss for node head predictions.

    Args:
        node_outputs: Tuple of (indices, logits dict), or None.
        node_labels: Dict of ground truth labels keyed by field name.

    Returns:
        Scalar loss tensor.
    """
    if node_outputs is None:
        return torch.tensor(0.0, device=next(iter(node_labels.values())).device)

    idx, logits_dict = node_outputs
    loss = 0.0
    for key, logits in logits_dict.items():
        targets = node_labels[key][idx]
        if key == "node_id":
            loss += F.cross_entropy(logits, targets, ignore_index=-100)
        else:
            loss += F.cross_entropy(logits, targets)
    return loss


def edge_head_loss(
    edge_outputs: Optional[Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]]],
    edge_labels: Dict[str, Tensor]
) -> Tensor:
    """
    Cross-entropy loss for edge head predictions.

    Args:
        edge_outputs: Tuple of (indices, logits dict), or None.
        edge_labels: Dict of ground truth labels keyed by field name.

    Returns:
        Scalar loss tensor.
    """
    if edge_outputs is None:
        return torch.tensor(0.0, device=next(iter(edge_labels.values())).device)

    idx, logits_dict = edge_outputs
    loss = 0.0
    for key, logits in logits_dict.items():
        targets = edge_labels[key][idx]
        loss += F.cross_entropy(logits, targets)
    return loss


def total_loss(
    outputs: Dict[str, Optional[object]],
    routing_labels: Tensor,
    node_labels: Dict[str, Tensor],
    edge_labels: Dict[str, Tensor],
    w_router: float = 1.0
) -> Tensor:
    """
    Compute the total loss from router, node, and edge predictions.

    Args:
        outputs: Dictionary with keys 'router_logits', 'node_outputs', 'edge_outputs'.
        routing_labels: Tensor of shape (B, T)
        node_labels: Dict of ground truth tensors for nodes.
        edge_labels: Dict of ground truth tensors for edges.
        w_router: Weight on router loss term.

    Returns:
        Scalar total loss tensor.
    """
    Lr = router_loss(outputs['router_logits'], routing_labels)
    Ln = node_head_loss(outputs['node_outputs'], node_labels)
    Le = edge_head_loss(outputs['edge_outputs'], edge_labels)
    return w_router * Lr + Ln + Le
