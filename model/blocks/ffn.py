import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_model * 2)  # GEGLU projects 2x larger
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj, gate = self.fc1(x).chunk(2, dim=-1)
        x = x_proj * F.gelu(gate)
        x = self.dropout(x)
        return self.fc2(x)
