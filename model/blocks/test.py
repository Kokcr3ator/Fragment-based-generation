import torch
import torch.nn as nn
from .norms import RMSNorm
from .ffn import GEGLUFeedForward
from .transformer_block import TransformerBlock  

def check_no_nans_infs(tensor: torch.Tensor, name: str):
    assert not torch.isnan(tensor).any(), f"{name} output contains NaNs"
    assert not torch.isinf(tensor).any(), f"{name} output contains Infs"

def test_rmsnorm():
    print("Testing RMSNorm...")
    x = torch.randn(2, 4, 16)
    norm = RMSNorm(d_model=16)
    out = norm(x)
    print("RMSNorm output:")
    print(out)
    assert out.shape == x.shape, "RMSNorm output shape mismatch"
    check_no_nans_infs(out, "RMSNorm")
    print("RMSNorm OK.\n")

def test_geglu_ffn():
    print("Testing GEGLUFeedForward...")
    x = torch.randn(2, 4, 32)
    ffn = GEGLUFeedForward(d_model=32)
    out = ffn(x)
    print("GEGLUFeedForward output:")
    print(out)
    assert out.shape == x.shape, "GEGLU FFN output shape mismatch"
    check_no_nans_infs(out, "GEGLUFeedForward")
    print("GEGLUFeedForward OK.\n")

def test_transformer_block():
    print("Testing TransformerBlock...")
    batch_size, seq_len, d_model = 2, 5, 16
    block = TransformerBlock(d_model=d_model, n_heads=4, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    out = block(x)
    print("TransformerBlock output:")
    print(out)
    assert out.shape == x.shape, "TransformerBlock output shape mismatch"
    check_no_nans_infs(out, "TransformerBlock")
    print("TransformerBlock OK.\n")

if __name__ == "__main__":
    torch.manual_seed(420)

    test_rmsnorm()
    test_geglu_ffn()
    test_transformer_block()

    print("All block tests passed")

