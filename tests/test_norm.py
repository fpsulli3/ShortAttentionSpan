import torch 
from model.norm import LayerNorm
import torch.nn.functional as F

def test_layer_norm_output_matches_pytorch():
    x = torch.randn(4, 10)
    ln = LayerNorm(emb_dim=10)
    expected = F.layer_norm(x, normalized_shape=(10,), weight=ln.scale, bias=ln.shift)
    actual = ln(x)
    assert torch.allclose(actual, expected, atol=1e-5), "LayerNorm output doesn't match PyTorch"

def test_layer_norm_gradcheck():
    x = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
    ln = LayerNorm(emb_dim=10).double()
    assert torch.autograd.gradcheck(ln, (x,)), "LayerNorm gradience check failed"

    

