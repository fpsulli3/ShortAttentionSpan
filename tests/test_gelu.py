import torch 
from model.gelu import GELU 

def test_gelu_against_torch():
    x = torch.linspace(-3, 3, steps=100)
    gelu = GELU()
    expected = torch.nn.functional.gelu(x, approximate='tanh')
    actual = gelu(x) 
    assert torch.allclose(actual, expected, atol=1e-5), "GELU output does not match torch"

def test_gelu_gradcheck():
    gelu = GELU()
    x = torch.randn(10, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gelu, (x,)), "GELU gradient check failed"

