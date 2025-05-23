import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))

