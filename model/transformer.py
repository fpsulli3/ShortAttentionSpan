import torch.nn as nn
from model.feedforward import FeedForward
from model.norm import LayerNorm
from . import attention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = attention.SelfAttention(
           embed_dim=cfg["emb_dim"],
           dropout_prob=cfg["drop_rate"],
           n_heads=cfg["n_heads"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
        

