import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout_prob):
        super().__init__()
        self.proj_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(p=dropout_prob)
        self.resid_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        B, T, D = x.shape
        Hdim = 64
        H = int(D / Hdim)

        qkv = self.proj_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, H, Hdim).transpose(1, 2)
        k = k.view(B, T, H, Hdim).transpose(1, 2)
        v = v.view(B, T, H, Hdim).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(Hdim)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.proj_out(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
