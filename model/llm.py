import torch
import torch.nn as nn
from model.transformer import TransformerBlock
from model.norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        self.embedding_dropout = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])],
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
      
        batch_size, seq_len = in_idx.shape

        # Look up token embeddings for the current batch
        token_embeddings = self.token_embedding(in_idx)
        
        # Look up positional embeddings for the current batch
        positional_embeddings = self.positional_embedding(torch.arange(seq_len, device=in_idx.device))

        # Add the positional embedings to the token embeddings
        x = token_embeddings + positional_embeddings

        x = self.embedding_dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

