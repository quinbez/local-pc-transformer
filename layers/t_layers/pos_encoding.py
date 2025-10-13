import math
import torch
import torch.nn as nn

class SinusoidalPE(nn.Module):
    """
    Fixed (non-trainable) sinusoidal positional encoding.
    """
    def __init__(self, n_embed: int, block_size: int):
        super().__init__()
        
        pe = torch.zeros(block_size, n_embed)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() *
                            (-math.log(10000.0) / n_embed))
        
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        pe = pe.unsqueeze(0) 
        
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        position_ids: (batch, seq_len)
        returns: (batch, seq_len, n_embed)
        """
        B, S = position_ids.shape
        pos_embeddings = self.pe[:, :S, :].expand(B, -1, -1)
        
        return pos_embeddings
