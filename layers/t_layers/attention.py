import torch.nn as nn
from ..pc_layers.attention_pc import PCAttention

class Attention(nn.Module):
    """
    Multi-head self-attention with predictive coding.
    Wraps PCAttention which handles local error computation and updates.
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.num_heads

        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.o_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.rms_norm = nn.RMSNorm(config.n_embed)

        self.pc_layer = PCAttention(
            T=config.T,
            local_lr=config.local_lr,
        )

    def forward(self, x, target, t=0, requires_update=True):
        mu = self.pc_layer(
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            x,
            target,
            self.rms_norm,
            t,
            requires_update,
        )

        if self.training:
            mu = self.dropout(mu)
        return mu