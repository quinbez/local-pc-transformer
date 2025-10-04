import torch.nn as nn
from ..pc_layers.embedding_pc import PCEmbed
from .pos_encoding import SinusoidalPE

class Embedding(nn.Module):
    """
    Embedding layer with predictive coding
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = SinusoidalPE(config.n_embed, config.block_size)
        self.rms_norm = nn.RMSNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

        self.pc_layer = PCEmbed(
            T=config.T,
            local_lr=config.local_lr,
        )

    def forward(self, input_ids, position_ids, target, t, requires_update=True):
        assert position_ids is not None, "position_ids must be provided"

        mu = self.pc_layer(
            self.word_embeddings,
            self.position_embeddings,
            input_ids,
            position_ids,
            target,
            self.rms_norm,
            t,
            requires_update
        )
        
        if self.training:
            mu = self.dropout(mu)
        return mu