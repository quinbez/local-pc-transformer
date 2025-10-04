import torch.nn as nn
from .attention import Attention
from .mlp_block import MLPBlock

class PCTransformerBlock(nn.Module):
    """
    Block combining Attention and MLP layers with predictive coding.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLPBlock(config)

    def forward(self,target_mlp, target_attn, t, requires_update=True):
        mu = self.mlp(target_mlp, t, requires_update)
        self.attention(target_attn, t, requires_update)
        return mu
