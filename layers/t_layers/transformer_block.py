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
        
        mu_mlp, _ = self.mlp(
            x = self.mlp.pc_fc.get_x(),
            target=target_mlp,
            step=t,
            requires_update=requires_update
        )
        
        mu_attn = self.attention(
            x=self.attention.pc_layer.get_x(), 
            target=target_attn,
            t=t,
            requires_update=requires_update
        )

        return mu_mlp, mu_attn
