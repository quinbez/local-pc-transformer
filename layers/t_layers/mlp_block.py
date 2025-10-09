import torch.nn as nn
from ..pc_layers.mlp_pc import PCMLP

class MLPBlock(nn.Module):
    """
    Standard MLP block with two linear layers.
    fc1: expands dimension
    fc2: projects back to embedding dimension
    Optional layer norms applied after each layer.
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)

        self.rms_norm = nn.RMSNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

        self.pc_fc = PCMLP(T=config.T, local_lr=config.local_lr)

    def forward(self, x, target, step, requires_update: bool = True):
        pred, act = self.pc_fc( 
            x = x,
            layers = {'fc1': self.fc1, 'fc2': self.fc2}, 
            target = target, 
            layer_norm = self.rms_norm, 
            step = step, 
            requires_update = requires_update
        )

        if self.training:
            pred = self.dropout(pred)

        return pred, act