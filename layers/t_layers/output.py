import torch
import torch.nn as nn
from ..pc_layers.output_pc import PCOutput
from ..model_config import ModelConfig as config

class Output(nn.Module):
    """
    Output layer of the transformer.
    - Linear projection from n_embed -> vocab_size
    - Integrated with predictive coding via PCOutput
    """
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.rms_norm = nn.RMSNorm(config.n_embed) 
        self.pc_layer = PCOutput(T=config.T, local_lr=config.local_lr)

    def forward(self, x, target, step, requires_update: bool = True):
        mu = self.pc_layer(
            x = x,
            layer=self.fc, 
            target=target, 
            layer_norm = self.rms_norm,
            step=step, 
            requires_update=requires_update
        )
        
        return mu