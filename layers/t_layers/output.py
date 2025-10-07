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
        self.pc_layer = PCOutput(T=config.T, local_lr=config.local_lr)

    def forward(self, target, step, requires_update: bool = True):
        mu = self.pc_layer(
            layer=self.fc, 
            target=target, 
            step=step, 
            requires_update=requires_update
        )
        return mu