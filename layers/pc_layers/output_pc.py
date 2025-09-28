import torch
import torch.nn as nn
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step

class PCOutput(nn.Module):
    """
    Predictive Coding layer for the final output.
    - Computes mu = linear(x)
    - Computes error with respect to target
    - Updates weights locally
    - Stores energy and step errors
    """
    def __init__(self, T: int, local_lr: float):
        super().__init__()
        self.T = T
        self.local_lr = local_lr
        self._energy = 0.0
        self._errors = []

    def forward(self, layer: nn.Linear, x: torch.Tensor, target: torch.Tensor, t: int = 0, requires_update: bool = True):
        mu = layer(x)
        mu = torch.softmax(mu, dim=-1)

        # Error
        error = target - mu

        if requires_update:
            with torch.no_grad():
                B, S, D = x.shape
                delta_W = torch.einsum("bsd,bse->de", x, error)
                delta_W = delta_W.permute(1, 0) 
                delta_W /= (B * S)
                
                layer.weight.data += torch.clamp(self.local_lr * delta_W, -config.clamp_value, config.clamp_value)

        energy, step_errors = finalize_step(mu, target, error, t, "output")
        self._energy += energy
        self._errors.extend(step_errors)

        return mu
    
    def get_energy(self): return self._energy
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors