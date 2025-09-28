import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step


class PCMLP(nn.Module):
    """
    Predictive Coding layer for a single linear layer in MLP.
    - Computes mu = layer(x)
    - Applies layer_norm and activation
    - Computes error against target
    - Updates weights locally 
    - Stores energy and step errors
    """
    def __init__(self, T: int, local_lr: float):
        super().__init__()
        self.T = T
        self.local_lr = local_lr
        self._energy = 0.0
        self._errors = []

    def forward(self, layer: nn.Linear, x: torch.Tensor, target: torch.Tensor, 
                layer_norm: nn.Module = None, t: int = 0, requires_update: bool = True):
        # Compute prediction
        mu = layer(x)
        mu = layer_norm(mu)
        mu = F.gelu(mu) 

        error = target - mu

        if requires_update:
            with torch.no_grad():
                B, S, D_in = x.shape
                D_out = error.shape[-1]
                
                x_flat = x.reshape(B*S, D_in)   # [B*S, D_in]
                error_flat = error.reshape(B*S, D_out)  # [B*S, D_out]

                delta_W = torch.matmul(error_flat.T, x_flat) / (B*S)
                layer.weight.data += torch.clamp(
                    self.local_lr * delta_W, -config.clamp_value, config.clamp_value
                )

        energy, step_errors = finalize_step(mu, target, error, t, "mlp")
        self._energy += energy
        self._errors.extend(step_errors)

        return mu

    def get_energy(self): return self._energy
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors