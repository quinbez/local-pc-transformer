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
        self.x = None
        self.T = T
        self.local_lr = local_lr
        self._energy = 0.0
        self._errors = []

    def forward(
            self,
            x: torch.Tensor,
            layer: nn.Linear,
            target: torch.Tensor,
            layer_norm: nn.Module = None,
            step: int = 0,
            requires_update: bool = True,
        ):
        
        x_norm = layer_norm(x)
        mu = layer(x_norm)
        mu_probs = torch.softmax(mu, dim=-1)

        # ---- Prediction error ----
        error = target - mu_probs
        dE_dmu = -error

        dE_dx = torch.einsum("bsv,vd->bsd", dE_dmu, layer.weight) 

        self.x = x - self.local_lr * dE_dx

        if requires_update:
            with torch.no_grad():
                delta_W = torch.einsum("bsv,bsd->vd", dE_dmu, x_norm)
                layer.weight.data -= torch.clamp(
                    self.local_lr * delta_W, -config.clamp_value, config.clamp_value
                )

        energy, step_errors = finalize_step(mu_probs, target, error, step, "output")
        self._energy += energy
        self._errors.extend(step_errors)
      
        return mu
    
    def get_energy(self): return self._energy
    def get_x(self): return self.x
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors
    def clear_errors(self): self._errors = []