import torch
import torch.nn as nn
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step, init_x
from typing import Optional

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
        self.x=None

    def forward(
            self,
            layer: nn.Linear,
            target: torch.Tensor,
            step: int = 0,
            requires_update: bool = True,
        ):
        x = self.x
        mu = layer(x)
        mu = torch.softmax(mu, dim=-1)

        # ---- Prediction error ----
        error = target - mu
        dE_dmu = -error

        dE_dx = torch.einsum("bsd,vd->bsv", dE_dmu, layer.weight) 

        x= x - self.local_lr * dE_dx

        if requires_update:
            with torch.no_grad():
                delta_W = torch.einsum("bsv,bsd->vd", dE_dmu, x)
                layer.weight.data -= torch.clamp(
                    self.local_lr * delta_W, -config.clamp_value, config.clamp_value
                )

        energy, step_errors = finalize_step(mu, target, error, step, "output")
        self._energy += energy
        self._errors.extend(step_errors)
        self.x=x
        return mu
    
    def get_energy(self): return self._energy
    def get_x(self): return self.x
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors