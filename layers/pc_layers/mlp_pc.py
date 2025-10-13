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
        self.x = None

    def forward(self, 
                x: torch.Tensor,
                layers: dict[nn.Linear], 
                target: torch.Tensor, 
                layer_norm: nn.Module = None, 
                step: int = 0, 
                requires_update: bool = True, 
        ):

        fc1 = layers['fc1']
        fc2 = layers['fc2']

        #  Optional normalization
        x_norm = layer_norm(x) if layer_norm else x
        
        # Forward pass
        h = fc1(x_norm)                 # h = W1 x
        act = F.gelu(h)                 # a = GELU(h)
        pred = fc2(act)                 # mu = W2 a

        # Local prediction error
        error = target - pred

        # Local Updates
        delta_x = torch.einsum("bsd,dh->bsh", error, fc2.weight)
        delta_x = torch.einsum("bsh,hd->bsd", delta_x, fc1.weight)
        x = x + self.local_lr * delta_x

        if requires_update:
           with torch.no_grad():
                B, S, _ = x.shape

                delta_W2 = torch.einsum("bsd,bse->de", error, act) / (B * S)

                # Compute local hidden error for W1
                h_err = torch.einsum("bsd,de->bse", error, fc2.weight) 
                gelu_grad = torch.sigmoid(1.702 * h)  
                h_err = h_err * gelu_grad

                delta_W1 = torch.einsum("bse,bsd->ed", h_err, x_norm) / (B * S)

                # Clamped updates
                fc1.weight.data -= torch.clamp(self.local_lr * delta_W1,
                                               -config.clamp_value, config.clamp_value)
                fc2.weight.data -= torch.clamp(self.local_lr * delta_W2,
                                              -config.clamp_value, config.clamp_value)

        energy, step_errors = finalize_step(pred, target, error, step, "mlp")
        self._energy += energy
        self._errors.extend(step_errors)
        self.x = x

        return pred, act

    def get_energy(self): return self._energy
    def get_x(self): return self.x
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors
    def clear_errors(self): self._errors = []