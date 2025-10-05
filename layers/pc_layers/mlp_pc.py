import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step, init_x
from typing import Optional


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
        self.x=None

    def forward(self, layers: dict[nn.Linear], target: torch.Tensor, 
            layer_norm: nn.Module = None, t: int = 0, requires_update: bool = True, 
            x: Optional[torch.Tensor] = None):
        if self.x is None:
           x = init_x(config.batch_size, config.block_size, config.n_embed, device=None)
        else:
           x = self.get_x()

        # Layers
        layer1 = layers['fc1']
        layer2 = layers['fc2']

    # Forward pass
        x_norm = layer_norm(x) if layer_norm else x
        mu_1 = layer1(x_norm)                  # [B, S, hidden]
        x2 = F.gelu(mu_1)                      # [B, S, hidden]
        mu = layer2(x2)                        # [B, S, D_out]

    # Error and energy
        error = target - mu

    # Backprop through both layers for x update
        dE_dx = torch.einsum("bsd,dh->bsh", error, layer2.weight)
        dE_dx = torch.einsum("bsh,hd->bsd", dE_dx, layer1.weight)
        x = x + self.local_lr * dE_dx

        if requires_update:
           with torch.no_grad():
                B, S, _ = x.shape

                # fc2 gradient: error^T * x2
                delta_W2 = torch.einsum("bsd,bse->de", error, x2) / (B * S)

                back_err = torch.einsum("bsd,de->bse", error, layer2.weight)  # [B,S,hidden]
                 # derivative of GELU approx
                gelu_grad = torch.sigmoid(1.702 * mu_1)  # smooth approx
                back_err = back_err * gelu_grad

                 # fc1 gradient: back_err^T * x_norm
                delta_W1 = torch.einsum("bse,bsd->ed", back_err, x_norm) / (B * S)

                layer1.weight.data += torch.clamp(self.local_lr * delta_W1,
                                              -config.clamp_value, config.clamp_value)
                layer2.weight.data += torch.clamp(self.local_lr * delta_W2,
                                              -config.clamp_value, config.clamp_value)

        # Finalize
        energy, step_errors = finalize_step(mu, target, error, t, "mlp")
        self._energy += energy
        self._errors.extend(step_errors)
        self.x = x

        return mu, mu_1

    def get_energy(self): return self._energy
    def get_x(self): return self.x
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors