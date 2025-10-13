import torch
import torch.nn as nn
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step

class PCEmbed(nn.Module):
    """
    Predictive Coding Layer for Embedding.
    - Maps tokens into vector space (word + positional embeddings).
    - Predicts next layer input.
    - Computes error with respect to target.
    - Updates embedding weights locally.
    - Stores energy for monitoring.
    """
    def __init__(self, T: int, local_lr: float):
        super().__init__()
        self.x = None
        self.T = T
        self.local_lr = local_lr
        self._energy = 0.0
        self._errors = []

    def forward(self, word_layer, pos_layer, input_ids, position_ids, target, layer_norm, t: int, requires_update: bool):
        input_ids = torch.clamp(input_ids, max=config.vocab_size - 1)
        position_ids = torch.clamp(position_ids, max=config.block_size - 1)

        # Prediction
        word_emb = word_layer(input_ids)
        pos_emb = pos_layer(position_ids).detach()
        mu = word_emb + pos_emb
        mu = layer_norm(mu)
        mu = nn.functional.gelu(mu)
        
        if mu.dim() == 4 and mu.size(0) == 1:
            mu = mu.squeeze(0)
        if target.dim() == 4 and target.size(0) == 1:
            target = target.squeeze(0)

        # Error computation
        error = target - mu

        # Weight update
        if requires_update:
            with torch.no_grad():
                unique_word_ids = torch.unique(input_ids)
                for word_id in unique_word_ids:
                    mask = (input_ids == word_id)
                    if mask.any():
                        word_error = error[mask].mean(dim=0)
                        word_layer.weight.data[word_id] += torch.clamp(
                            self.local_lr * word_error, 
                            -config.clamp_value, 
                            config.clamp_value
                        )
                           
        energy, step_errors = finalize_step(mu, target, error, t, "embed")
        self._energy += energy
        self._errors.extend(step_errors)

        return mu
    
    def get_energy(self): return self._energy
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors
    def clear_errors(self): self._errors = []