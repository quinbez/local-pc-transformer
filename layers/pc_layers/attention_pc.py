import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step, init_x
from typing import Optional

class PCAttention(nn.Module):
    """
    - Predictive Coding Layer for Multi-Head Self-Attention.
    - Uses Q, K, V, O layers passed in from Attention.
    - Computes attention and predictions.
    - Updates weights locally (Hebbian-like).
    - Stores energy and error signals.
    """
    def __init__(self, T: int, local_lr: float):
        super().__init__()
        self.T = T
        self.local_lr = local_lr
        self._energy = 0.0
        self._errors = []
        self.x=None

    def forward(self, q_proj, k_proj, v_proj, o_proj, target, layer_norm, t: int, requires_update: bool, x: Optional[torch.Tensor] = None):
        x = self.get_x()  # use self.x from previous step

        B, S, D = x.shape
        num_heads = config.num_heads
        head_dim = D // num_heads

        # Q, K, V projections: [B, H, S, D/H]
        Q = q_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)  
        K = k_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
        V = v_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)

        # Attention Scores & Causal Mask 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_probs, V)   # [B, H, S, D/H]
        context = context.transpose(1, 2).contiguous().view(B, S, D)    # [B, S, D]

        mu = o_proj(context)
        mu = layer_norm(mu)
        mu = F.gelu(mu)

        error = target - mu  

        if requires_update:
            with torch.no_grad():
                dW_o = torch.einsum("bsd,bse->de", context, error) / (B * S)
                o_proj.weight.data += torch.clamp(self.local_lr * dW_o, -config.clamp_value, config.clamp_value)

                # Multi-head Q, K, V updates
                for h in range(num_heads):
                    q_slice = Q[:, h, :, :]  # [B, S, D]
                    k_slice = K[:, h, :, :]
                    v_slice = V[:, h, :, :]
                    
                    dW_q_h = torch.einsum("bsd,bse->de", q_slice, x) / (B * S)
                    dW_k_h = torch.einsum("bsd,bse->de", k_slice, x) / (B * S)
                    dW_v_h = torch.einsum("bsd,bse->de", v_slice, x) / (B * S)

                    start = h * head_dim
                    end = (h + 1) * head_dim

                    q_proj.weight.data[start:end, :] += torch.clamp(self.local_lr * dW_q_h, -config.clamp_value, config.clamp_value)
                    k_proj.weight.data[start:end, :] += torch.clamp(self.local_lr * dW_k_h, -config.clamp_value, config.clamp_value)
                    v_proj.weight.data[start:end, :] += torch.clamp(self.local_lr * dW_v_h, -config.clamp_value, config.clamp_value)

        energy, step_errors = finalize_step(mu, target, error, t, "attention")
        self._energy += energy
        self._errors.extend(step_errors)
        self.x=x

        return mu
        
    def get_energy(self): return self._energy
    def get_x(self): return self.x
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors