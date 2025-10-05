import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_config import ModelConfig as config
from utils.pc_utils import finalize_step, init_x
from typing import Optional
import math 

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
        
        x_norm= layer_norm(x) if layer_norm else x
        
        # Q, K, V projections: [B, H, S, D/H]
        Q_norm=q_proj(x_norm)
        K_norm=k_proj(x_norm)
        V_norm=v_proj(x_norm)
        
        Q = Q_norm.view(B, S, num_heads, head_dim).transpose(1, 2)  
        K = K_norm.view(B, S, num_heads, head_dim).transpose(1, 2)
        V = V_norm.view(B, S, num_heads, head_dim).transpose(1, 2)

        # Attention Scores & Causal Mask 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_probs, V)   # [B, H, S, D/H]
        context = context.transpose(1, 2).contiguous().view(B, S, D)    # [B, S, D]
       
        mu = o_proj(context)
        # mu = F.gelu(mu)
        

        error = target - mu  
        dE_dmu= - error
        
        dE_dcontext = torch.matmul(dE_dmu, o_proj.weight)  # [B, S, D]
        
        # Reshape dE_dcontext for multi-head
        dE_dcontext_heads = dE_dcontext.view(B, S, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
        dE_dattn_probs = torch.matmul(dE_dcontext_heads, V.transpose(-2, -1))  # [B, H, S, S]
        
        # Gradient through softmax (softmax derivative)
        dE_dscores = attn_probs * (dE_dattn_probs - torch.sum(dE_dattn_probs * attn_probs, dim=-1, keepdim=True))
        dE_dscores = dE_dscores / (head_dim ** 0.5)  
        dE_dscores = dE_dscores.masked_fill(causal_mask, 0)  
        
        dE_dQ_heads = torch.matmul(dE_dscores, K)  # [B, H, S, D/H]
        dE_dK_heads = torch.matmul(dE_dscores.transpose(-2, -1), Q)  # [B, H, S, D/H]
        dE_dV_heads = torch.matmul(attn_probs.transpose(-2, -1), dE_dcontext_heads)  # [B, H, S, D/H]
        
        dE_dQ = dE_dQ_heads.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]
        dE_dK = dE_dK_heads.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D] 
        dE_dV = dE_dV_heads.transpose(1, 2).contiguous().view(B, S, D)  # [B, S, D]
        
        dE_dx_Q = torch.matmul(dE_dQ, q_proj.weight)  # [B, S, D]
        dE_dx_K = torch.matmul(dE_dK, k_proj.weight)  # [B, S, D]
        dE_dx_V = torch.matmul(dE_dV, v_proj.weight)  # [B, S, D]
        dE_dx = dE_dx_Q + dE_dx_K + dE_dx_V  # [B, S, D]
        
        # Update neural activity
        x = x - self.local_lr * dE_dx 
        
        if requires_update:
            with torch.no_grad():
                dW_o = torch.einsum("bsd,bse->de", context, dE_dmu) 
                o_proj.weight.data -= torch.clamp(self.local_lr * dW_o, -config.clamp_value, config.clamp_value)

                # Reshape gradients and inputs for head-specific computation
                dE_dQ_heads_flat = dE_dQ_heads.permute(0, 2, 1, 3).contiguous().view(B * S, num_heads * head_dim)  # [B*S, D]
                dE_dK_heads_flat = dE_dK_heads.permute(0, 2, 1, 3).contiguous().view(B * S, num_heads * head_dim)  # [B*S, D]
                dE_dV_heads_flat = dE_dV_heads.permute(0, 2, 1, 3).contiguous().view(B * S, num_heads * head_dim)  # [B*S, D]
                x_norm_flat = x_norm.view(B * S, D)  # [B*S, D]
                
                dW_q = torch.einsum("bd,be->de", x_norm_flat, dE_dQ_heads_flat) / (B * S)
                dW_k = torch.einsum("bd,be->de", x_norm_flat, dE_dK_heads_flat) / (B * S)
                dW_v = torch.einsum("bd,be->de", x_norm_flat, dE_dV_heads_flat) / (B * S)
                
                # Apply updates
                q_proj.weight.data -= torch.clamp(self.local_lr * dW_q, -config.clamp_value, config.clamp_value)
                k_proj.weight.data -= torch.clamp(self.local_lr * dW_k, -config.clamp_value, config.clamp_value)
                v_proj.weight.data -= torch.clamp(self.local_lr * dW_v, -config.clamp_value, config.clamp_value)
        energy, step_errors = finalize_step(mu, target, error, t, "attention")
        self._energy += energy
        self._errors.extend(step_errors)
        self.x=x

        return mu
        
    def get_energy(self): return self._energy
    def get_x(self): return self.x
    def clear_energy(self): self._energy = 0.0; self._errors = []
    def get_errors(self): return self._errors