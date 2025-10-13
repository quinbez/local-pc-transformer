
import torch.nn as nn
from layers.t_layers.embedding import Embedding
from layers.t_layers.output import Output
from layers.t_layers.transformer_block import PCTransformerBlock
from utils.pc_utils import x_init as x_init

class GPTPCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block=nn.ModuleList([PCTransformerBlock(config) for _ in range(config.n_layer)])
        self.embedding = Embedding(config)
        self.output = Output(config)

    def forward(self, input_ids, position_ids, target_ids, requires_update=True):
        B, S = input_ids.shape
        D = self.config.n_embed
        T = self.config.T
        vocab_size = self.config.vocab_size
        
        device = input_ids.device

        # ---- Initialize latent states ----
        self.output.pc_layer.x = x_init(B, S, D, device=device)
        
        for block in self.block:
            block.mlp.pc_fc.x = x_init(B, S, D, device=device)
            block.attention.pc_layer.x = x_init(B, S, D, device=device)

        # ---- Clear energies before inference ----
        for module in self.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

        target_onehot = nn.functional.one_hot(target_ids, num_classes=vocab_size).float().to(device)

        # ---- Iterative top-down PC ----
        for t in range(T):
            logits = self.output(
                x=self.output.pc_layer.get_x(),
                target=target_onehot,
                step=t,
                requires_update=requires_update
            )
            
            for idx in reversed(range(len(self.block))):
                block = self.block[idx]

                if idx == len(self.block) - 1:         # Last block
                    target_mlp = self.output.pc_layer.get_x()
                else:
                    target_mlp = self.block[idx + 1].attention.pc_layer.get_x()     # Earlier blocks

                target_attn = block.mlp.pc_fc.get_x()
                
                mu_mlp, mu_attn = block(
                    target_mlp=target_mlp,
                    target_attn=target_attn,
                    t=t,
                    requires_update=requires_update
                )
            
            target_embed = self.block[0].attention.pc_layer.get_x()
            self.embedding(input_ids, position_ids, target_embed, t, requires_update)

        return logits