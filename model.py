
import torch.nn as nn
from layers.t_layers.embedding import Embedding
from layers.t_layers.attention import Attention
from layers.t_layers.mlp_block import MLPBlock
from layers.t_layers.output import Output
from layers.t_layers.transformer_block import PCTransformerBlock
from utils.pc_utils import init_x
class GPTPCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block=nn.ModuleList([PCTransformerBlock(config) for _ in range(config.n_layer)])
        self.embedding = Embedding(config)
        self.output = Output(config)
        T=config.T
    def forward(self, input_ids, position_ids, target, requires_update=True):
        T= self.config.T
        
        for block in self.block:
            block.attention.pc_layer.x = init_x(
                self.config.batch_size, self.config.block_size, self.config.n_embed, device=input_ids.device
                )
            block.mlp.pc_fc.x = init_x(
                self.config.batch_size, self.config.block_size, self.config.n_embed, device=input_ids.device
                )

        self.output.pc_layer.x = init_x(
                 self.config.batch_size, self.config.block_size, self.config.n_embed, device=input_ids.device
                )
        
        for module in self.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
        for t in range(T):
            logits = self.output(target, t, requires_update)
            for idx in range(len(self.block) - 1, -1, -1):
                block=self.block[idx]
                if idx==len(self.block) - 1:
                    target_mlp = self.output.pc_layer.get_x()
                else:
                    target_mlp = self.block[idx+1].attention.pc_layer.get_x()
                target_attn = self.block[idx].mlp.pc_fc.get_x()
                block(target_mlp, target_attn, t, requires_update)
                # block.attention(target_attn, t, requires_update)
            target_embed= self.block[0].attention.pc_layer.get_x()
            self.embedding(input_ids, position_ids, target_embed, t, requires_update)

        return logits