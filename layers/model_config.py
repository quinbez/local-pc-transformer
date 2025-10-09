class ModelConfig:
    vocab_size = 4000
    T = 5                  
    local_lr = 1e-7         
    clamp_value = 0.01        
    n_embed = 64            
    block_size = 128          
    dropout = 0.1
    num_heads = 4
    n_layer = 4
    batch_size=8
    num_epochs = 5