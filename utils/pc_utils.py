import torch

def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, embedding_size, device = device)

def energy_fn(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the predictive coding energy between predicted and target activity.
    """
    return 0.5 * (mu - target) ** 2

def finalize_step(mu, target, error, t, layer_type):
    """
    Finalize a predictive coding inference step by computing energy and error statistics.
    """
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = energy_fn(mu, target).mean().item() 
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors