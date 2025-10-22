import torch
import torch.nn.functional as F
from utils.data_utils import get_loaders
from layers.model_config import ModelConfig
from model import GPTPCModel

import optuna


def combined_loss(energy, ce_loss, alpha=0.5):
    """
    Combine energy and cross-entropy loss.
    alpha: weight between energy and CE loss (0.0 = only CE, 1.0 = only energy)
    """
    return alpha * energy + (1 - alpha) * ce_loss


def objective(trial):

    config = ModelConfig()
    
    # Hyperparameters to tune
    config.T = trial.suggest_int("T", 2, 14)
    config.n_layer = trial.suggest_int("n_layer", 2, 8)
    config.local_lr = trial.suggest_float("local_lr", 1e-6, 1e-3, log=True)

    alpha = 0.8  # Weight for combined loss
    train_loader, valid_loader, _ = get_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTPCModel(config).to(device)


    model.train()
    total_combined_loss = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5: 
            break

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).repeat(input_ids.size(0), 1)

        
        logits = model(input_ids, position_ids, target_ids, requires_update=True)

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Compute energy across layers
        layer_energies = []
        for module in model.modules():
            if hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None:
                    layer_energies.append(energy)
        batch_energy = sum(layer_energies) / len(layer_energies) if layer_energies else 0.0

        # Combine both losses
        combined = combined_loss(batch_energy, ce_loss, alpha)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Energy={batch_energy:.4f}, CE Loss={ce_loss.item():.4f}, Combined={combined.item():.4f}")

        total_combined_loss += combined.item()
        batch_count += 1

        # Clear stored states
        for module in model.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

    avg_combined_loss = total_combined_loss / batch_count if batch_count > 0 else float("inf")
    return avg_combined_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)

    print("========== Best Hyperparameters ==========")
    trial = study.best_trial
    print(f"Avg Combined Loss: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
