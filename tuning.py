import torch
from utils.data_utils import get_loaders
from layers.model_config import ModelConfig
from model import GPTPCModel

import optuna


def objective(trial):
    
    config = ModelConfig()
    config.T = trial.suggest_int("T", 2, 14)
    config.n_layer = trial.suggest_int("n_layer", 2, 8)
    config.local_lr = trial.suggest_float("local_lr", 1e-6, 1e-3, log=True)

    
    train_loader, valid_loader, _ = get_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTPCModel(config).to(device)

    
    model.train()
    total_energy = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2: 
            break

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).repeat(input_ids.size(0), 1)

        
        logits = model(input_ids, position_ids, target_ids, requires_update=True)

        
        layer_energies = []
        for module in model.modules():
            if hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None:
                    layer_energies.append(energy)
        batch_energy = sum(layer_energies) / len(layer_energies)
        total_energy += batch_energy
        batch_count += 1

        
        for module in model.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    return avg_energy 

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=20)

    print("========== Best Hyperparameters ==========")
    trial = study.best_trial
    print(f"Avg Energy: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
