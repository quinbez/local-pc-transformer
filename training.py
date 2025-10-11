import torch
import torch.nn.functional as F
from utils.data_utils import get_loaders
from layers.model_config import ModelConfig
from model import GPTPCModel

def train(model, dataloader):
    """
    Trains the model using energy-based updates.

    Returns
    -------
    avg_energy : float
        Average energy across all batches.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    total_energy = 0.0
    total_ce_loss = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).repeat(input_ids.size(0), 1)

        logits = model(input_ids, position_ids, target_ids, requires_update=True)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0
        )

        total_ce_loss += ce_loss.item()

        layer_energies = []
        for module in model.modules():
            if hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None:
                    layer_energies.append(energy)

        batch_energy = sum(layer_energies) / len(layer_energies)
        total_energy += batch_energy
        batch_count += 1
        
        if batch_idx % 10 == 0:
            batch_perplexity = torch.exp(torch.tensor(ce_loss))
            print(f"  Batch {batch_idx}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {batch_perplexity:.4f}")

        # ---- Clear energy and errors for next batch ----
        for module in model.modules():
            if hasattr(module, "clear_energy"):
                module.clear_energy()
            if hasattr(module, "clear_errors"):
                module.clear_errors()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()

    return avg_energy, avg_perplexity

# ---- Training loop ----
if __name__ == "__main__":
    config = ModelConfig()
    train_loader, _, _ = get_loaders()
    model = GPTPCModel(config).to("cuda" if torch.cuda.is_available() else "cpu")
    train_energies = []

    print("========== Training Started ==========")
    
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1} started")
        avg_energy, avg_perplexity = train(model, train_loader)
        print(f"Epoch {epoch+1} | Avg Energy: {avg_energy:.4f} | Avg Perplexity: {avg_perplexity:.4f}")
        train_energies.append(avg_energy)

# ---- Save model ----
save_path = "pc_transformer.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}.")
