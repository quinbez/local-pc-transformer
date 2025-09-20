import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader
from data_preparation.config import encoded_dir, max_len, batch_size
from data_preparation.dataset import EncodedDataset

def get_datasets():
    """ Load train, validation, and test datasets from encoded token ID files."""
    train_dataset = EncodedDataset(encoded_dir/"train.pt", max_len)
    valid_dataset = EncodedDataset(encoded_dir/"valid.pt", max_len)
    test_dataset = EncodedDataset(encoded_dir/"test.pt", max_len)
    
    return train_dataset, valid_dataset, test_dataset

def get_loaders():
    """Wrap datasets into PyTorch DataLoaders with batching and shuffling."""
    train_dataset, valid_dataset, test_dataset = get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size= batch_size)
    test_loader = DataLoader(test_dataset, batch_size= batch_size)

    return train_loader, valid_loader, test_loader
