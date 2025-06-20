import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import ISLDataset

def get_dataloaders(features_dir, batch_size=16, val_split=0.2, test_split=0.1):
    dataset = ISLDataset(features_dir)
    total_size = len(dataset)

    # Calculate split sizes
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(42))

    # Wrap in DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader
