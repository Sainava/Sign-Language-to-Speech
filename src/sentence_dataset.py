# src/sentence_dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np

class ISLSentenceDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        # Match files by base name (e.g., 'are you free today_1')
        self.sample_ids = [
            fname.replace(".npy", "")
            for fname in os.listdir(features_dir)
            if fname.endswith(".npy")
        ]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        feature_path = os.path.join(self.features_dir, f"{sample_id}.npy")
        label_path = os.path.join(self.labels_dir, f"{sample_id}.pt")

        features = np.load(feature_path)
        labels = torch.load(label_path)

        return torch.tensor(features, dtype=torch.float32), labels


def pad_collate_sentence(batch):
    """
    Collate function for sentence-level batching.
    Pads features to the longest sequence in the batch.
    """
    from torch.nn.utils.rnn import pad_sequence
    features, labels = zip(*batch)

    lengths = torch.tensor([f.shape[0] for f in features])
    padded_features = pad_sequence(features, batch_first=True)  # (B, T, D)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # (B, T)

    return padded_features, padded_labels, lengths