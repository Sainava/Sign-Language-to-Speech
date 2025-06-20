import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ISLDataset(Dataset):
    def __init__(self, features_dir, save_label_map_path="data/processed/label_map.json"):
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        for fname in sorted(os.listdir(features_dir)):
            if not fname.endswith(".npy"):
                continue

            parts = fname.split("_")
            label = "_".join(parts[2:]).replace(".npy", "").replace("_", " ")  # e.g. "A_LOT" → "A LOT"

            if label not in self.label_to_idx:
                idx = len(self.label_to_idx)
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label

            file_path = os.path.join(features_dir, fname)
            self.samples.append(file_path)
            self.labels.append(self.label_to_idx[label])

        # ✅ Save idx_to_label (inverse mapping) for real-time inference
        os.makedirs(os.path.dirname(save_label_map_path), exist_ok=True)
        with open(save_label_map_path, "w") as f:
            json.dump(self.idx_to_label, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features = np.load(self.samples[idx])  # shape: (T, F)
        features = torch.from_numpy(features).float()
        label = self.labels[idx]
        return features, label


def pad_collate(batch):
    features, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in features])  # sequence lengths
    padded_features = pad_sequence(features, batch_first=True)  # shape: (B, T_max, F)
    return padded_features, torch.tensor(labels), lengths
