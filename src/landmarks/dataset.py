import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir):
        self.pose = np.load(os.path.join(data_dir, "X_pose.npy"))
        self.face = np.load(os.path.join(data_dir, "X_face.npy"))
        self.lhand = np.load(os.path.join(data_dir, "X_lhand.npy"))
        self.rhand = np.load(os.path.join(data_dir, "X_rhand.npy"))

        self.pose_mask = np.load(os.path.join(data_dir, "M_pose.npy"))
        self.face_mask = np.load(os.path.join(data_dir, "M_face.npy"))
        self.lhand_mask = np.load(os.path.join(data_dir, "M_lhand.npy"))
        self.rhand_mask = np.load(os.path.join(data_dir, "M_rhand.npy"))

        self.labels = np.load(os.path.join(data_dir, "y.npy"))

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)

        assert len(self.pose) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pose = torch.tensor(self.pose[idx], dtype=torch.float32)
        face = torch.tensor(self.face[idx], dtype=torch.float32)
        lhand = torch.tensor(self.lhand[idx], dtype=torch.float32)
        rhand = torch.tensor(self.rhand[idx], dtype=torch.float32)

        pose_mask = torch.tensor(self.pose_mask[idx], dtype=torch.float32)
        face_mask = torch.tensor(self.face_mask[idx], dtype=torch.float32)
        lhand_mask = torch.tensor(self.lhand_mask[idx], dtype=torch.float32)
        rhand_mask = torch.tensor(self.rhand_mask[idx], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask, label

    def num_classes(self):
        return len(self.le.classes_)

    def classes(self):
        return self.le.classes_

def make_dataloaders(data_dir, batch_size=8, val_split=0.2):
    dataset = SignLanguageDataset(data_dir)

    indices = np.arange(len(dataset))
    labels = dataset.labels

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=42, stratify=labels
    )

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.num_classes(), dataset.classes()
