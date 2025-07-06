import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        """
        Loads normalized landmark arrays + masks + labels.
        Encodes labels to integer indices.
        If augment=True: add random scale & shift.
        """
        self.pose = np.load(os.path.join(data_dir, "X_pose.npy"))
        self.face = np.load(os.path.join(data_dir, "X_face.npy"))
        self.lhand = np.load(os.path.join(data_dir, "X_lhand.npy"))
        self.rhand = np.load(os.path.join(data_dir, "X_rhand.npy"))

        self.pose_mask = np.load(os.path.join(data_dir, "M_pose.npy"))
        self.face_mask = np.load(os.path.join(data_dir, "M_face.npy"))
        self.lhand_mask = np.load(os.path.join(data_dir, "M_lhand.npy"))
        self.rhand_mask = np.load(os.path.join(data_dir, "M_rhand.npy"))

        self.labels = np.load(os.path.join(data_dir, "y.npy"))

        # Label encoding
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)

        self.augment = augment

    def random_scale_shift(self, x, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
        """
        Apply random scale and shift. Landmarks are already normalized,
        so small jitter is enough.
        """
        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range)
        return x * scale + shift

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pose = self.pose[idx]
        face = self.face[idx]
        lhand = self.lhand[idx]
        rhand = self.rhand[idx]

        if self.augment:
            pose = self.random_scale_shift(pose)
            face = self.random_scale_shift(face)
            lhand = self.random_scale_shift(lhand)
            rhand = self.random_scale_shift(rhand)

        pose = torch.tensor(pose, dtype=torch.float32)
        face = torch.tensor(face, dtype=torch.float32)
        lhand = torch.tensor(lhand, dtype=torch.float32)
        rhand = torch.tensor(rhand, dtype=torch.float32)

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


def make_dataloaders(data_dir, batch_size=16, val_split=0.2):
    """
    Creates train + val DataLoaders.
    """
    dataset = SignLanguageDataset(data_dir, augment=True)  # << Train set: augment ON

    num_samples = len(dataset)
    val_size = int(val_split * num_samples)
    train_size = num_samples - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    # Important: val should NOT augment
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, dataset.num_classes(), dataset.classes()
