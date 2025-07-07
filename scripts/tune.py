import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import optuna
from tqdm import tqdm

from models.tunable_sign_model import TunableSignLanguageModel

# ----------------------------------------
# Dataset class
# ----------------------------------------
class SignLanguageDataset(Dataset):
    def __init__(self, pose, face, lhand, rhand,
                 pose_mask, face_mask, lhand_mask, rhand_mask, labels):
        self.pose = torch.tensor(pose, dtype=torch.float32)
        self.face = torch.tensor(face, dtype=torch.float32)
        self.lhand = torch.tensor(lhand, dtype=torch.float32)
        self.rhand = torch.tensor(rhand, dtype=torch.float32)

        self.pose_mask = torch.tensor(pose_mask, dtype=torch.float32)
        self.face_mask = torch.tensor(face_mask, dtype=torch.float32)
        self.lhand_mask = torch.tensor(lhand_mask, dtype=torch.float32)
        self.rhand_mask = torch.tensor(rhand_mask, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.pose[idx],
            self.face[idx],
            self.lhand[idx],
            self.rhand[idx],
            self.pose_mask[idx],
            self.face_mask[idx],
            self.lhand_mask[idx],
            self.rhand_mask[idx],
            self.labels[idx]
        )

# ----------------------------------------
# Load data
# ----------------------------------------
pose = np.load('datasets/landmarks_masked/X_pose.npy')
face = np.load('datasets/landmarks_masked/X_face.npy')
lhand = np.load('datasets/landmarks_masked/X_lhand.npy')
rhand = np.load('datasets/landmarks_masked/X_rhand.npy')

pose_mask = np.load('datasets/landmarks_masked/M_pose.npy')
face_mask = np.load('datasets/landmarks_masked/M_face.npy')
lhand_mask = np.load('datasets/landmarks_masked/M_lhand.npy')
rhand_mask = np.load('datasets/landmarks_masked/M_rhand.npy')

labels = np.load('datasets/landmarks_masked/y.npy')
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
print(f"Classes: {encoder.classes_}")

# Create dataset and stratified split
full_dataset = SignLanguageDataset(
    pose, face, lhand, rhand,
    pose_mask, face_mask, lhand_mask, rhand_mask,
    labels
)
indices = np.arange(len(labels))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)
train_set = Subset(full_dataset, train_idx)
val_set   = Subset(full_dataset, val_idx)

# ----------------------------------------
# Optuna objective
# ----------------------------------------
def objective(trial):
    # Hyperparameter suggestions
    lstm_hidden   = trial.suggest_int("lstm_hidden",   64, 256, step=32)
    cnn_out       = trial.suggest_int("cnn_out",       32, 128, step=32)
    fc_dropout    = trial.suggest_float("fc_dropout",  0.2, 0.6, step=0.1)
    lr            = trial.suggest_loguniform("lr",     1e-5, 5e-4)
    weight_decay  = trial.suggest_loguniform("wd",    1e-6, 1e-2)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    batch_size    = trial.suggest_categorical("batch_size",    [4, 8, 16])

    # DataLoaders with dynamic batch size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TunableSignLanguageModel(
        cnn_out=cnn_out,
        lstm_hidden=lstm_hidden,
        num_classes=num_classes,
        bidirectional=bidirectional,
        dropout=fc_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )

    # Train & Validate
    best_acc = 0.0
    EPOCHS = 5  # fewer epochs for tuning
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            pose, face, lhand, rhand, pm, fm, lm, rm, lbls = [b.to(device) for b in batch]
            logits = model(pose, face, lhand, rhand, pm, fm, lm, rm)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pose, face, lhand, rhand, pm, fm, lm, rm, lbls = [b.to(device) for b in batch]
                logits = model(pose, face, lhand, rhand, pm, fm, lm, rm)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        val_acc = 100 * correct / total
        trial.report(val_acc, epoch)
        scheduler.step(val_acc)

        # prune
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_acc = max(best_acc, val_acc)

    return best_acc

# ----------------------------------------
# Run study
# ----------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Save summary CSV
os.makedirs("optuna", exist_ok=True)
study.trials_dataframe().to_csv("optuna/optuna_tune_summary.csv", index=False)
print("Study completed. Summary saved to optuna/optuna_tune_summary.csv")
