import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.preprocessing import LabelEncoder
import optuna
from tqdm import tqdm

#from models.sign_model import SignLanguageModel
from models.tunable_sign_model import TunableSignLanguageModel

# -------------------------------------------------------------
# Dataset Class (Same as train.py)
# -------------------------------------------------------------
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
        return (self.pose[idx],
                self.face[idx],
                self.lhand[idx],
                self.rhand[idx],
                self.pose_mask[idx],
                self.face_mask[idx],
                self.lhand_mask[idx],
                self.rhand_mask[idx],
                self.labels[idx])

# -------------------------------------------------------------
# Load Data
# -------------------------------------------------------------
pose = np.load('data/landmarks_norm/X_pose.npy')
face = np.load('data/landmarks_norm/X_face.npy')
lhand = np.load('data/landmarks_norm/X_lhand.npy')
rhand = np.load('data/landmarks_norm/X_rhand.npy')

pose_mask = np.load('data/landmarks_norm/M_pose.npy')
face_mask = np.load('data/landmarks_norm/M_face.npy')
lhand_mask = np.load('data/landmarks_norm/M_lhand.npy')
rhand_mask = np.load('data/landmarks_norm/M_rhand.npy')

labels = np.load('data/landmarks_norm/y.npy')

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
print("Classes:", encoder.classes_)

dataset = SignLanguageDataset(
    pose, face, lhand, rhand,
    pose_mask, face_mask, lhand_mask, rhand_mask, labels
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

# Tiny batch for MPS
BATCH_SIZE = 4

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------------------------------------------
# Objective Function for Optuna
# -------------------------------------------------------------
def objective(trial):
    # --- Suggest hyperparameters ---
    lstm_hidden = trial.suggest_int("lstm_hidden", 64, 256, step=32)
    cnn_out = trial.suggest_int("cnn_out", 32, 128, step=32)
    fc_dropout = trial.suggest_float("fc_dropout", 0.2, 0.6, step=0.1)
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-4)

    model = TunableSignLanguageModel(
        cnn_out=cnn_out,
        lstm_hidden=lstm_hidden,
        num_classes=num_classes,
        bidirectional=True,  # or trial.suggest_categorical("bidirectional", [True, False])
        dropout=fc_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    EPOCHS = 6  # Small for speed

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask, labels = [b.to(device) for b in batch]

            logits = model(pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask, labels = [b.to(device) for b in batch]
            logits = model(pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    return val_acc

# -------------------------------------------------------------
# Run Study
# -------------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, timeout=60*60*4)  # Max ~4 hours

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value:.2f}%")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save study for later
study.trials_dataframe().to_csv("optuna_tune_summary.csv", index=False)

