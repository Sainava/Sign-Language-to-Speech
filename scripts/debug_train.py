import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

from models.sign_model import SignLanguageModel

# Load data
X = np.load('data/landmarks/X_balanced.npy')
y = np.load('data/landmarks/y_balanced.npy')

print("Loaded shapes:", X.shape, y.shape)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)
print("Encoded labels min/max:", y_encoded.min(), y_encoded.max())

# Torch tensors
X = torch.tensor(X, dtype=torch.float32)
y_encoded = torch.tensor(y_encoded, dtype=torch.long)

# Dataset + dataloader
dataset = TensorDataset(X, y_encoded)
train_dl = DataLoader(dataset, batch_size=4, shuffle=True)

# Get 1 batch
xb, yb = next(iter(train_dl))
print("Batch X shape:", xb.shape)
print("Batch y shape:", yb.shape)

print("X min/max/mean:", xb.min().item(), xb.max().item(), xb.mean().item())
print("First yb:", yb)

# Model
model = SignLanguageModel(num_landmarks=X.shape[2], num_coords=X.shape[3], num_classes=num_classes)
print(model)

# One forward pass
outputs = model(xb)
print("Outputs shape:", outputs.shape)
print("Outputs sample:", outputs[0])

# Loss check
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, yb)
print("Loss:", loss.item())

# Try backward
loss.backward()
print("Backward pass OK ✅")
