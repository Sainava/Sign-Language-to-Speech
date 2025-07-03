import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split

from src.dataset import ISLDataset, pad_collate 
from src.model import LSTMClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Configs
INPUT_DIM = 144 # for hybrid features
# INPUT_DIM = 18 # for YOLO features
HIDDEN_DIM = 512
NUM_LAYERS = 2
BIDIRECTIONAL = True
NUM_CLASSES = 114
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# Load dataset
full_dataset = ISLDataset(features_dir="data/processed/hybrid_features")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    collate_fn=pad_collate  
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=pad_collate
)

# Initialize model, loss, optimizer
model = LSTMClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                       num_classes=NUM_CLASSES, bidirectional=BIDIRECTIONAL).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
best_val_acc = 0.0
train_losses = []
train_accs = []
val_accs = []
os.makedirs("models", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training")

    for features, labels, lengths in loop:
        features, labels, lengths = features.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(features, lengths)  # 👈 pass lengths
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct / total)


    scheduler.step()

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for features, labels, lengths in val_loader:
            features, labels, lengths = features.to(device), labels.to(device), lengths.to(device)
            outputs = model(features, lengths)  # 👈 pass lengths
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)


    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1} | Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")

    train_losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total)
    val_accs.append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pt")
        print("✅ Saved best model.")

print("Training complete.")
