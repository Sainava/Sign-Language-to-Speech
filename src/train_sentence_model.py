import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.sentence_dataset import ISLSentenceDataset, pad_collate_sentence

# === Configurations ===
FEATURE_DIM = 81
HIDDEN_DIM = 512
NUM_LAYERS = 2
BIDIRECTIONAL = True
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LABEL_MAP_PATH = "data/processed/label_map_sentence.json"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# === Load label map ===
import json
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
NUM_CLASSES = len(label_map)

# === Dataset ===
dataset = ISLSentenceDataset(
    features_dir="data/processed/hybrid_features_sentences",
    labels_dir="data/processed/hybrid_labels_sentences"
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_sentence)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_sentence)

# === Define the model ===
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        enc_out, (h_n, _) = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        logits = self.classifier(dec_out)
        return logits

model = Seq2SeqLSTM(FEATURE_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)  # ignore padded targets
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# === Training Loop ===
best_val_acc = 0
os.makedirs("models", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training")

    for  features, labels, lengths in loop:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(features)
        logits = logits.view(-1, NUM_CLASSES)
        labels = labels.view(-1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        mask = labels != -1
        correct += (preds[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        loop.set_postfix(loss=loss.item(), acc=correct / total if total > 0 else 0.0)

    scheduler.step()

    # === Validation ===
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            logits = logits.view(-1, NUM_CLASSES)
            labels = labels.view(-1)

            preds = torch.argmax(logits, dim=1)
            mask = labels != -1
            val_correct += (preds[mask] == labels[mask]).sum().item()
            val_total += mask.sum().item()

    val_acc = val_correct / val_total if val_total > 0 else 0.0
    print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_sentence_model.pt")
        print("✅ Saved best model.")

print("Training complete.")
