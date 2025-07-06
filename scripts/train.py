import os
import torch
import torch.nn as nn
from tqdm import tqdm

from models.sign_model import SignLanguageModel
from src.landmarks.dataset import make_dataloaders

if __name__ == "__main__":   # ✅ ADD THIS

    # --------------------------------------------
    # Data Loaders
    # --------------------------------------------

    train_loader, val_loader, num_classes, classes = make_dataloaders(
        data_dir="data/landmarks_norm",
        batch_size=8,
        val_split=0.2
    )

    print("Classes:", classes)

    # --------------------------------------------
    # Model, Loss, Optimizer, Scheduler
    # --------------------------------------------

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # === Best hyperparameters ===
    best_cnn_out = 96
    best_lstm_hidden = 64
    best_fc_dropout = 0.2
    best_lr = 0.0002596
    # =============================

    model = SignLanguageModel(
        cnn_out=best_cnn_out,
        lstm_hidden=best_lstm_hidden,
        num_classes=num_classes,
        bidirectional=True
    ).to(device)

    model.classifier = nn.Sequential(
        nn.Linear(best_lstm_hidden * 2 * 4, 256),
        nn.ReLU(),
        nn.Dropout(best_fc_dropout),
        nn.Linear(256, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # --------------------------------------------
    # Training Loop
    # --------------------------------------------

    best_val_acc = 0

    for epoch in range(1, 201):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask, labels = [b.to(device) for b in batch]

            logits = model(pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask, labels = [b.to(device) for b in batch]

                logits = model(pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_sign_model_hp_tuned.pth")
            print(f"Saved new best model at Val Acc: {val_acc:.2f}%")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
