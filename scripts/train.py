import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from models.sign_model import SignLanguageModel
from src.landmarks.dataset import make_dataloaders

if __name__ == "__main__":

    # === Create output dir ===
    os.makedirs("models", exist_ok=True)

    # --------------------------------------------
    # Data Loaders
    # --------------------------------------------
    train_loader, val_loader, num_classes, classes = make_dataloaders(
        data_dir="datasets/landmarks_masked",
        batch_size=4,
        val_split=0.2
    )
    print("Classes:", classes)

    # --------------------------------------------
    # Model, Loss, Optimizer, Scheduler
    # --------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    best_cnn_out = 64
    best_lstm_hidden = 96
    best_fc_dropout = 0.5
    best_lr = 0.0004888498364531054

    model = SignLanguageModel(
        cnn_out=best_cnn_out,
        lstm_hidden=best_lstm_hidden,
        num_classes=num_classes,
        bidirectional=True,
        fc_dropout=best_fc_dropout  # <-- Add this to __init__
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=0.0007768214685554032)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # --------------------------------------------
    # Metric tracking
    # --------------------------------------------
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0

    for epoch in range(1, 201):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask, labels = [b.to(device) for b in batch]

            logits = model(pose, face, lhand, rhand, pose_mask, face_mask, lhand_mask, rhand_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss /= len(train_loader)
        train_acc = 100 * correct_train / total_train

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

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_web_model_2.pth")
            print(f"Saved new best model at Val Acc: {val_acc:.2f}%")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    torch.save(model.state_dict(), "models/last_epoch_model.pth")
    print("Saved final epoch model to models/last_epoch_model.pth")

    df = pd.DataFrame(history)
    df.to_csv("training_history.csv", index=False)
    print("Saved training history to training_history.csv")
