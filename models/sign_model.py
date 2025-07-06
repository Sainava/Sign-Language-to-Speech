import torch
import torch.nn as nn


class LandmarkBranch(nn.Module):
    """
    One branch for pose/face/hand:
    - Deep 1D CNN over landmarks per frame
    - LSTM over sequence
    - Mask: zero out invalid frames
    """
    def __init__(self, in_dim, cnn_out, lstm_hidden, lstm_layers=1, bidirectional=True):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, cnn_out, kernel_size=1),
            nn.BatchNorm1d(cnn_out),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.bidirectional = bidirectional
        self.lstm_hidden = lstm_hidden

    def forward(self, x, mask):
        """
        x: (B, T, N, F)
        mask: (B, T)
        """
        B, T, N, F = x.size()

        # CNN
        x = x.view(B * T, F, N)  # (B*T, F, N)
        x = self.cnn(x)          # (B*T, C, N)
        x = torch.mean(x, dim=2)  # global mean landmarks → (B*T, C)

        # Reshape sequence
        x = x.view(B, T, -1)  # (B, T, C)

        # Mask: zero invalid frames
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        x = x * mask  # zeroed

        # LSTM
        output, _ = self.lstm(x)  # (B, T, H * num_directions)

        # Mask output sequence:
        mask = mask  # (B, T, 1)
        output = output * mask  # (B, T, H * D)

        # Mean over valid timesteps:
        sum_mask = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        feat = output.sum(dim=1) / sum_mask  # (B, H * D)

        return feat


class SignLanguageModel(nn.Module):
    def __init__(self, pose_dim=4, face_dim=3, hand_dim=3,
                 cnn_out=64, lstm_hidden=128, num_classes=20,
                 bidirectional=True):
        super().__init__()

        # Each LSTM is bidirectional → output dim doubles
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        self.pose_branch = LandmarkBranch(
            in_dim=pose_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden, bidirectional=bidirectional
        )
        self.face_branch = LandmarkBranch(
            in_dim=face_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden, bidirectional=bidirectional
        )
        self.lhand_branch = LandmarkBranch(
            in_dim=hand_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden, bidirectional=bidirectional
        )
        self.rhand_branch = LandmarkBranch(
            in_dim=hand_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden, bidirectional=bidirectional
        )

        combined_dim = lstm_output_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, pose, face, lhand, rhand,
                pose_mask, face_mask, lhand_mask, rhand_mask):
        pose_feat = self.pose_branch(pose, pose_mask)
        face_feat = self.face_branch(face, face_mask)
        lhand_feat = self.lhand_branch(lhand, lhand_mask)
        rhand_feat = self.rhand_branch(rhand, rhand_mask)

        combined = torch.cat([pose_feat, face_feat, lhand_feat, rhand_feat], dim=1)
        logits = self.classifier(combined)
        return logits
