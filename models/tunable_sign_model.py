import torch
import torch.nn as nn

class LandmarkBranch(nn.Module):
    def __init__(self, in_dim, cnn_out, lstm_hidden, lstm_layers=1, bidirectional=True, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        B, T, N, F = x.size()
        x = x.view(B * T, F, N)
        x = self.cnn(x)
        x = torch.mean(x, dim=2)
        x = x.view(B, T, -1)

        mask = mask.unsqueeze(-1)
        x = x * mask

        output, _ = self.lstm(x)
        output = output * mask

        sum_mask = mask.sum(dim=1).clamp(min=1)
        feat = output.sum(dim=1) / sum_mask

        return feat


class TunableSignLanguageModel(nn.Module):
    def __init__(self, pose_dim=4, face_dim=3, hand_dim=3,
                 cnn_out=64, lstm_hidden=128, num_classes=20,
                 bidirectional=True, dropout=0.5):
        super().__init__()

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        self.pose_branch = LandmarkBranch(
            in_dim=pose_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden,
            bidirectional=bidirectional, dropout=dropout
        )
        self.face_branch = LandmarkBranch(
            in_dim=face_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden,
            bidirectional=bidirectional, dropout=dropout
        )
        self.lhand_branch = LandmarkBranch(
            in_dim=hand_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden,
            bidirectional=bidirectional, dropout=dropout
        )
        self.rhand_branch = LandmarkBranch(
            in_dim=hand_dim, cnn_out=cnn_out, lstm_hidden=lstm_hidden,
            bidirectional=bidirectional, dropout=dropout
        )

        combined_dim = lstm_output_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
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
