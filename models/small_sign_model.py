import torch
import torch.nn as nn


class LandmarkBranch(nn.Module):
    """
    CNN-only branch for pose, face, or hands.
    Does:
      - 1D CNN over landmarks in each frame
      - Mean over landmarks
      - Mean over valid frames (masked)
    """
    def __init__(self, in_dim, cnn_out):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, cnn_out, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x, mask):
        """
        x: (B, T, N, F)
        mask: (B, T)
        """
        B, T, N, F = x.size()

        # Reshape: treat landmarks as "spatial", features as channels
        x = x.view(B * T, F, N)  # (B*T, F, N)
        x = self.cnn(x)          # (B*T, C, N)
        x = torch.mean(x, dim=2) # (B*T, C)

        # Back to sequence
        x = x.view(B, T, -1)     # (B, T, C)

        # Apply mask to zero out invalid frames
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        x = x * mask

        # Average over valid frames
        sum_x = torch.sum(x, dim=1)  # (B, C)
        valid_counts = mask.sum(dim=1)  # (B, 1)
        out = sum_x / (valid_counts + 1e-8)

        return out  # (B, C)


class SignLanguageModel(nn.Module):
    def __init__(self,
                 pose_dim=4, face_dim=3, hand_dim=3,
                 cnn_out=64, num_classes=20):
        super().__init__()

        self.pose_branch = LandmarkBranch(pose_dim, cnn_out)
        self.face_branch = LandmarkBranch(face_dim, cnn_out)
        self.lhand_branch = LandmarkBranch(hand_dim, cnn_out)
        self.rhand_branch = LandmarkBranch(hand_dim, cnn_out)

        combined_dim = cnn_out * 4

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
