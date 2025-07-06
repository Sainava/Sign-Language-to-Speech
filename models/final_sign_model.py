import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustCNNBranch(nn.Module):
    """
    Branch for pose, face, hand:
    - Multiple 1D convs over landmarks.
    - Global mean over landmarks.
    - Global mean over time.
    """
    def __init__(self, in_dim, num_conv_layers=4, hidden_dim=64, final_dim=128):
        super().__init__()
        layers = []

        input_dim = in_dim
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=1))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        # Final projection layer
        layers.append(nn.Conv1d(hidden_dim, final_dim, kernel_size=1))
        layers.append(nn.ReLU())

        self.cnn = nn.Sequential(*layers)

    def forward(self, x, mask):
        """
        x: (B, T, N, F)
        mask: (B, T)
        """
        B, T, N, F = x.size()

        # Combine batch & time for CNN
        x = x.view(B * T, N, F).transpose(1, 2)  # (B*T, F, N)

        x = self.cnn(x)  # (B*T, C, N)

        # Mean over landmarks
        x = torch.mean(x, dim=2)  # (B*T, C)

        # Restore sequence
        x = x.view(B, T, -1)  # (B, T, C)

        # Mask invalid frames
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        x = x * mask  # zeros for invalid frames

        # Mean over valid frames
        sum_mask = mask.sum(dim=1)  # (B, 1)
        sum_mask = sum_mask.clamp(min=1)  # avoid div by zero
        out = x.sum(dim=1) / sum_mask  # (B, C)

        return out


class SignLanguageModel(nn.Module):
    def __init__(self, 
                 pose_dim=4, face_dim=3, hand_dim=3,
                 pose_points=33, face_points=468, hand_points=21,
                 branch_hidden_dim=64, branch_final_dim=128,
                 num_classes=20):
        super().__init__()

        self.pose_branch = RobustCNNBranch(
            in_dim=pose_dim,
            num_conv_layers=4,
            hidden_dim=branch_hidden_dim,
            final_dim=branch_final_dim
        )

        self.face_branch = RobustCNNBranch(
            in_dim=face_dim,
            num_conv_layers=4,
            hidden_dim=branch_hidden_dim,
            final_dim=branch_final_dim
        )

        self.lhand_branch = RobustCNNBranch(
            in_dim=hand_dim,
            num_conv_layers=4,
            hidden_dim=branch_hidden_dim,
            final_dim=branch_final_dim
        )

        self.rhand_branch = RobustCNNBranch(
            in_dim=hand_dim,
            num_conv_layers=4,
            hidden_dim=branch_hidden_dim,
            final_dim=branch_final_dim
        )

        combined_dim = branch_final_dim * 4

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
