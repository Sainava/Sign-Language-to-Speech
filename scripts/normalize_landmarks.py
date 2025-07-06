import numpy as np
import os

# === Load ===
X_pose = np.load("data/landmarks/X_pose.npy")
X_face = np.load("data/landmarks/X_face.npy")
X_lhand = np.load("data/landmarks/X_lhand.npy")
X_rhand = np.load("data/landmarks/X_rhand.npy")
y = np.load("data/landmarks/y.npy")

N, T, _, _ = X_pose.shape

# === Pose: normalize relative to shoulder width ===
def get_shoulder_width(pose_frame):
    L = pose_frame[11][:3]  # Left shoulder
    R = pose_frame[12][:3]  # Right shoulder
    return np.linalg.norm(L[:2] - R[:2])

X_pose_norm = np.zeros_like(X_pose)
X_lhand_norm = np.zeros_like(X_lhand)
X_rhand_norm = np.zeros_like(X_rhand)

M_pose = np.zeros((N, T), dtype=np.float32)
M_lhand = np.zeros((N, T), dtype=np.float32)
M_rhand = np.zeros((N, T), dtype=np.float32)

for i in range(N):
    for t in range(T):
        pose_f = X_pose[i, t]

        shoulder_width = get_shoulder_width(pose_f)
        if shoulder_width < 1e-5:
            continue  # skip if not valid

        # Use pose nose as center
        nose = pose_f[0][:3]
        X_pose_norm[i, t, :, :3] = (pose_f[:, :3] - nose) / shoulder_width
        X_pose_norm[i, t, :, 3] = pose_f[:, 3]  # Keep confidence

        M_pose[i, t] = 1.0

        # Normalize hands relative to same shoulders
        lh_f = X_lhand[i, t]
        rh_f = X_rhand[i, t]

        if not np.all(lh_f == 0):
            X_lhand_norm[i, t] = (lh_f - nose) / shoulder_width
            M_lhand[i, t] = 1.0

        if not np.all(rh_f == 0):
            X_rhand_norm[i, t] = (rh_f - nose) / shoulder_width
            M_rhand[i, t] = 1.0

# Face: just standardize to zero mean per frame
X_face_norm = np.zeros_like(X_face)
M_face = np.zeros((N, T), dtype=np.float32)

for i in range(N):
    for t in range(T):
        f = X_face[i, t]
        if np.all(f == 0):
            continue

        mean = np.mean(f, axis=0)
        X_face_norm[i, t] = f - mean
        M_face[i, t] = 1.0

# === Save ===
out_dir = "data/landmarks_norm"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X_pose.npy", X_pose_norm)
np.save(f"{out_dir}/X_face.npy", X_face_norm)
np.save(f"{out_dir}/X_lhand.npy", X_lhand_norm)
np.save(f"{out_dir}/X_rhand.npy", X_rhand_norm)
np.save(f"{out_dir}/M_pose.npy", M_pose)
np.save(f"{out_dir}/M_face.npy", M_face)
np.save(f"{out_dir}/M_lhand.npy", M_lhand)
np.save(f"{out_dir}/M_rhand.npy", M_rhand)
np.save(f"{out_dir}/y.npy", y)

print("Saved normalized landmarks and masks to:", out_dir)
