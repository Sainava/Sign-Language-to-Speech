import numpy as np
import os

# === Load ===
X_pose = np.load("datasets/landmarks/X_pose.npy")
X_face = np.load("datasets/landmarks/X_face.npy")
X_lhand = np.load("datasets/landmarks/X_lhand.npy")
X_rhand = np.load("datasets/landmarks/X_rhand.npy")
y = np.load("datasets/landmarks/y.npy")

N, T, _, _ = X_pose.shape

# === Copy original ===
X_pose_masked = X_pose.copy()
X_face_masked = X_face.copy()
X_lhand_masked = X_lhand.copy()
X_rhand_masked = X_rhand.copy()

# === Make masks ===
M_pose = np.zeros((N, T), dtype=np.float32)
M_face = np.zeros((N, T), dtype=np.float32)
M_lhand = np.zeros((N, T), dtype=np.float32)
M_rhand = np.zeros((N, T), dtype=np.float32)

for i in range(N):
    for t in range(T):
        pose_f = X_pose[i, t]
        face_f = X_face[i, t]
        lh_f = X_lhand[i, t]
        rh_f = X_rhand[i, t]

        # === Pose ===
        if not np.all(pose_f == 0):
            M_pose[i, t] = 1.0

        # === Face ===
        if not np.all(face_f == 0):
            M_face[i, t] = 1.0

        # === Left hand ===
        if not np.all(lh_f == 0):
            M_lhand[i, t] = 1.0

        # === Right hand ===
        if not np.all(rh_f == 0):
            M_rhand[i, t] = 1.0

# === Save ===
out_dir = "datasets/landmarks_masked"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X_pose.npy", X_pose_masked)
np.save(f"{out_dir}/X_face.npy", X_face_masked)
np.save(f"{out_dir}/X_lhand.npy", X_lhand_masked)
np.save(f"{out_dir}/X_rhand.npy", X_rhand_masked)
np.save(f"{out_dir}/M_pose.npy", M_pose)
np.save(f"{out_dir}/M_face.npy", M_face)
np.save(f"{out_dir}/M_lhand.npy", M_lhand)
np.save(f"{out_dir}/M_rhand.npy", M_rhand)
np.save(f"{out_dir}/y.npy", y)

print("Saved mask-only version to:", out_dir)
