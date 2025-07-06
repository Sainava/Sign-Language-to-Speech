import numpy as np
import os

# Load your split landmark arrays
pose = np.load('data/landmarks/X_pose.npy')   # (N, T, 33, 4)
face = np.load('data/landmarks/X_face.npy')   # (N, T, 468, 3)
lhand = np.load('data/landmarks/X_lhand.npy') # (N, T, 21, 3)
rhand = np.load('data/landmarks/X_rhand.npy') # (N, T, 21, 3)
y = np.load('data/landmarks/y.npy')

def normalize_landmarks(X):
    """
    Normalize landmarks: for each coordinate dimension,
    compute mean/std only on non-zero entries (valid landmarks).
    Zeros stay zero.
    """
    mask_landmark = ~(np.all(X == 0, axis=-1, keepdims=True))  # (N, T, L, 1)

    valid = X[mask_landmark.squeeze(-1)]
    mean = np.mean(valid, axis=0)
    std = np.std(valid, axis=0)

    print(f"Mean: {np.mean(mean):.4f}, Std: {np.mean(std):.4f}")

    # Normalize only valid points
    X_norm = np.where(mask_landmark, (X - mean) / (std + 1e-8), 0.0)

    # ✅ FRAME MASK: any landmark present means frame is valid
    mask_frame = np.any(mask_landmark, axis=(2, 3)).astype(np.float32)  # (N, T)

    return X_norm, mask_frame

pose_norm, pose_mask = normalize_landmarks(pose)
face_norm, face_mask = normalize_landmarks(face)
lhand_norm, lhand_mask = normalize_landmarks(lhand)
rhand_norm, rhand_mask = normalize_landmarks(rhand)

# Save normalized + frame-level masks
out_dir = 'data/landmarks_norm'
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, 'X_pose.npy'), pose_norm)
np.save(os.path.join(out_dir, 'X_face.npy'), face_norm)
np.save(os.path.join(out_dir, 'X_lhand.npy'), lhand_norm)
np.save(os.path.join(out_dir, 'X_rhand.npy'), rhand_norm)

np.save(os.path.join(out_dir, 'M_pose.npy'), pose_mask)
np.save(os.path.join(out_dir, 'M_face.npy'), face_mask)
np.save(os.path.join(out_dir, 'M_lhand.npy'), lhand_mask)
np.save(os.path.join(out_dir, 'M_rhand.npy'), rhand_mask)

np.save(os.path.join(out_dir, 'y.npy'), y)

print("Saved normalized landmarks and **frame-level masks** to:", out_dir)

# Check shapes
print(f"Pose: {pose_norm.shape}, Mask: {pose_mask.shape}")
print(f"Face: {face_norm.shape}, Mask: {face_mask.shape}")
print(f"Left Hand: {lhand_norm.shape}, Mask: {lhand_mask.shape}")
print(f"Right Hand: {rhand_norm.shape}, Mask: {rhand_mask.shape}")
print(f"Labels: {y.shape}")
