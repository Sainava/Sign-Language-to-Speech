# generate_classes_list.py
import os

features_dir = "data/processed/features"
label_to_idx = {}

for fname in sorted(os.listdir(features_dir)):
    if not fname.endswith(".npy"):
        continue

    parts = fname.split("_")
    label = "_".join(parts[2:]).replace(".npy", "").replace("_", " ")

    if label not in label_to_idx:
        idx = len(label_to_idx)
        label_to_idx[label] = idx

# Sort by index to get list
idx_to_label = [None] * len(label_to_idx)
for label, idx in label_to_idx.items():
    idx_to_label[idx] = label

print("CLASSES = [")
for label in idx_to_label:
    print(f"    \"{label}\",")
print("]")
