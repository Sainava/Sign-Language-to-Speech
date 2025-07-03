# src/convert_sentence_labels_to_indices.py

import os
import json
import torch

# Paths
label_dir = "data/processed/hybrid_labels_sentences"
label_map_path = "data/processed/label_map_sentence.json"

# Load label map
with open(label_map_path) as f:
    label_map = json.load(f)

# Process all .txt files
for fname in os.listdir(label_dir):
    if not fname.endswith(".txt"):
        continue

    txt_path = os.path.join(label_dir, fname)
    pt_path = txt_path.replace(".txt", ".pt")

    with open(txt_path, "r") as f:
        glosses = f.read().strip().split()

    indices = [label_map[word] for word in glosses]
    tensor = torch.tensor(indices, dtype=torch.long)
    torch.save(tensor, pt_path)

print("✅ Converted all .txt label files to .pt index tensors.")
