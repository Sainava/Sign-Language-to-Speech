import pandas as pd
import json
import os

# Load gloss file
csv_path = "data/raw/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"
df = pd.read_csv(csv_path)

# Extract unique words from all SIGN GLOSSES
unique_words = set()
for gloss in df["SIGN GLOSSES"].dropna():
    unique_words.update(gloss.strip().split())

# Sort and map each word to a unique integer ID
sorted_words = sorted(unique_words)
label_map = {word: idx for idx, word in enumerate(sorted_words)}

# Save to JSON
os.makedirs("data/processed", exist_ok=True)
json_path = "data/processed/label_map_sentence.json"
with open(json_path, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"Saved sentence-level label map with {len(label_map)} classes to {json_path}")
