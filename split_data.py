"""
split_data.py — Split standardized data into train/test sets.

Input:  data/botsv3_standardized.csv  (already cleaned by standardize_training_data.py)
Output: data/botsv3_train.csv, data/botsv3_test.csv

Run: python split_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

INPUT = "data/botsv3_standardized.csv"
TRAIN_PATH = "data/botsv3_train.csv"
TEST_PATH = "data/botsv3_test.csv"
TEST_SIZE = 0.2
SEED = 42

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows from {INPUT}")
print(f"  Threat: {(df['label'] == 'threat').sum()} | Benign: {(df['label'] == 'benign').sum()}")

# Stratified split by label + source_category
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED,
    stratify=df[["label", "source_category"]],
)

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"\nTrain: {len(train_df)} rows -> {TRAIN_PATH}")
print(f"Test:  {len(test_df)} rows -> {TEST_PATH}")

# Per-category breakdown
for name, split in [("Train", train_df), ("Test", test_df)]:
    print(f"\n{name}:")
    for cat in sorted(split["source_category"].unique()):
        sub = split[split["source_category"] == cat]
        t = (sub["label"] == "threat").sum()
        b = (sub["label"] == "benign").sum()
        print(f"  {cat:12s}: {len(sub):>5} ({t:>4} threat, {b:>4} benign)")
