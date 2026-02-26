import pandas as pd

# Load datasets
train_df = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15_testing-set.csv")

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
print(f"\nAttack categories:\n{train_df['attack_cat'].value_counts()}")
print(f"\nLabel distribution:\n{train_df['label'].value_counts()}")

# Convert network flow rows into natural language text for SecureBERT
def row_to_text(row):
    return (
        f"Protocol: {row['proto']}. "
        f"Service: {row['service']}. "
        f"State: {row['state']}. "
        f"Source packets: {row['spkts']}, Destination packets: {row['dpkts']}. "
        f"Source bytes: {row['sbytes']}, Destination bytes: {row['dbytes']}. "
        f"TTL source: {row['sttl']}, TTL destination: {row['dttl']}."
    )

train_df["text"] = train_df.apply(row_to_text, axis=1)
test_df["text"] = test_df.apply(row_to_text, axis=1)

# Save processed versions
train_df[["text", "label", "attack_cat"]].to_csv("data/train_processed.csv", index=False)
test_df[["text", "label", "attack_cat"]].to_csv("data/test_processed.csv", index=False)

print("\nSample text:")
print(train_df["text"].iloc[0])
print("\nPreprocessing complete!")