import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
import numpy as np

MODEL_ID = "cisco-ai/SecureBERT2.0-base"
SAMPLE_SIZE = 5000
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128

# Load data
print("Loading data...")
train_df = pd.read_csv("/Users/aakashpremnath/HF_Model_Tester/data/data/train_processed.csv").sample(SAMPLE_SIZE, random_state=42)
test_df = pd.read_csv("/Users/aakashpremnath/HF_Model_Tester/data/data/test_processed.csv").sample(1000, random_state=42)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Label distribution:\n{train_df['label'].value_counts()}")

# Dataset class
class AlertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# Load model and tokenizer
print("\nLoading SecureBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

# Device — use MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Datasets
train_dataset = AlertDataset(train_df["text"], train_df["label"], tokenizer)
test_dataset = AlertDataset(test_df["text"], test_df["label"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
print("\nStarting fine-tuning...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Benign", "Threat"]))

# Save model
print("\nSaving model...")
model.save_pretrained("model/securebert-finetuned")
tokenizer.save_pretrained("model/securebert-finetuned")
print("Model saved to model/securebert-finetuned")