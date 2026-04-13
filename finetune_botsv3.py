"""
finetune_botsv3.py — Fine-tune SecureBERT 2.0 on BOTS v3 labeled alert data.

Designed for: NVIDIA RTX 3050 (4GB VRAM) with CUDA
Comparison target: UNSW-NB15 model (90% accuracy, Threat F1 0.93)

Usage:
    python finetune_botsv3.py

Input:  data/botsv3_train.csv, data/botsv3_test.csv (from preprocess_botsv3.py)
Output: models/secureBERT_botsv3/ (saved model + tokenizer)
        results/botsv3_classification_report.txt
"""

import pandas as pd
import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID = "cisco-ai/SecureBERT2.0-base"
TRAIN_PATH = "data/botsv3_train.csv"
TEST_PATH = "data/botsv3_test.csv"
MODEL_SAVE_PATH = "models/secureBERT_botsv3"
RESULTS_DIR = "results"

# Training hyperparameters (tuned for RTX 3050 4GB VRAM)
BATCH_SIZE = 16          # Safe for 4GB VRAM with 512 tokens
MAX_LEN = 256            # BOTS text is shorter than 512 on average
EPOCHS = 4               # More epochs since data is diverse (4 source types)
LEARNING_RATE = 2e-5     # Standard for BERT fine-tuning
WARMUP_STEPS = 200       # Gradual LR ramp-up
WEIGHT_DECAY = 0.01      # Regularization
USE_AMP = True           # Mixed precision — saves VRAM on RTX 3050
GRAD_ACCUM_STEPS = 2     # Effective batch size = 16 * 2 = 32


# ── DATASET ───────────────────────────────────────────────────────────────────

class AlertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=USE_AMP):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            acc = 100 * correct / total
            print(f"  Epoch {epoch+1}/{total_epochs} | Step {step+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=USE_AMP):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(threat)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        total_loss / len(loader),
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SecureBERT 2.0 Fine-Tuning on BOTS v3")
    print("=" * 60)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, training will be slow")
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Train threats: {(train_df['label_int']==1).sum()} ({100*(train_df['label_int']==1).mean():.1f}%)")
    print(f"Test threats:  {(test_df['label_int']==1).sum()} ({100*(test_df['label_int']==1).mean():.1f}%)")

    # Compute class weights for imbalanced data (27% threat / 73% benign)
    n_threat = (train_df["label_int"] == 1).sum()
    n_benign = (train_df["label_int"] == 0).sum()
    weight_threat = n_benign / n_threat  # upweight minority class
    weight_benign = 1.0
    class_weights = torch.tensor([weight_benign, weight_threat], dtype=torch.float32).to(device)
    print(f"\nClass weights: benign={weight_benign:.2f}, threat={weight_threat:.2f}")

    # Load model and tokenizer
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        problem_type="single_label_classification",
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Datasets and loaders
    train_dataset = AlertDataset(train_df["text"], train_df["label_int"], tokenizer)
    test_dataset = AlertDataset(test_df["text"], test_df["label_int"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             num_workers=2, pin_memory=True)

    # Optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP)

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"Steps per epoch: {len(train_loader)}")
    print("-" * 60)

    best_f1 = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, EPOCHS
        )
        epoch_time = time.time() - epoch_start

        # Evaluate after each epoch
        preds, labels, probs, eval_loss = evaluate(model, test_loader, device)
        report = classification_report(labels, preds, target_names=["Benign", "Threat"],
                                       output_dict=True)
        threat_f1 = report["Threat"]["f1-score"]
        accuracy = report["accuracy"]

        print(f"\n  Epoch {epoch+1}/{EPOCHS} complete ({epoch_time:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.1f}%")
        print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {100*accuracy:.1f}%")
        print(f"  Threat F1:  {threat_f1:.4f} | Benign F1: {report['Benign']['f1-score']:.4f}")
        print("-" * 60)

        # Save best model
        if threat_f1 > best_f1:
            best_f1 = threat_f1
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            print(f"  *** New best model saved (Threat F1: {best_f1:.4f}) ***")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Final Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (best model)")
    print(f"{'='*60}")

    # Reload best model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.to(device)

    preds, labels, probs, _ = evaluate(model, test_loader, device)

    # Classification report
    report_text = classification_report(labels, preds, target_names=["Benign", "Threat"])
    print(f"\n{report_text}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Benign  Threat")
    print(f"  Actual Benign  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual Threat  {cm[1][0]:>6}  {cm[1][1]:>6}")

    # Per-category evaluation
    print(f"\n{'='*60}")
    print("PER-CATEGORY EVALUATION")
    print(f"{'='*60}")

    test_df_eval = test_df.copy()
    test_df_eval["pred"] = preds
    test_df_eval["prob_threat"] = probs

    for cat in sorted(test_df_eval["source_category"].unique()):
        cat_df = test_df_eval[test_df_eval["source_category"] == cat]
        cat_preds = cat_df["pred"].values
        cat_labels = cat_df["label_int"].values
        if len(set(cat_labels)) > 1:
            cat_report = classification_report(
                cat_labels, cat_preds,
                target_names=["Benign", "Threat"],
                output_dict=True,
            )
            print(f"\n  {cat.upper()} ({len(cat_df)} samples)")
            print(f"    Accuracy:  {100*cat_report['accuracy']:.1f}%")
            print(f"    Threat F1: {cat_report['Threat']['f1-score']:.4f}")
            print(f"    Benign F1: {cat_report['Benign']['f1-score']:.4f}")
        else:
            acc = (cat_preds == cat_labels).mean()
            print(f"\n  {cat.upper()} ({len(cat_df)} samples) — single class only")
            print(f"    Accuracy: {100*acc:.1f}%")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "botsv3_classification_report.txt")
    with open(results_path, "w") as f:
        f.write("SecureBERT 2.0 Fine-Tuned on BOTS v3\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Best Threat F1: {best_f1:.4f}\n")
        f.write(f"Training time: {total_time:.0f}s\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"\nResults saved to {results_path}")

    # ── Comparison with UNSW-NB15 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON: BOTS v3 vs UNSW-NB15")
    print(f"{'='*60}")

    report_dict = classification_report(labels, preds, target_names=["Benign", "Threat"],
                                        output_dict=True)
    print(f"{'Metric':<20} {'UNSW-NB15':>12} {'BOTS v3':>12}")
    print(f"{'-'*44}")
    print(f"{'Accuracy':<20} {'90.0%':>12} {100*report_dict['accuracy']:>11.1f}%")
    print(f"{'Threat F1':<20} {'0.9300':>12} {report_dict['Threat']['f1-score']:>12.4f}")
    print(f"{'Threat Precision':<20} {'—':>12} {report_dict['Threat']['precision']:>12.4f}")
    print(f"{'Threat Recall':<20} {'—':>12} {report_dict['Threat']['recall']:>12.4f}")
    print(f"{'Benign F1':<20} {'—':>12} {report_dict['Benign']['f1-score']:>12.4f}")
    print(f"{'Data type':<20} {'Network flow':>12} {'SIEM alerts':>12}")
    print(f"{'Alert categories':<20} {'1':>12} {'4':>12}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()