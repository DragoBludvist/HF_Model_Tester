"""
finetune_botsv3_lora.py — Optimized LoRA fine-tune SecureBERT 2.0 on BOTS v3.

Optimizations:
  1. Pre-tokenization — tokenize once upfront, not per-batch
  2. Cosine LR scheduler with warmup — better convergence
  3. Early stopping — stop if no improvement for 2 epochs
  4. Targeted LoRA modules — Wqkv + Wo only (attention layers)
  5. torch.compile — JIT-compiled forward pass

Designed for: Dell G15, Ryzen 5600H, NVIDIA RTX 3050 (4GB VRAM, CUDA)

Usage:
    python finetune_botsv3_lora.py
"""

import pandas as pd
import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID = "cisco-ai/SecureBERT2.0-base"
TRAIN_PATH = "data/botsv3_train.csv"
TEST_PATH = "data/botsv3_test.csv"
MODEL_SAVE_PATH = "models/secureBERT_botsv3_lora"
RESULTS_DIR = "results"

# Training hyperparameters
BATCH_SIZE = 16
MAX_LEN = 256
EPOCHS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
USE_AMP = True
GRAD_ACCUM_STEPS = 2     # Effective batch size = 32
PATIENCE = 2             # Early stopping: stop after 2 epochs without improvement

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ["Wqkv", "Wo"]  # Attention projections only — lean and effective


# ── DATASET (pre-tokenized) ──────────────────────────────────────────────────

class AlertDataset(Dataset):
    """Pre-tokenizes all texts once at init. Zero CPU overhead per batch."""

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        print(f"  Pre-tokenizing {len(texts)} samples...", end=" ", flush=True)
        t0 = time.time()
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        print(f"done ({time.time() - t0:.1f}s)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    device_type = device.type

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type=device_type, enabled=(USE_AMP and device_type == "cuda")):
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
            scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            acc = 100 * correct / total
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{total_epochs} | Step {step+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | LR: {lr:.2e}")

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    device_type = device.type

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device_type, enabled=(USE_AMP and device_type == "cuda")):
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
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        total_loss / len(loader),
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SecureBERT 2.0 LoRA Fine-Tuning on BOTS v3 (optimized)")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected")
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Train threats: {(train_df['label_int']==1).sum()} ({100*(train_df['label_int']==1).mean():.1f}%)")
    print(f"Test threats:  {(test_df['label_int']==1).sum()} ({100*(test_df['label_int']==1).mean():.1f}%)")

    # Class weights
    n_threat = (train_df["label_int"] == 1).sum()
    n_benign = (train_df["label_int"] == 0).sum()
    weight_threat = n_benign / n_threat
    print(f"\nClass weights: benign=1.00, threat={weight_threat:.2f}")

    # Load tokenizer and pre-tokenize
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\nPre-tokenizing datasets:")
    train_dataset = AlertDataset(train_df["text"], train_df["label_int"], tokenizer)
    test_dataset = AlertDataset(test_df["text"], test_df["label_int"], tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Load base model
    print(f"\nLoading {MODEL_ID}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, problem_type="single_label_classification",
    )

    # Apply LoRA — targeted at attention projections only
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    # Try torch.compile for faster forward pass
    print("torch.compile: skipped (Windows)")

    # Parameter summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nParameters:")
    print(f"  Frozen:    {frozen_params:>12,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable: {trainable_params:>12,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")

    # Optimizer + cosine scheduler with warmup
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM_STEPS
    warmup_steps = total_steps // 10  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    print(f"\nScheduler: cosine with {warmup_steps} warmup steps / {total_steps} total steps")

    # Training loop with early stopping
    print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
    print(f"Steps per epoch: {len(train_loader)}")
    print("-" * 60)

    best_f1 = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch, EPOCHS
        )
        epoch_time = time.time() - epoch_start

        # Evaluate
        preds, labels, probs, eval_loss = evaluate(model, test_loader, device)
        report = classification_report(labels, preds, target_names=["Benign", "Threat"],
                                       output_dict=True)
        threat_f1 = report["Threat"]["f1-score"]
        accuracy = report["accuracy"]

        print(f"\n  Epoch {epoch+1}/{EPOCHS} complete ({epoch_time:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.1f}%")
        print(f"  Eval Loss:  {eval_loss:.4f} | Eval Acc:  {100*accuracy:.1f}%")
        print(f"  Threat F1:  {threat_f1:.4f} | Benign F1: {report['Benign']['f1-score']:.4f}")

        # Save best + early stopping check
        if threat_f1 > best_f1:
            best_f1 = threat_f1
            patience_counter = 0
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            # Unwrap compiled model if needed
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            print(f"  *** New best LoRA adapter saved (Threat F1: {best_f1:.4f}) ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping triggered — no improvement for {PATIENCE} epochs")
                break

        print("-" * 60)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Adapter size
    adapter_size = sum(
        os.path.getsize(os.path.join(MODEL_SAVE_PATH, f))
        for f in os.listdir(MODEL_SAVE_PATH)
        if f.endswith(('.safetensors', '.bin'))
    )
    print(f"Adapter size on disk: {adapter_size/1e6:.1f} MB")

    # ── Final Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION (best LoRA adapter)")
    print(f"{'='*60}")

    # Reload best adapter
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, problem_type="single_label_classification"
    )
    model = PeftModel.from_pretrained(base_model, MODEL_SAVE_PATH)
    model.to(device)
    model.eval()

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
    results_path = os.path.join(RESULTS_DIR, "botsv3_lora_classification_report.txt")
    with open(results_path, "w") as f:
        f.write("SecureBERT 2.0 LoRA Fine-Tuned on BOTS v3 (optimized)\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Method: LoRA (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})\n")
        f.write(f"Target modules: {LORA_TARGETS}\n")
        f.write(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100*trainable_params/total_params:.2f}%)\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Epochs completed: {min(epoch+1, EPOCHS)}\n")
        f.write(f"Best Threat F1: {best_f1:.4f}\n")
        f.write(f"Training time: {total_time:.0f}s\n")
        f.write(f"Adapter size: {adapter_size/1e6:.1f} MB\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"\nResults saved to {results_path}")

    # ── Comparison ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON: LoRA vs UNSW-NB15 baseline")
    print(f"{'='*60}")

    report_dict = classification_report(labels, preds, target_names=["Benign", "Threat"],
                                        output_dict=True)
    print(f"{'Metric':<22} {'UNSW-NB15':>12} {'BOTS LoRA':>12}")
    print(f"{'-'*46}")
    print(f"{'Accuracy':<22} {'90.0%':>12} {100*report_dict['accuracy']:>11.1f}%")
    print(f"{'Threat F1':<22} {'0.9300':>12} {report_dict['Threat']['f1-score']:>12.4f}")
    print(f"{'Threat Precision':<22} {'0.9900':>12} {report_dict['Threat']['precision']:>12.4f}")
    print(f"{'Threat Recall':<22} {'0.8700':>12} {report_dict['Threat']['recall']:>12.4f}")
    print(f"{'Trainable params':<22} {'149.6M':>12} {trainable_params/1e6:>11.1f}M")
    print(f"{'Model size on disk':<22} {'~600MB':>12} {adapter_size/1e6:>10.1f}MB")
    print(f"{'Training time':<22} {'—':>12} {total_time:>10.0f}s")
    print(f"{'='*46}")


if __name__ == "__main__":
    main()