"""
inference.py — SecureBERT 2.0 LoRA inference pipeline.

Classifies security alerts as threat or benign using the trained LoRA adapter.
Works with any CSV containing a 'text' column. Labels optional.

Usage:
    python inference.py                                    # default test set
    python inference.py --input data/alerts.csv            # custom input
    python inference.py --input data/alerts.csv --top 20   # show top 20 threats

Output: results/classified_alerts.json
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_ID = "cisco-ai/SecureBERT2.0-base"
ADAPTER_PATH = "models/secureBERT_botsv3_lora"
DEFAULT_INPUT = "data/botsv3_test.csv"
DEFAULT_OUTPUT = "results/classified_alerts.json"
BATCH_SIZE = 32
MAX_LEN = 256
CONFIDENCE_THRESHOLD = 0.5


# ── DATASET ───────────────────────────────────────────────────────────────────

class AlertDataset(Dataset):
    """Pre-tokenizes all texts once. Zero CPU overhead per batch."""

    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


# ── RULE-BASED SUMMARIES ─────────────────────────────────────────────────────

THREAT_NOTES = {
    "windows": {
        "cmd.exe|powershell|wmic": "Suspicious process execution. Review command line.",
        "blocked": "Connection blocked by WFP. Possible C2 attempt.",
        "service was installed": "New service installed. Check for persistence.",
        "special privileges": "Privilege escalation detected. Verify if expected.",
        "failed to log on": "Failed authentication. Check for brute force.",
    },
    "linux": {
        "invalid user": "SSH brute force — invalid username. Check source IP.",
        "failed password": "Failed SSH authentication. Correlate with other attempts.",
    },
    "network": {
        "dns": "Suspicious DNS activity. Check for C2 or exfiltration.",
    },
    "cloud": {
        "runinstances": "Unauthorized EC2 launch. Possible crypto mining.",
        "createuser|createaccesskey": "IAM modification. Possible backdoor creation.",
        "consolelogin": "Console login from external IP. Verify with owner.",
    },
}

BENIGN_NOTES = {
    "windows": "Normal Windows activity. No action required.",
    "linux": "Normal Linux system activity.",
    "network": "Normal network traffic.",
    "cloud": "Normal cloud API activity.",
}


def extract_key_fields(text, category):
    """Extract structured fields from alert text by category."""
    fields = {}
    lower = text.lower()

    if category == "windows":
        desc = text.split("|")[0].replace("[windows]", "").strip() if "|" in text else text[:100]
        fields["event"] = desc[:80]
        for label in ["Account Name:", "Process Name:", "Process Command Line:"]:
            if label.lower() in lower:
                start = lower.index(label.lower()) + len(label)
                fields[label.rstrip(":")] = text[start:start + 80].split("|")[0].strip()

    elif category == "linux":
        fields["raw_log"] = text.replace("[linux]", "").strip()[:200]

    elif category == "network":
        for part in text.replace("[network]", "").split("|"):
            part = part.strip()
            if "from " in part and " to " in part:
                fields["connection"] = part
            elif "bytes_in" in part:
                fields["traffic"] = part
            elif "query" in part.lower():
                fields["query"] = part[:100]

    elif category == "cloud":
        for part in text.replace("[cloud]", "").split("|"):
            part = part.strip()
            if "Cloud event" in part:
                fields["event"] = part
            elif "from " in part:
                fields["source_ip"] = part.replace("from ", "")

    return fields if fields else {"raw": text[:200]}


def generate_note(text, category, is_threat):
    """Match alert text against known patterns to generate analyst note."""
    lower = text.lower()

    if is_threat:
        patterns = THREAT_NOTES.get(category, {})
        for keywords, note in patterns.items():
            if any(kw in lower for kw in keywords.split("|")):
                return note
        return "Suspicious activity detected. Manual review recommended."

    return BENIGN_NOTES.get(category, "No threat indicators detected.")


def build_summary(text, category, prediction, confidence):
    """Combine key fields and analyst note into a rule-based summary."""
    is_threat = prediction == "threat"
    return {
        "key_fields": extract_key_fields(text, category),
        "analyst_note": generate_note(text, category, is_threat),
    }


# ── MODEL LOADING ────────────────────────────────────────────────────────────

def load_model(device):
    """Load base model + LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, problem_type="single_label_classification"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """Run batch inference. Returns predictions and probabilities."""
    all_preds, all_probs = [], []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits

            probs = torch.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def build_output(df, preds, probs, has_labels, input_path, inference_time):
    """Build structured JSON output from predictions."""
    alerts = []
    threat_count = 0

    for i, row in df.iterrows():
        pred = "threat" if preds[i] == 1 else "benign"
        conf = float(probs[i][preds[i]])
        threat_prob = float(probs[i][1])
        cat = row.get("source_category", "unknown")

        if pred == "threat":
            threat_count += 1

        alerts.append({
            "id": i + 1,
            "source_category": cat,
            "alert_text": row["text"],
            "prediction": pred,
            "confidence": round(conf, 4),
            "threat_probability": round(threat_prob, 4),
            "actual_label": row["label"] if has_labels else "unknown",
            "rule_summary": build_summary(row["text"], cat, pred, conf),
        })

    alerts.sort(key=lambda x: x["threat_probability"], reverse=True)

    # Accuracy only when labels exist
    accuracy = None
    if has_labels:
        correct = sum(
            1 for i, r in df.iterrows()
            if (preds[i] == 1) == (r["label"] == "threat")
        )
        accuracy = round(correct / len(df) * 100, 2)

    benign_count = len(alerts) - threat_count
    return {
        "metadata": {
            "model": "SecureBERT 2.0 + LoRA",
            "adapter": ADAPTER_PATH,
            "input_file": input_path,
            "total_alerts": len(alerts),
            "threats_found": threat_count,
            "benign_filtered": benign_count,
            "threat_rate": round(threat_count / len(alerts) * 100, 2),
            "accuracy": accuracy,
            "labeled": has_labels,
            "inference_time_seconds": round(inference_time, 2),
            "avg_time_per_alert_ms": round(inference_time / len(alerts) * 1000, 2),
            "generated_at": datetime.now().isoformat(),
        },
        "alerts": alerts,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SecureBERT 2.0 LoRA — Alert Classifier")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--top", type=int, default=10, help="Print top N threats")
    args = parser.parse_args()

    print("=" * 60)
    print("SecureBERT 2.0 LoRA — Alert Classifier")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    # Load data
    print(f"\nInput: {args.input}")
    df = pd.read_csv(args.input)
    has_labels = "label" in df.columns and not (df["label"] == "unknown").all()
    print(f"Alerts: {len(df)} | Labels: {'yes' if has_labels else 'no'}")

    if has_labels:
        t = (df["label"] == "threat").sum()
        b = (df["label"] == "benign").sum()
        print(f"  threat: {t} | benign: {b}")

    # Load model
    print(f"\nLoading model...")
    model, tokenizer = load_model(device)
    print("Model ready.")

    # Pre-tokenize
    print("Tokenizing...", end=" ", flush=True)
    t0 = time.time()
    dataset = AlertDataset(df["text"], tokenizer)
    print(f"done ({time.time() - t0:.1f}s)")

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Inference
    print(f"Classifying {len(df)} alerts...", end=" ", flush=True)
    t0 = time.time()
    preds, probs = run_inference(model, loader, device)
    inference_time = time.time() - t0
    print(f"done ({inference_time:.1f}s, {inference_time / len(df) * 1000:.1f}ms/alert)")

    # Build output
    output = build_output(df, preds, probs, has_labels, args.input, inference_time)
    meta = output["metadata"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total:    {meta['total_alerts']}")
    print(f"  Threats:  {meta['threats_found']} ({meta['threat_rate']}%)")
    print(f"  Benign:   {meta['benign_filtered']}")
    if meta["accuracy"] is not None:
        print(f"  Accuracy: {meta['accuracy']}%")
    print(f"  Speed:    {meta['avg_time_per_alert_ms']}ms/alert")
    print(f"  Output:   {args.output}")

    # Top threats
    threats = [a for a in output["alerts"] if a["prediction"] == "threat"]
    if threats:
        n = min(args.top, len(threats))
        print(f"\n  TOP {n} THREATS:")
        for a in threats[:n]:
            conf = f"{a['confidence'] * 100:.1f}%"
            cat = a["source_category"]
            note = a["rule_summary"]["analyst_note"]
            print(f"    [{cat:>8}] {conf:>6} — {note}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
