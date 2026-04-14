"""
inference_botsv3.py — Run the trained SecureBERT LoRA model on test data
and output classified results as JSON for the analyst dashboard.

Usage:
    python inference_botsv3.py

Input:  data/botsv3_test.csv + models/secureBERT_botsv3_lora/
Output: results/classified_alerts.json
"""

import pandas as pd
import torch
import numpy as np
import json
import os
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.amp import autocast
from peft import PeftModel
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID = "cisco-ai/SecureBERT2.0-base"
ADAPTER_PATH = "models/secureBERT_botsv3_lora"
TEST_PATH = "data/botsv3_test.csv"
OUTPUT_PATH = "results/classified_alerts.json"
BATCH_SIZE = 32
MAX_LEN = 256


class AlertDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


def extract_rule_summary(alert_text, source_category, prediction, confidence):
    """Rule-based summary — extracts key fields and generates a short analyst note."""
    summary = {"key_fields": {}, "analyst_note": ""}

    text_lower = alert_text.lower()

    if source_category == "windows":
        # Extract event description (first line before the pipe separator)
        if "|" in alert_text:
            desc = alert_text.split("|")[0].replace("[windows]", "").strip()
        else:
            desc = alert_text[:100].replace("[windows]", "").strip()
        summary["key_fields"]["event"] = desc

        # Extract key identifiers
        for field in ["Security ID:", "Account Name:", "Process Name:", "Process Command Line:"]:
            if field.lower() in text_lower:
                start = text_lower.index(field.lower()) + len(field)
                value = alert_text[start:start+80].split("|")[0].strip()
                summary["key_fields"][field.replace(":", "").strip()] = value

        if prediction == "threat":
            if "cmd.exe" in text_lower or "powershell" in text_lower or "wmic" in text_lower:
                summary["analyst_note"] = "Suspicious process execution detected. Review command line for malicious intent."
            elif "blocked" in text_lower:
                summary["analyst_note"] = "Connection blocked by Windows Filtering Platform. Possible C2 communication attempt."
            elif "service was installed" in text_lower:
                summary["analyst_note"] = "New service installed. Check for persistence mechanism."
            elif "special privileges" in text_lower:
                summary["analyst_note"] = "Privilege escalation detected. Verify if expected for this account."
            elif "failed to log on" in text_lower:
                summary["analyst_note"] = "Failed authentication attempt. Check for brute force pattern."
            else:
                summary["analyst_note"] = "Suspicious Windows event. Manual review recommended."
        else:
            summary["analyst_note"] = "Normal Windows activity. No action required."

    elif source_category == "linux":
        summary["key_fields"]["raw_log"] = alert_text.replace("[linux]", "").strip()[:200]

        if prediction == "threat":
            if "invalid user" in text_lower:
                summary["analyst_note"] = "SSH brute force attempt — invalid username tried. Check source IP for known scanners."
            elif "failed password" in text_lower:
                summary["analyst_note"] = "Failed SSH authentication. Correlate with other failed attempts from same source."
            else:
                summary["analyst_note"] = "Suspicious Linux event. Review source IP and activity pattern."
        else:
            summary["analyst_note"] = "Normal Linux system activity."

    elif source_category == "network":
        # Extract IPs and ports from synthesized text
        parts = alert_text.replace("[network]", "").split("|")
        for part in parts:
            part = part.strip()
            if "from " in part and " to " in part:
                summary["key_fields"]["connection"] = part
            elif "bytes_in" in part:
                summary["key_fields"]["traffic"] = part

        if prediction == "threat":
            summary["analyst_note"] = "Network connection involving known attacker IP. Investigate lateral movement or C2."
        else:
            summary["analyst_note"] = "Normal network traffic."

    elif source_category == "cloud":
        parts = alert_text.replace("[cloud]", "").split("|")
        for part in parts:
            part = part.strip()
            if "Cloud event" in part:
                summary["key_fields"]["event"] = part
            elif "from " in part:
                summary["key_fields"]["source_ip"] = part.replace("from ", "")

        if prediction == "threat":
            if "runinstances" in text_lower:
                summary["analyst_note"] = "Unauthorized EC2 instance launch. Possible crypto mining. Verify IAM permissions."
            elif "createuser" in text_lower or "createaccesskey" in text_lower:
                summary["analyst_note"] = "IAM modification detected. Possible backdoor creation. Review CloudTrail immediately."
            elif "consolelogin" in text_lower:
                summary["analyst_note"] = "Console login from external IP. Verify with account owner."
            else:
                summary["analyst_note"] = "Suspicious cloud API activity. Review source IP and action."
        else:
            summary["analyst_note"] = "Normal cloud API activity."

    return summary


def main():
    print("=" * 60)
    print("SecureBERT 2.0 LoRA — Inference Pipeline")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Load data
    print("\nLoading test data...")
    df = pd.read_csv(TEST_PATH)
    print(f"Loaded {len(df)} alerts")

    # Load model
    print(f"\nLoading model from {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, problem_type="single_label_classification"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Pre-tokenize
    print("Pre-tokenizing...", end=" ", flush=True)
    dataset = AlertDataset(df["text"], tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    print("done.")

    # Inference
    print(f"\nClassifying {len(df)} alerts...")
    all_preds = []
    all_probs = []
    device_type = device.type
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(device_type=device_type, enabled=(device_type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    inference_time = time.time() - start_time
    avg_time_ms = (inference_time / len(df)) * 1000

    print(f"Inference complete: {inference_time:.1f}s total, {avg_time_ms:.1f}ms per alert")

    # Build output
    print("\nGenerating classified output...")
    alerts = []
    threat_count = 0
    benign_count = 0

    for i, row in df.iterrows():
        pred_label = "threat" if all_preds[i] == 1 else "benign"
        confidence = float(all_probs[i][all_preds[i]])
        threat_prob = float(all_probs[i][1])

        if pred_label == "threat":
            threat_count += 1
        else:
            benign_count += 1

        # Rule-based summary
        rule_summary = extract_rule_summary(
            row["text"], row["source_category"], pred_label, confidence
        )

        alert_entry = {
            "id": i + 1,
            "timestamp": row.get("timestamp", "N/A") if "timestamp" in row else "N/A",
            "host": row.get("host", "N/A") if "host" in row else "N/A",
            "sourcetype": row.get("sourcetype", "N/A") if "sourcetype" in row else "N/A",
            "source_category": row["source_category"],
            "alert_text": row["text"],
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "threat_probability": round(threat_prob, 4),
            "actual_label": row.get("label", "unknown"),
            "rule_summary": rule_summary,
        }
        alerts.append(alert_entry)

    # Sort by threat probability (highest first)
    alerts.sort(key=lambda x: x["threat_probability"], reverse=True)

    output = {
        "metadata": {
            "model": "SecureBERT 2.0 + LoRA",
            "adapter": ADAPTER_PATH,
            "total_alerts": len(alerts),
            "threats_found": threat_count,
            "benign_filtered": benign_count,
            "accuracy": round((df["label_int"].values == np.array(all_preds)).mean() * 100, 2),
            "inference_time_seconds": round(inference_time, 2),
            "avg_time_per_alert_ms": round(avg_time_ms, 2),
            "generated_at": datetime.now().isoformat(),
        },
        "alerts": alerts,
    }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {OUTPUT_PATH}")
    print(f"  Total alerts:    {len(alerts)}")
    print(f"  Threats found:   {threat_count}")
    print(f"  Benign filtered: {benign_count}")
    print(f"  Accuracy:        {output['metadata']['accuracy']}%")
    print(f"  Avg inference:   {avg_time_ms:.1f}ms per alert")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()