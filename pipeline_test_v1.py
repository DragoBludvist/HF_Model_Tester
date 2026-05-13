"""
pipeline_test.py — End-to-end pipeline test using simulated Wazuh JSON alerts.

Demonstrates the full inference pipeline:
    Wazuh JSON  →  standardize_v2.handle_wazuh()  →  tokenizer  →  model  →  prediction

For each test alert, prints:
  1. The original Wazuh JSON (compact)
  2. The canonical text emitted by standardize_v2
  3. The model's classification + confidence
  4. Whether the verdict matches the expected label

Each test scenario represents a category-label combination from production:
  - threat / identity / auth_failure (SSH brute force)
  - threat / network / connection_denied (ASA inbound denied)
  - threat / endpoint / process_created (suspicious cmd.exe spawn)
  - threat / cloud / compute_instance_launched (RunInstances from attacker IP)
  - benign / dns / dns_query (normal DNS lookup)
  - benign / identity / auth_success (legitimate login)
  - etc.

Usage:
    cd C:\\Users\\DELL\\HF_Model_Tester
    python pipeline_test.py
"""

import json
import sys
import os

import numpy as np
import torch
from peft import PeftModel
from torch.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# standardize_v2 must be importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import standardize_v2

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_ID = "cisco-ai/SecureBERT2.0-base"
ADAPTER_PATH = "models/secureBERT_botsv3_lora"
MAX_LEN = 256


# ─── TEST ALERTS — realistic Wazuh JSON with known expected labels ───────────

TEST_ALERTS = [
    # ── THREATS ───────────────────────────────────────────────────────────────
    {
        "scenario": "SSH brute force from known attacker IP",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["sshd", "authentication_failed"],
                "level": 5,
                "description": "PAM: User login failed — brute force attempt",
            },
            "agent": {"name": "ssh-edge-01"},
            "data": {"srcip": "157.97.121.132", "srcuser": "root"},
        },
    },
    {
        "scenario": "SSH invalid user attempt",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["sshd", "invalid_login"],
                "level": 5,
                "description": "SSH invalid user attempt",
            },
            "agent": {"name": "ssh-edge-01"},
            "data": {"srcip": "221.194.47.205", "srcuser": "admin"},
        },
    },
    {
        "scenario": "Cisco ASA inbound TCP connection denied from attacker IP",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["firewall", "cisco-asa"],
                "level": 6,
                "description": "Cisco ASA: connection denied by ACL",
            },
            "agent": {"name": "FROTHLY-FW1"},
            "data": {
                "srcip": "80.211.181.211", "srcport": "44521",
                "dstip": "10.0.0.50",     "dstport": "3389",
                "protocol": "tcp",
            },
        },
    },
    {
        "scenario": "Suspicious DNS activity (training gap — expected to fail)",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["ids", "attacks"],
                "level": 10,
                "description": "Suspicious DNS activity detected — possible C2 beacon",
            },
            "agent": {"name": "dns-server-01"},
            "data": {},
        },
    },
    {
        "scenario": "Windows Filtering Platform blocked outbound connection",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["windows", "windows_security"],
                "level": 6,
                "description": "Windows Filtering Platform has blocked a connection",
            },
            "agent": {"name": "BSTOLL-L"},
            "data": {"process": "C:\\Windows\\System32\\cmd.exe"},
        },
    },
    {
        "scenario": "Sysmon process_created — cmd.exe spawned from powershell on compromised host",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["sysmon", "sysmon_event1"],
                "level": 7,
                "description": "Sysmon Event 1: Process Created",
            },
            "agent": {"name": "BSTOLL-L"},
            "data": {
                "process": "C:\\Windows\\System32\\cmd.exe",
                "parent_process": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "user": "NT AUTHORITY\\SYSTEM",
            },
        },
    },
    {
        "scenario": "CloudTrail RunInstances from known attacker IP",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["amazon", "aws", "cloudtrail"],
                "level": 8,
                "description": "AWS RunInstances API call",
            },
            "agent": {"name": "splunk.froth.ly"},
            "data": {
                "eventName": "RunInstances",
                "srcip": "139.198.18.205",
            },
        },
    },
    {
        "scenario": "CloudTrail CreateAccessKey from attacker IP (IAM persistence)",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["amazon", "aws"],
                "level": 8,
                "description": "AWS CreateAccessKey API call",
            },
            "agent": {"name": "iam-monitor"},
            "data": {
                "eventName": "CreateAccessKey",
                "srcip": "35.153.154.221",
            },
        },
    },
    {
        "scenario": "Malware detected by AV (no training data — expected weak)",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["vipre", "virus"],
                "level": 12,
                "description": "Malware detected: Trojan.Win32.Generic in user download",
            },
            "agent": {"name": "WKS-042"},
            "data": {},
        },
    },
    {
        "scenario": "FIM — critical system file modified (no training data — expected weak)",
        "expected": "threat",
        "alert": {
            "rule": {
                "groups": ["syscheck", "syscheck_entry_modified"],
                "level": 8,
                "description": "Integrity checksum changed for /etc/shadow",
            },
            "agent": {"name": "linux-host-01"},
            "data": {},
        },
    },

    # ── BENIGNS ──────────────────────────────────────────────────────────────
    {
        "scenario": "Normal SSH login from internal IP",
        "expected": "benign",
        "alert": {
            "rule": {
                "groups": ["sshd", "authentication_success"],
                "level": 3,
                "description": "PAM: Login session opened successfully",
            },
            "agent": {"name": "ssh-edge-01"},
            "data": {"srcip": "10.0.1.42", "srcuser": "alice"},
        },
    },
    {
        "scenario": "Normal DNS A-record query (legitimate domain)",
        "expected": "benign",
        "alert": {
            "rule": {
                "groups": ["syslog"],
                "level": 3,
                "description": "DNS query",
            },
            "agent": {"name": "dns-server-01"},
            "data": {
                "query": "splunk.froth.ly",
                "query_type": "A",
                "reply_code": "NoError",
            },
        },
    },
    {
        "scenario": "AWS API call from service principal (legitimate automation)",
        "expected": "benign",
        "alert": {
            "rule": {
                "groups": ["amazon", "aws", "cloudtrail"],
                "level": 4,
                "description": "AWS API call",
            },
            "agent": {"name": "cloudtrail-collector"},
            "data": {
                "eventName": "DescribeInstances",
                "srcip": "ec2.amazonaws.com",
            },
        },
    },
    {
        "scenario": "Routine Windows network event (EventCode 5156)",
        "expected": "benign",
        "alert": {
            "rule": {
                "groups": ["windows", "win_evt_channel"],
                "level": 3,
                "description": "Windows Filtering Platform has permitted a connection",
            },
            "agent": {"name": "WKS-042"},
            "data": {},
        },
    },
    {
        "scenario": "Benign NXDomain reverse lookup (expired IP)",
        "expected": "benign",
        "alert": {
            "rule": {
                "groups": ["syslog"],
                "level": 3,
                "description": "DNS NXDomain response",
            },
            "agent": {"name": "dns-server-01"},
            "data": {
                "query": "22.78.0.192.in-addr.arpa",
                "reply_code": "NXDomain",
            },
        },
    },
]


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def load_model():
    """Load base SecureBERT 2.0 + the trained LoRA adapter."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("Running on CPU (slower; install CUDA if available)")

    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading base model {MODEL_ID}...")
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2,
        problem_type="single_label_classification",
    )
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to(device).eval()
    return tokenizer, model, device


def classify(tokenizer, model, device, text):
    """Run a single canonical text through the model. Returns (pred, conf, threat_prob)."""
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt",
    ).to(device, non_blocking=True)

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return (
        "threat" if pred_idx == 1 else "benign",
        float(probs[pred_idx]),
        float(probs[1]),
    )


# ─── PIPELINE RUNNER ─────────────────────────────────────────────────────────

def compact_json(d):
    """Format a Wazuh alert compactly for display."""
    return json.dumps(d, separators=(",", ":"))


def main():
    print("=" * 80)
    print("END-TO-END PIPELINE TEST")
    print("Wazuh JSON  →  standardize_v2  →  tokenizer  →  model  →  prediction")
    print("=" * 80)
    print()

    tokenizer, model, device = load_model()

    correct = 0
    by_outcome = {"correct_threat": 0, "correct_benign": 0,
                  "missed_threat":  0, "false_alarm":   0,
                  "standardize_failed": 0}
    results = []

    print()
    print("=" * 80)
    print(f"RUNNING {len(TEST_ALERTS)} SCENARIOS")
    print("=" * 80)

    for i, item in enumerate(TEST_ALERTS, 1):
        expected = item["expected"]
        print(f"\n[{i:2d}/{len(TEST_ALERTS)}] {item['scenario']}")
        print(f"     expected: {expected}")
        print(f"     wazuh:    {compact_json(item['alert'])[:140]}")

        # STAGE 1: standardize
        text, category, severity = standardize_v2.handle_wazuh(item["alert"])
        if text is None:
            print(f"     standardize: REJECTED (this should not happen for valid input)")
            by_outcome["standardize_failed"] += 1
            results.append({**item, "canonical": None, "prediction": None})
            continue
        print(f"     canonical: {text}")
        print(f"     category:  {category} | severity: {severity}")

        # STAGE 2 & 3: tokenize + classify
        pred, conf, threat_prob = classify(tokenizer, model, device, text)
        match = (pred == expected)
        marker = "✓" if match else "✗"
        print(f"     model:     {pred} (conf={conf:.4f}, threat_prob={threat_prob:.4f}) {marker}")

        if match:
            correct += 1
            if expected == "threat":
                by_outcome["correct_threat"] += 1
            else:
                by_outcome["correct_benign"] += 1
        else:
            if expected == "threat":
                by_outcome["missed_threat"] += 1
            else:
                by_outcome["false_alarm"] += 1

        results.append({
            **item, "canonical": text, "prediction": pred,
            "confidence": conf, "threat_probability": threat_prob,
            "match": match,
        })

    # ─── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("PIPELINE TEST RESULTS")
    print("=" * 80)
    print(f"\nOverall: {correct}/{len(TEST_ALERTS)} correct ({100*correct/len(TEST_ALERTS):.1f}%)")
    print(f"\nBreakdown:")
    print(f"  Correctly identified threats:   {by_outcome['correct_threat']}")
    print(f"  Correctly identified benigns:   {by_outcome['correct_benign']}")
    print(f"  Missed threats (false negative): {by_outcome['missed_threat']}")
    print(f"  False alarms (false positive):   {by_outcome['false_alarm']}")
    if by_outcome["standardize_failed"]:
        print(f"  Standardize failures: {by_outcome['standardize_failed']}")

    # Per-category summary
    print(f"\nPer-category breakdown:")
    by_cat = {}
    for r in results:
        if r.get("canonical") is None: continue
        cat = r["canonical"].split("]")[0].strip("[")
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r.get("match"):
            by_cat[cat]["correct"] += 1
    print(f"  {'category':>12} {'correct':>10} {'total':>8} {'%':>6}")
    for cat in sorted(by_cat):
        s = by_cat[cat]
        pct = 100 * s["correct"] / s["total"]
        print(f"  {cat:>12} {s['correct']:>10} {s['total']:>8} {pct:>5.0f}%")

    # Save full results to a JSON file for review
    out_path = "results/pipeline_test_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {out_path}")

    # Highlight the known training gaps
    print("\n" + "─" * 80)
    print("Known gaps in current training data (expected to fail):")
    print("─" * 80)
    gaps = [r for r in results
            if r.get("prediction") and not r.get("match")
            and any(x in r["scenario"].lower() for x in ("dns", "malware", "fim"))]
    if gaps:
        for r in gaps:
            print(f"  • {r['scenario']}")
            print(f"    canonical: {r['canonical']}")
            print(f"    predicted: {r['prediction']} (threat_prob={r['threat_probability']:.4f})")
    else:
        print("  (none of the gap scenarios failed — model may be generalizing better than expected)")


if __name__ == "__main__":
    main()
