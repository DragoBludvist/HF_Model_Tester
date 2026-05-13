"""
combine_botsv3_v3.py — Threat-aware BOTS v3 combiner using standardize_v2.

Produces a labeled training CSV in canonical v2 format. Replaces the inline
build_alert_text_*() builders from combine_botsv3_v2.py with calls to
standardize_v2.route_row(), completing the format adapter pattern that
preprocess_pipeline.py originally designed (the adapt_wazuh() placeholder
is now subsumed by standardize_v2).

Threat-aware sampling logic is preserved verbatim from combine_botsv3_v2.py
— it's load-bearing and corrects the earlier random-sampling bug where rare
attack events (SSH brute force, etc.) were missed when sampling 5,000 rows
out of 284,538.

Three meaningful changes vs. v2:
  1. Text generation goes through standardize_v2.route_row() — same
     canonical schema as production inference uses.
  2. is_threat_network() extended to extract IPs from %ASA syslog _raw
     fields. The v2 version only checked column-extracted src_ip/dest_ip,
     which exist for stream:* events but not for cisco:asa.
  3. Folds in new_atk_data.csv (200 ASA threats, labeled-by-filename) and
     new_begnin_data.csv (300 mixed benigns) as supplementary training.
     These give the model its first labeled cisco:asa exposure.

Output schema:
    text, label_int, label, source_category

Where source_category is the canonical v2 category (endpoint, identity,
network, dns, cloud, web, fim, antivirus, system) — not the four-bucket
windows/linux/network/cloud taxonomy from v2.

KNOWN LIMITATION: Endpoint will be unrepresented until the Splunk re-export
with _raw lands. The current Endpoint export is hollow (Sysmon fields empty).

Usage:
    python combine_botsv3_v3.py --input-dir /path/to/source/csvs \\
                                --output data/botsv3_train_v3.csv
"""

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter

# Allow standardize_v2 to be importable regardless of how this is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from standardize_v2 import route_row

csv.field_size_limit(10 * 1024 * 1024)
random.seed(42)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Source-bucket sample targets. These cap how many rows we pull from each
# BOTS v3 source file. The output category will typically be more granular
# (one bucket → multiple v2 categories), so the real distribution shifts.
TARGETS = {
    "windows": 10000,
    "linux":   5000,
    "network": 5000,
    "cloud":   5000,
}

SOURCE_FILES = {
    "windows":  "BotsV3_Windows.csv",
    "linux":    "BotsV3_LinuxFixed.csv",
    "network":  "BotsV3_Network.csv",
    "cloud":    "BotsV3_Cloud.csv",
}

# Supplementary labeled-by-filename data (already has _raw, runs through v2).
SUPPLEMENTARY = [
    ("new_atk_data.csv",     "threat"),
    ("new_begnin_data.csv",  "benign"),
]


# ════════════════════════════════════════════════════════════════════════════
# THREAT INDICATORS — preserved verbatim from combine_botsv3_v2.py
# ════════════════════════════════════════════════════════════════════════════

ATTACKER_IPS = {
    "139.198.18.205", "35.153.154.221", "157.97.121.132", "82.102.18.111",
    "209.107.196.112", "5.101.40.81", "167.114.13.150", "138.122.255.144",
    "122.226.181.165", "122.226.181.167", "221.194.47.205", "221.194.47.236",
    "221.194.47.233", "121.18.238.115", "121.18.238.123", "139.59.59.111",
    "50.115.191.161", "87.251.221.188", "80.211.181.211", "158.69.159.234",
    "203.141.134.128", "118.163.24.179", "217.61.6.175", "59.48.228.162",
    "200.41.190.179", "103.79.143.113", "45.119.82.125", "182.61.44.11",
    "182.100.67.105", "95.179.156.192", "115.238.245.2", "40.67.197.92",
    "103.233.123.104", "138.204.135.210", "189.27.101.5", "27.116.127.68",
    "34.212.137.11",
}

THREAT_CLOUD_APIS = {
    "RunInstances", "CreateUser", "CreateAccessKey", "DeleteAccessKey",
    "AuthorizeSecurityGroupIngress", "CreateKeyPair", "DeleteTrail",
    "StopLogging", "ConsoleLogin", "PutBucketPolicy",
}

# CloudTrail "service principal" sourceIPs that aren't real attackers.
AWS_SERVICE_PRINCIPALS = {
    "autoscaling.amazonaws.com", "config.amazonaws.com",
    "ec2.amazonaws.com", "lambda.amazonaws.com",
    "events.amazonaws.com", "inspector.amazonaws.com",
    "guardduty.amazonaws.com", "vpc-flow-logs.amazonaws.com",
    "AWS Internal",
}

COMPROMISED_HOST = "BSTOLL-L"

SUSPICIOUS_PROCS = [
    "cmd.exe", "powershell", "wscript", "cscript", "mshta",
    "certutil", "bitsadmin", "regsvr32", "rundll32", "net.exe",
    "net1.exe", "whoami", "ipconfig", "systeminfo", "tasklist",
    "sc.exe", "schtasks", "wmic", "msbuild", "psexec",
]

# Pre-compile the IP regex used by the ASA extension.
_IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")


# ════════════════════════════════════════════════════════════════════════════
# THREAT HEURISTICS
# ════════════════════════════════════════════════════════════════════════════

def has_attacker_ip(text):
    """Substring-match against the attacker IP list. Cheap and works on _raw."""
    if not text:
        return False
    return any(ip in text for ip in ATTACKER_IPS)


def is_threat_linux(raw):
    """Verbatim from combine_botsv3_v2.py."""
    if not raw:
        return False
    lower = raw.lower()
    if "invalid user" in lower:          return True
    if "failed password" in lower:       return True
    if "authentication failure" in lower: return True
    if has_attacker_ip(raw):              return True
    # Suspicious sudo'd commands
    if "command" in lower and "sudo" in lower:
        for sig in ("chmod 777", "wget", "curl http", "nc ", "ncat",
                    "/tmp/", "base64", "python -c", "perl -e"):
            if sig in lower:
                return True
    return False


def is_threat_network(row):
    """v2 logic + ASA-aware extension.

    Original v2 only looked at row['src_ip'] / row['dest_ip'], which are
    populated for stream:* events but NOT for cisco:asa (whose IPs live
    inside _raw). We now also regex-extract IPs from _raw on ASA rows
    and match against the attacker list.
    """
    src = (row.get("src_ip") or "").strip()
    dst = (row.get("dest_ip") or "").strip()
    if src in ATTACKER_IPS or dst in ATTACKER_IPS:
        return True

    raw = row.get("_raw") or ""
    stype = row.get("sourcetype") or ""
    if "cisco:asa" in stype or "%ASA-" in raw:
        for m in _IP_RE.finditer(raw):
            if m.group(1) in ATTACKER_IPS:
                return True

    return False


def is_threat_cloud(row):
    """Verbatim from combine_botsv3_v2.py."""
    ip = (row.get("sourceIPAddress") or "").strip()
    event = (row.get("eventName") or "").strip()
    if ip in ATTACKER_IPS:
        return True
    if event in THREAT_CLOUD_APIS and ip not in AWS_SERVICE_PRINCIPALS:
        return True
    return False


def is_threat_windows(row):
    """Verbatim from combine_botsv3_v2.py."""
    msg = (row.get("Message") or "").lower()
    host = (row.get("host") or "").strip()

    if "account failed to log on" in msg:
        return True
    for phrase in ("user account was created", "user account was enabled",
                   "password reset", "member was added to a security",
                   "member was added to a local", "user account was changed"):
        if phrase in msg:
            return True

    if host == COMPROMISED_HOST:
        for proc in SUSPICIOUS_PROCS:
            if proc in msg:
                return True
        if "windows filtering platform has blocked" in msg: return True
        if "service was installed" in msg:                  return True
        if "special privileges assigned" in msg:            return True

    return False


# Bucket → predicate mapping. Each predicate takes a row dict.
THREAT_PREDICATES = {
    "windows": is_threat_windows,
    "linux":   lambda r: is_threat_linux(r.get("_raw") or ""),
    "network": is_threat_network,
    "cloud":   is_threat_cloud,
}


# ════════════════════════════════════════════════════════════════════════════
# CORE LOADER — threat-aware sampling + v2 standardization
# ════════════════════════════════════════════════════════════════════════════

def standardize_or_skip(row):
    """Run row through standardize_v2; return (text, category) or None."""
    text = route_row(row)
    if text is None:
        return None
    if "]" not in text:
        return None
    category = text.split("]", 1)[0].strip("[")
    return text, category


def load_bucket(bucket_name, source_path, target_count, predicate):
    """
    Read a BOTS v3 source CSV. Classify every row as threat or benign via
    `predicate`. Take ALL threats first, fill remaining quota with random
    benigns, then standardize each kept row through standardize_v2.

    Returns: list of output dicts, plus stats.
    """
    if not os.path.exists(source_path):
        print(f"[{bucket_name}] Source missing: {source_path} — skipping bucket")
        return [], {"missing": True}

    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    threats = [r for r in rows if predicate(r)]
    benigns = [r for r in rows if not predicate(r)]

    print(f"[{bucket_name}] {len(rows):,} total | "
          f"{len(threats):,} threats | {len(benigns):,} benigns")

    # Take all threats; fill remaining quota randomly with benigns
    quota_for_benign = max(0, target_count - len(threats))
    random.shuffle(benigns)
    selected_benigns = benigns[:quota_for_benign]

    out = []
    threat_kept = threat_dropped = 0
    benign_kept = benign_dropped = 0

    for r in threats:
        result = standardize_or_skip(r)
        if result is None:
            threat_dropped += 1; continue
        text, cat = result
        out.append({"text": text, "label_int": 1, "label": "threat",
                    "source_category": cat})
        threat_kept += 1

    for r in selected_benigns:
        result = standardize_or_skip(r)
        if result is None:
            benign_dropped += 1; continue
        text, cat = result
        out.append({"text": text, "label_int": 0, "label": "benign",
                    "source_category": cat})
        benign_kept += 1

    print(f"[{bucket_name}] Standardized: {threat_kept:,} threat "
          f"(+{threat_dropped} dropped) | {benign_kept:,} benign "
          f"(+{benign_dropped} dropped)")

    return out, {"threat": threat_kept, "benign": benign_kept,
                 "threat_dropped": threat_dropped,
                 "benign_dropped": benign_dropped, "missing": False}


def load_supplementary(input_dir, filename, label):
    """
    Load a supplementary file where the label is fixed by filename
    (new_atk_data → threat, new_begnin_data → benign). Standardize each row.
    """
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        print(f"[supp:{label}] Missing: {path}")
        return []
    out = []
    dropped = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            result = standardize_or_skip(r)
            if result is None:
                dropped += 1; continue
            text, cat = result
            out.append({
                "text": text,
                "label_int": 1 if label == "threat" else 0,
                "label": label,
                "source_category": cat,
            })
    print(f"[supp:{label}] {len(out):,} added from {filename} "
          f"(+{dropped} dropped)")
    return out


# ════════════════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════════════════

def report(all_rows):
    """Print per-v2-category × label distribution. Useful for sanity-checking
    that DNS/cisco:asa events landed where we expected."""
    n = len(all_rows)
    n_t = sum(1 for r in all_rows if r["label"] == "threat")
    n_b = n - n_t
    by = Counter((r["source_category"], r["label"]) for r in all_rows)
    cats = sorted({c for c, _ in by})

    print()
    print("=" * 64)
    print(f"FINAL: {n:,} rows  |  {n_t:,} threat ({100*n_t/n:.1f}%)  "
          f"|  {n_b:,} benign ({100*n_b/n:.1f}%)")
    print()
    print(f"  {'category':>12} {'threat':>10} {'benign':>10} {'total':>10}")
    print("  " + "-" * 44)
    for c in cats:
        t = by.get((c, "threat"), 0)
        b = by.get((c, "benign"), 0)
        print(f"  {c:>12} {t:>10,} {b:>10,} {t+b:>10,}")
    print("=" * 64)
    print("\nNOTE: 'endpoint' will be missing or near-zero — Splunk re-export")
    print("      with _raw is still pending. This is expected.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Directory containing BotsV3_*.csv source files")
    p.add_argument("--output", default="data/botsv3_train_v3.csv",
                   help="Output CSV path")
    p.add_argument("--skip-supplementary", action="store_true",
                   help="Don't fold in new_atk_data/new_begnin_data")
    args = p.parse_args()

    print("=" * 64)
    print("BOTS v3 Combiner v3 — standardize_v2 + threat-aware sampling")
    print("=" * 64)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output}\n")

    all_rows = []

    for bucket, target in TARGETS.items():
        path = os.path.join(args.input_dir, SOURCE_FILES[bucket])
        rows, _ = load_bucket(bucket, path, target, THREAT_PREDICATES[bucket])
        all_rows.extend(rows)
        print()

    if not args.skip_supplementary:
        for filename, label in SUPPLEMENTARY:
            all_rows.extend(load_supplementary(args.input_dir, filename, label))
        print()

    if not all_rows:
        print("ERROR: no rows produced. Check --input-dir.")
        return 1

    random.shuffle(all_rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label_int",
                                           "label", "source_category"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    report(all_rows)
    print(f"\nWritten: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
