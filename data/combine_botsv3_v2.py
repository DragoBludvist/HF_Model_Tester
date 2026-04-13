"""
combine_botsv3_v2.py — Combine & balance BOTS v3 exports with threat-aware sampling.

Strategy: For each source category, pull ALL rows matching known attack indicators first,
then fill the remaining quota with randomly sampled benign rows. This ensures no attack
events are lost to random sampling.

Excludes: IAM, Endpoint (needs re-export with _raw)
"""

import csv
import random
import os

random.seed(42)

INPUT_DIR = "/mnt/user-data/uploads"
OUTPUT_PATH = "/mnt/user-data/outputs/botsv3_combined.csv"

HEADERS = ["timestamp", "host", "sourcetype", "source_category", "alert_text", "label"]

TARGETS = {
    "windows": 10000,
    "linux": 5000,
    "network": 5000,
    "cloud": 5000,
}

# ── KNOWN ATTACK INDICATORS ──────────────────────────────────────────────────

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

COMPROMISED_HOST = "BSTOLL-L"

SUSPICIOUS_PROCS = [
    "cmd.exe", "powershell", "wscript", "cscript", "mshta",
    "certutil", "bitsadmin", "regsvr32", "rundll32", "net.exe",
    "net1.exe", "whoami", "ipconfig", "systeminfo", "tasklist",
    "sc.exe", "schtasks", "wmic", "msbuild", "psexec",
]

stats = {}


def has_attacker_ip(text):
    """Check if any known attacker IP appears in the text."""
    for ip in ATTACKER_IPS:
        if ip in text:
            return True
    return False


def is_threat_linux(raw):
    """Check if a Linux raw log is a threat event."""
    lower = raw.lower()
    if "invalid user" in lower:
        return True
    if "failed password" in lower:
        return True
    if "authentication failure" in lower:
        return True
    if has_attacker_ip(raw):
        return True
    # Suspicious commands
    if "command" in lower and "sudo" in lower:
        for sig in ["chmod 777", "wget", "curl http", "nc ", "ncat",
                     "/tmp/", "base64", "python -c", "perl -e"]:
            if sig in lower:
                return True
    return False


def is_threat_network(row):
    """Check if a network event is a threat."""
    src = row.get("src_ip", "").strip()
    dest = row.get("dest_ip", "").strip()
    if src in ATTACKER_IPS or dest in ATTACKER_IPS:
        return True
    return False


def is_threat_cloud(row):
    """Check if a cloud event is a threat."""
    ip = row.get("sourceIPAddress", "").strip()
    event = row.get("eventName", "").strip()
    if ip in ATTACKER_IPS:
        return True
    if event in THREAT_CLOUD_APIS and ip not in {
        "autoscaling.amazonaws.com", "config.amazonaws.com",
        "ec2.amazonaws.com", "lambda.amazonaws.com",
        "events.amazonaws.com", "inspector.amazonaws.com",
        "guardduty.amazonaws.com", "vpc-flow-logs.amazonaws.com",
        "AWS Internal",
    }:
        return True
    return False


def is_threat_windows(row):
    """Check if a Windows event is a threat."""
    msg = row.get("Message", "").lower()
    host = row.get("host", "").strip()

    # Failed logons
    if "account failed to log on" in msg:
        return True

    # Account manipulation
    for phrase in ["user account was created", "user account was enabled",
                   "password reset", "member was added to a security",
                   "member was added to a local", "user account was changed"]:
        if phrase in msg:
            return True

    # On compromised host
    if host == COMPROMISED_HOST:
        for proc in SUSPICIOUS_PROCS:
            if proc in msg:
                return True
        if "windows filtering platform has blocked" in msg:
            return True
        if "service was installed" in msg:
            return True
        if "special privileges assigned" in msg:
            return True

    return False


def build_alert_text_network(r):
    parts = [f"Network event ({r.get('sourcetype', '')})"]
    src = r.get("src_ip", "").strip()
    dest = r.get("dest_ip", "").strip()
    sp = r.get("src_port", "").strip()
    dp = r.get("dest_port", "").strip()
    bi = r.get("bytes_in", "").strip()
    bo = r.get("bytes_out", "").strip()
    if src and dest:
        conn = f"from {src}"
        if sp: conn += f":{sp}"
        conn += f" to {dest}"
        if dp: conn += f":{dp}"
        parts.append(conn)
    elif src:
        parts.append(f"source {src}")
    elif dest:
        parts.append(f"destination {dest}")
    if bi or bo:
        parts.append(f"bytes_in={bi or '0'} bytes_out={bo or '0'}")
    host = r.get("host", "").strip()
    if host:
        parts.append(f"on host {host}")
    return " | ".join(parts)


def build_alert_text_cloud(r):
    event = r.get("eventName", "").strip()
    ip = r.get("sourceIPAddress", "").strip()
    st = r.get("sourcetype", "").strip()
    host = r.get("host", "").strip()
    parts = [f"Cloud event ({st}): {event}"]
    if ip: parts.append(f"from {ip}")
    if host: parts.append(f"on host {host}")
    return " | ".join(parts)


def truncate(text, max_len=1000):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ── LOADERS WITH THREAT-AWARE SAMPLING ────────────────────────────────────────

def load_windows():
    path = os.path.join(INPUT_DIR, "BotsV3_Windows.csv")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    threats = [r for r in rows if r.get("Message", "").strip() and is_threat_windows(r)]
    benign = [r for r in rows if r.get("Message", "").strip() and not is_threat_windows(r)]

    print(f"[Windows] {len(rows)} total | {len(threats)} threats | {len(benign)} benign")

    # Take all threats, fill rest with benign
    remaining = max(0, TARGETS["windows"] - len(threats))
    random.shuffle(benign)
    selected_benign = benign[:remaining]

    results = []
    for r in threats:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", "") or r.get("ComputerName", ""),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "windows",
            "alert_text": truncate(r.get("Message", "").strip()),
            "label": "threat",
        })
    for r in selected_benign:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", "") or r.get("ComputerName", ""),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "windows",
            "alert_text": truncate(r.get("Message", "").strip()),
            "label": "benign",
        })

    stats["windows"] = {"threat": len(threats), "benign": len(selected_benign)}
    print(f"[Windows] Selected: {len(threats)} threats + {len(selected_benign)} benign = {len(results)}")
    return results


def load_linux():
    path = os.path.join(INPUT_DIR, "BotsV3_LinuxFixed.csv")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    threats = [r for r in rows if r.get("_raw", "").strip() and is_threat_linux(r.get("_raw", ""))]
    benign = [r for r in rows if r.get("_raw", "").strip() and not is_threat_linux(r.get("_raw", ""))]

    print(f"[Linux] {len(rows)} total | {len(threats)} threats | {len(benign)} benign")

    remaining = max(0, TARGETS["linux"] - len(threats))
    random.shuffle(benign)
    selected_benign = benign[:remaining]

    results = []
    for r in threats:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", ""),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "linux",
            "alert_text": truncate(r.get("_raw", "").strip()),
            "label": "threat",
        })
    for r in selected_benign:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", ""),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "linux",
            "alert_text": truncate(r.get("_raw", "").strip()),
            "label": "benign",
        })

    stats["linux"] = {"threat": len(threats), "benign": len(selected_benign)}
    print(f"[Linux] Selected: {len(threats)} threats + {len(selected_benign)} benign = {len(results)}")
    return results


def load_network():
    path = os.path.join(INPUT_DIR, "BotsV3_Network.csv")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    # Filter: must have at least one IP
    rows = [r for r in rows if r.get("src_ip", "").strip() or r.get("dest_ip", "").strip()]

    threats = [r for r in rows if is_threat_network(r)]
    benign = [r for r in rows if not is_threat_network(r)]

    print(f"[Network] {len(rows)} w/ IPs | {len(threats)} threats | {len(benign)} benign")

    remaining = max(0, TARGETS["network"] - len(threats))
    random.shuffle(benign)
    selected_benign = benign[:remaining]

    results = []
    for r in threats:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", "").strip(),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "network",
            "alert_text": build_alert_text_network(r),
            "label": "threat",
        })
    for r in selected_benign:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", "").strip(),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "network",
            "alert_text": build_alert_text_network(r),
            "label": "benign",
        })

    stats["network"] = {"threat": len(threats), "benign": len(selected_benign)}
    print(f"[Network] Selected: {len(threats)} threats + {len(selected_benign)} benign = {len(results)}")
    return results


def load_cloud():
    path = os.path.join(INPUT_DIR, "BotsV3_Cloud.csv")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    # Filter: must have eventName
    rows = [r for r in rows if r.get("eventName", "").strip()]

    threats = [r for r in rows if is_threat_cloud(r)]
    benign = [r for r in rows if not is_threat_cloud(r)]

    print(f"[Cloud] {len(rows)} w/ eventName | {len(threats)} threats | {len(benign)} benign")

    remaining = max(0, TARGETS["cloud"] - len(threats))
    random.shuffle(benign)
    selected_benign = benign[:remaining]

    results = []
    for r in threats:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", "").strip(),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "cloud",
            "alert_text": build_alert_text_cloud(r),
            "label": "threat",
        })
    for r in selected_benign:
        results.append({
            "timestamp": r.get("_time", ""),
            "host": r.get("host", "").strip(),
            "sourcetype": r.get("sourcetype", ""),
            "source_category": "cloud",
            "alert_text": build_alert_text_cloud(r),
            "label": "benign",
        })

    stats["cloud"] = {"threat": len(threats), "benign": len(selected_benign)}
    print(f"[Cloud] Selected: {len(threats)} threats + {len(selected_benign)} benign = {len(results)}")
    return results


def main():
    print("=" * 60)
    print("BOTS v3 Threat-Aware Combiner v2")
    print("=" * 60 + "\n")

    all_rows = []
    all_rows.extend(load_windows())
    print()
    all_rows.extend(load_linux())
    print()
    all_rows.extend(load_network())
    print()
    all_rows.extend(load_cloud())
    print()

    random.shuffle(all_rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(all_rows)

    total_threat = sum(s["threat"] for s in stats.values())
    total_benign = sum(s["benign"] for s in stats.values())
    total = total_threat + total_benign

    print("=" * 60)
    print(f"FINAL DATASET: {total:,} rows")
    print(f"  Threat: {total_threat:,} ({100*total_threat/total:.1f}%)")
    print(f"  Benign: {total_benign:,} ({100*total_benign/total:.1f}%)")
    print(f"\nPer category:")
    for cat in sorted(stats.keys()):
        s = stats[cat]
        t = s["threat"] + s["benign"]
        print(f"  {cat:12s}: {t:>6,} total | {s['threat']:>5,} threat | {s['benign']:>5,} benign")
    print(f"\nWritten to: {OUTPUT_PATH}")
    print(f"\nNOTE: Endpoint excluded (needs Splunk re-export with _raw).")
    print(f"      IAM excluded per project decision.")
    print("=" * 60)


if __name__ == "__main__":
    main()
