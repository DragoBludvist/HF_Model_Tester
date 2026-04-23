"""
standardize_training_data.py — Convert raw BOTS v3 text into noise-free SecureBERT format.

Takes botsv3_combined_v2.csv and produces botsv3_standardized.csv with clean text.
Every field that the model sees carries signal. Timestamps, PIDs, logon IDs,
device paths, format versions, and repeated data are stripped.

Run: python standardize_training_data.py
"""

import csv
import re
import json

INPUT = "botsv3_combined_v2.csv"
OUTPUT = "botsv3_standardized.csv"


# ── WINDOWS STANDARDIZER ─────────────────────────────────────────────────────

def standardize_windows(text):
    """
    Windows Event Logs: literal \\n \\t separated key:value pairs.
    Extract: description, account, process, network info, direction.
    Strip: PIDs, logon IDs, token types, filter IDs, full device paths, labels.
    """
    # Remove category tag
    text = re.sub(r'^\[windows\]\s*', '', text)

    # Normalize literal \\n and \\t to spaces for easier parsing
    normalized = text.replace('\\n', ' ').replace('\\t', ' ')
    normalized = re.sub(r'\s{2,}', ' ', normalized)

    # Get description (first sentence)
    desc = normalized.split('.')[0].strip()
    if len(desc) < 10:
        # Too short — try getting more
        desc = '. '.join(normalized.split('.')[:2]).strip()
    # Cap at reasonable length
    desc = desc[:100]

    parts = [f"[endpoint] {desc}."]

    # Account Name — extract username
    account = re.search(r'Account Name:\s*(\S+)', normalized)
    if account and account.group(1) not in ('-', 'NULL', 'NULL SID'):
        parts.append(f"account={account.group(1)}")

    # Security ID — simplify
    sid = re.search(r'Security ID:\s*([\w\\\\]+)', normalized)
    if sid:
        s = sid.group(1)
        if 'SYSTEM' in s:
            parts.append("sid=SYSTEM")
        elif 'LOCAL SERVICE' in s:
            parts.append("sid=LOCAL_SERVICE")
        elif 'NETWORK SERVICE' in s:
            parts.append("sid=NETWORK_SERVICE")
        elif 'AzureAD' in s:
            user = s.split('\\\\')[-1] if '\\\\' in s else s.split('\\')[-1]
            parts.append(f"sid=AzureAD\\{user}")

    # Process / Application — just the exe name
    for pattern in [r'Application Name:\s*(\S+)', r'Process Name:\s*(\S+)',
                    r'New Process Name:\s*(\S+)']:
        m = re.search(pattern, normalized)
        if m:
            exe = m.group(1).split('\\\\')[-1].split('\\')[-1]
            if exe and exe not in ('-', 'Name:'):
                key = "new_process" if "New Process" in pattern else "process"
                parts.append(f"{key}={exe}")
                break

    # Network direction
    direction = re.search(r'Direction:\s*(\w+)', normalized)
    if direction:
        parts.append(f"direction={direction.group(1)}")

    # Source/Dest addresses
    src_addr = re.search(r'Source Address:\s*([\d.:a-fA-F]+)', normalized)
    dst_addr = re.search(r'Destination Address:\s*([\d.:a-fA-F]+)', normalized)
    src_port = re.search(r'Source Port:\s*(\d+)', normalized)
    dst_port = re.search(r'Destination Port:\s*(\d+)', normalized)
    proto = re.search(r'Protocol:\s*(\d+)', normalized)

    if src_addr: parts.append(f"src={src_addr.group(1)}")
    if dst_addr: parts.append(f"dst={dst_addr.group(1)}")
    if src_port and dst_port: parts.append(f"port={src_port.group(1)}->{dst_port.group(1)}")
    if proto:
        proto_map = {'6': 'tcp', '17': 'udp'}
        parts.append(f"proto={proto_map.get(proto.group(1), proto.group(1))}")

    # Service name
    service = re.search(r'Service Name:\s*(\S+)', normalized)
    if service and service.group(1) not in ('-',):
        parts.append(f"service={service.group(1)}")

    # PowerShell
    if 'Scriptblock' in text or 'powershell' in text.lower():
        if '-enc ' in text:
            parts.append("powershell=encoded")
        else:
            parts.append("powershell=yes")

    return " ".join(parts)


# ── LINUX STANDARDIZER ────────────────────────────────────────────────────────

def standardize_linux(text):
    """
    Linux syslog: three sub-formats (SSH, nvzFlow, generic).
    Extract: action, user, source IP, port, hostname, process.
    Strip: timestamps, epoch times, device UUIDs, flow metadata.
    """
    # SSH events
    if "sshd" in text:
        parts = ["[identity]"]

        # Action
        if "Invalid user" in text or "invalid user" in text:
            parts.append("SSH invalid user attempt.")
            user = re.search(r'[Ii]nvalid user\s+(\S+)', text)
            if user: parts.append(f"user={user.group(1)}")
        elif "Failed password" in text:
            parts.append("SSH failed password.")
            user = re.search(r'for\s+(\S+)', text)
            if user and user.group(1) != 'invalid': parts.append(f"user={user.group(1)}")
        elif "Accepted" in text:
            parts.append("SSH login accepted.")
            user = re.search(r'for\s+(\S+)', text)
            if user: parts.append(f"user={user.group(1)}")
        elif "Received disconnect" in text:
            parts.append("SSH disconnect.")
        elif "session opened" in text:
            parts.append("SSH session opened.")
        elif "session closed" in text:
            parts.append("SSH session closed.")
        else:
            parts.append("SSH event.")

        ip = re.search(r'from\s+([\d.]+)', text)
        if ip: parts.append(f"src={ip.group(1)}")

        port = re.search(r'port\s+(\d+)', text)
        if port: parts.append(f"sport={port.group(1)}")

        host = re.search(r'^\w+\s+\d+\s+[\d:]+\s+(\S+)', text)
        if host: parts.append(f"agent={host.group(1)}")

        return " ".join(parts)

    # nvzFlow network flows — reclassify as firewall
    if "nvzFlow" in text:
        parts = ["[firewall] Network flow."]

        sa = re.search(r'sa="([\d.]+)"', text)
        sp = re.search(r'sp="(\d+)"', text)
        da = re.search(r'da="([\d.]+)"', text)
        dp = re.search(r'dp="(\d+)"', text)
        pr = re.search(r'pr="(\d+)"', text)

        if sa: parts.append(f"src={sa.group(1)}")
        if da: parts.append(f"dst={da.group(1)}")
        if sp and dp: parts.append(f"port={sp.group(1)}->{dp.group(1)}")
        if pr:
            proto_map = {'6': 'tcp', '17': 'udp', '1': 'icmp'}
            parts.append(f"proto={proto_map.get(pr.group(1), pr.group(1))}")

        user = re.search(r'liuidp="(\w+)"', text)
        if user: parts.append(f"user={user.group(1)}")

        return " ".join(parts)

    # Generic syslog
    parts = ["[system]"]
    host = re.search(r'^\w+\s+\d+\s+[\d:]+\s+(\S+)', text)
    if host: parts.append(f"agent={host.group(1)}")

    proc = re.search(r'(\S+)\[\d+\]:', text)
    if proc: parts.append(f"process={proc.group(1)}")

    # Get the message after the process[PID]: part
    msg = re.search(r'\[\d+\]:\s*(.+)', text)
    if msg:
        clean_msg = msg.group(1).strip()[:150]
        parts.append(clean_msg)

    return " ".join(parts)


# ── NETWORK STANDARDIZER ─────────────────────────────────────────────────────

def standardize_network(text):
    """
    Network stream: from IP:port to IP:port | bytes | host
    Extract: protocol, IPs, ports, byte counts, host.
    Strip: "Network event" label, stream format name.
    """
    parts = ["[firewall]"]

    proto = re.search(r'stream:(\w+)', text)
    if proto:
        parts.append(f"{proto.group(1).upper()} connection.")
    else:
        parts.append("Network connection.")

    conn = re.search(r'from\s+([\d.]+):?(\d+)?\s+to\s+([\d.]+):?(\d+)?', text)
    if conn:
        parts.append(f"src={conn.group(1)}")
        parts.append(f"dst={conn.group(3)}")
        sport = conn.group(2)
        dport = conn.group(4)
        if sport and dport:
            parts.append(f"port={sport}->{dport}")
        elif dport:
            parts.append(f"dport={dport}")

    for field in ['bytes_in', 'bytes_out']:
        m = re.search(rf'{field}=(\d+)', text)
        if m: parts.append(f"{field}={m.group(1)}")

    host = re.search(r'on host\s+(\S+)', text)
    if host: parts.append(f"agent={host.group(1)}")

    return " ".join(parts)


# ── CLOUD STANDARDIZER ────────────────────────────────────────────────────────

def standardize_cloud(text):
    """
    Cloud: "Cloud event (aws:cloudtrail): APIName | from IP | on host H"
    Already fairly clean. Just normalize the format.
    """
    text = re.sub(r'^\[cloud\]\s*', '', text)
    parts = ["[cloud]"]

    api = re.search(r':\s*(\w+)\s*\|', text)
    if api:
        parts.append(f"api={api.group(1)}")

    ip = re.search(r'from\s+([\d.]+)', text)
    if ip: parts.append(f"src={ip.group(1)}")

    host = re.search(r'on host\s+(\S+)', text)
    if host: parts.append(f"agent={host.group(1)}")

    return " ".join(parts)


# ── ENDPOINT STANDARDIZER ────────────────────────────────────────────────────

def standardize_endpoint(text):
    """
    Endpoint: Sysmon XML or Osquery JSON.
    Extract structured fields, strip XML/JSON noise.
    """
    text = re.sub(r'^\[endpoint\]\s*', '', text)

    # Sysmon XML
    if text.startswith("<Event") or "<EventID>" in text:
        parts = ["[endpoint]"]

        eid = re.search(r'<EventID>(\d+)</EventID>', text)
        if eid:
            eid_map = {
                '1': 'Process created', '2': 'File time changed', '3': 'Network connection',
                '5': 'Process terminated', '7': 'Image loaded', '8': 'CreateRemoteThread',
                '11': 'File created', '12': 'Registry created/deleted',
                '13': 'Registry value set', '15': 'FileCreateStreamHash',
            }
            parts.append(f"Sysmon: {eid_map.get(eid.group(1), f'EventID={eid.group(1)}')}")

        for field, key in [('Image', 'process'), ('TargetFilename', 'file'),
                           ('DestinationIp', 'dst'), ('DestinationPort', 'dport'),
                           ('SourceIp', 'src'), ('User', 'user'), ('Protocol', 'proto'),
                           ('ParentImage', 'parent_process')]:
            m = re.search(rf"Name='{field}'[^>]*>([^<]+)<", text)
            if m:
                val = m.group(1).strip()
                if key in ('process', 'file', 'parent_process', 'user'):
                    val = val.split('\\')[-1]
                parts.append(f"{key}={val}")

        comp = re.search(r"<Computer>([^<]+)</Computer>", text)
        if comp: parts.append(f"agent={comp.group(1)}")

        return " ".join(parts)

    # Osquery JSON
    if text.startswith("{"):
        parts = ["[endpoint]"]
        try:
            j = json.loads(text[:3000])

            name = j.get("name", "")
            if name:
                short = name.replace("pack_", "").replace("process-monitoring_", "").replace("incident-response_", "IR:")
                parts.append(f"osquery: {short}")

            host = j.get("hostIdentifier", "")
            if host: parts.append(f"agent={host}")

            user = j.get("decorations", {}).get("username", "")
            if user: parts.append(f"user={user}")

            cols = j.get("columns", {})
            for key in ["cmdline", "path", "name", "target_path"]:
                if key in cols and cols[key]:
                    val = str(cols[key])
                    if key in ('cmdline', 'path', 'target_path'):
                        val = val.split('/')[-1] if '/' in val else val.split('\\')[-1]
                    parts.append(f"{key}={val[:60]}")

        except (json.JSONDecodeError, KeyError):
            parts.append("osquery: parse_error")

        return " ".join(parts)

    return f"[endpoint] {text[:200]}"


# ── DNS STANDARDIZER ─────────────────────────────────────────────────────────

def standardize_dns(text):
    """
    DNS: JSON with query, response, host_addr, reply_code.
    Extract: query domain, reply code, query type, host.
    Strip: timestamps, message arrays, raw JSON structure.
    """
    text = re.sub(r'^\[dns\]\s*', '', text)
    parts = ["[dns]"]

    try:
        j = json.loads(text[:2000])

        # Query domain — could be "query" or "name"
        query = j.get("query", "") or j.get("name", "")
        if isinstance(query, list):
            query = query[0] if query else ""
        if query:
            parts.append(f"query={query}")

        qtype = j.get("query_type", "")
        if isinstance(qtype, list):
            qtype = qtype[0] if qtype else ""
        if qtype:
            parts.append(f"type={qtype}")

        reply = j.get("reply_code", "")
        if reply:
            parts.append(f"reply={reply}")

        # Host — could be "host_addr", "src_ip", or "dest_ip"
        host = j.get("host_addr", "") or j.get("src_ip", "") or j.get("dest_ip", "")
        if isinstance(host, list):
            host = host[0] if host else ""
        if host:
            parts.append(f"host={host}")

    except (json.JSONDecodeError, KeyError):
        parts.append(text[:150])

    return " ".join(parts)


# ── MAIN ROUTER ──────────────────────────────────────────────────────────────

def standardize(text, category):
    """Route to the correct standardizer based on category."""
    if category == "windows":
        return standardize_windows(text)
    elif category == "linux":
        return standardize_linux(text)
    elif category == "network":
        return standardize_network(text)
    elif category == "cloud":
        return standardize_cloud(text)
    elif category == "endpoint":
        return standardize_endpoint(text)
    elif category == "dns":
        return standardize_dns(text)
    else:
        return f"[{category}] {text[:200]}"


# ── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Locate input
    import os
    input_path = INPUT
    for candidate in [INPUT, f"data/{INPUT}", f"/mnt/user-data/outputs/{INPUT}"]:
        if os.path.exists(candidate):
            input_path = candidate
            break

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {input_path}")

    # Standardize
    standardized = []
    for r in rows:
        text = r['text']
        cat = r['source_category']
        clean = standardize(text, cat)
        standardized.append({
            'text': clean,
            'label_int': r['label_int'],
            'label': r['label'],
            'source_category': cat,
        })

    # Write
    output_path = OUTPUT
    for candidate in [OUTPUT, f"/mnt/user-data/outputs/{OUTPUT}"]:
        try:
            with open(candidate, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['text', 'label_int', 'label', 'source_category'])
                writer.writeheader()
                writer.writerows(standardized)
            output_path = candidate
            break
        except:
            continue

    print(f"Written {len(standardized)} rows to {output_path}")

    # Show before/after samples
    print(f"\n{'=' * 70}")
    print("BEFORE vs AFTER")
    print(f"{'=' * 70}")

    shown = set()
    for orig, std in zip(rows, standardized):
        cat = orig['source_category']
        label = orig['label']
        key = f"{cat}_{label}"
        if key in shown: continue
        shown.add(key)

        print(f"\n--- {cat} / {label} ---")
        print(f"  BEFORE: {orig['text'][:120]}...")
        print(f"  AFTER:  {std['text'][:120]}")

        orig_len = len(orig['text'])
        std_len = len(std['text'])
        reduction = round((1 - std_len / orig_len) * 100) if orig_len > 0 else 0
        print(f"  Reduction: {orig_len} -> {std_len} chars ({reduction}% smaller)")
