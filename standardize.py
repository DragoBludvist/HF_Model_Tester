"""
standardize.py — Converts any SIEM export into clean text for SecureBERT.

Handles messy data. Mixed sourcetypes. Missing fields. Unknown formats.
Takes whatever the SIEM gives you and outputs clean, consistent text.

Usage:
    python standardize.py --input data/export.csv --output data/standardized.csv
"""

import csv
import json
import re
import argparse
import os
import sys

csv.field_size_limit(10 * 1024 * 1024)

# ── NON-SECURITY SOURCETYPES (skip these entirely) ────────────────────────────

SKIP_SOURCETYPES = {
    "PerfmonMk:Process", "WinHostMon", "top", "ps", "vmstat", "cpu", "iostat",
    "bandwidth", "interfaces", "who", "netstat", "protocol", "df",
    "Script:ListeningPorts", "Script:GetEndpointInfo", "Script:InstalledApps",
    "osquery:info", "amazon-ssm-agent-too_small", "errors-too_small",
    "ess_content_importer", "aws:config:rule",
    "WinEventLog:Application",  # application logs, not security
}


# ── HANDLERS (one per sourcetype family) ──────────────────────────────────────

def handle_windows(row):
    """WinEventLog:Security — structured columns from Splunk."""
    parts = ["[endpoint]"]
    msg = row.get("Message", "") or row.get("_raw", "")
    desc = msg.split("\n")[0].strip().split("\r")[0].strip()
    if "\\n" in desc:
        desc = desc.split("\\n")[0].strip()
    if desc:
        parts.append(desc[:120])
    code = row.get("EventCode", "")
    if code:
        parts.append(f"EventCode={code}")
    account = row.get("Account_Name", "")
    if account and account not in ("-", "NULL", ""):
        parts.append(f"account={account}")
    host = row.get("ComputerName", "") or row.get("host", "")
    if host:
        parts.append(f"agent={host}")
    return " ".join(parts)


def handle_cloud(row):
    """aws:cloudtrail — structured columns from Splunk."""
    parts = ["[cloud]"]
    api = row.get("eventName", "")
    if api:
        parts.append(f"api={api}")
    src = row.get("sourceIPAddress", "")
    if src:
        parts.append(f"src={src}")
    host = row.get("host", "")
    if host:
        parts.append(f"agent={host}")
    return " ".join(parts)


def handle_network_structured(row):
    """stream:* with structured columns (src_ip, dest_ip populated)."""
    parts = ["[firewall]"]
    stype = row.get("sourcetype", "")
    proto = row.get("proto", "")
    if proto:
        parts.append(f"{proto.upper()} connection.")
    elif "tcp" in stype:
        parts.append("TCP connection.")
    elif "udp" in stype:
        parts.append("UDP connection.")
    elif "dns" in stype:
        parts.append("DNS connection.")
    elif "http" in stype:
        parts.append("HTTP connection.")
    else:
        parts.append("Connection.")
    src = row.get("src_ip", "")
    dst = row.get("dest_ip", "")
    if src: parts.append(f"src={src}")
    if dst: parts.append(f"dst={dst}")
    sport = row.get("src_port", "")
    dport = row.get("dest_port", "")
    if sport and dport: parts.append(f"port={sport}->{dport}")
    elif dport: parts.append(f"dport={dport}")
    if proto: parts.append(f"proto={proto}")
    bi = row.get("bytes_in", "")
    bo = row.get("bytes_out", "")
    if bi: parts.append(f"bytes_in={bi}")
    if bo: parts.append(f"bytes_out={bo}")
    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_stream_json(row):
    """stream:tcp/udp/smtp etc. where structured fields are empty — extract from _raw."""
    raw = row.get("_raw", "")
    stype = row.get("sourcetype", "")
    parts = ["[firewall]"]

    if "tcp" in stype: parts.append("TCP connection.")
    elif "udp" in stype: parts.append("UDP connection.")
    elif "smtp" in stype: parts.append("SMTP connection.")
    elif "smb" in stype: parts.append("SMB connection.")
    elif "dhcp" in stype: parts.append("DHCP connection.")
    elif "icmp" in stype: parts.append("ICMP connection.")
    elif "igmp" in stype: parts.append("IGMP connection.")
    elif "arp" in stype: parts.append("ARP connection.")
    else: parts.append("Connection.")

    # Try JSON parse first
    extracted = False
    try:
        j = json.loads(raw[:5000])
        src = j.get("src_ip", "") or j.get("src", "")
        dst = j.get("dest_ip", "") or j.get("dst", "") or j.get("dest", "")
        if src: parts.append(f"src={src}"); extracted = True
        if dst: parts.append(f"dst={dst}"); extracted = True
        sport = j.get("src_port", "")
        dport = j.get("dest_port", "")
        if sport and dport: parts.append(f"port={sport}->{dport}")
        elif dport: parts.append(f"dport={dport}")
        bi = j.get("bytes_in", "") or j.get("bytes", "")
        bo = j.get("bytes_out", "")
        if bi: parts.append(f"bytes_in={bi}")
        if bo: parts.append(f"bytes_out={bo}")
    except:
        pass

    # If JSON parse failed or didn't have IPs, try regex on raw string
    if not extracted:
        src = re.search(r'"src_ip"\s*:\s*"([^"]+)"', raw)
        dst = re.search(r'"dest_ip"\s*:\s*"([^"]+)"', raw)
        sp = re.search(r'"src_port"\s*:\s*(\d+)', raw)
        dp = re.search(r'"dest_port"\s*:\s*(\d+)', raw)
        if src: parts.append(f"src={src.group(1)}")
        if dst: parts.append(f"dst={dst.group(1)}")
        if sp and dp: parts.append(f"port={sp.group(1)}->{dp.group(1)}")
        elif dp: parts.append(f"dport={dp.group(1)}")

    # Use structured columns as fallback for bytes
    if "bytes_in=" not in " ".join(parts):
        bi = row.get("bytes_in", "")
        bo = row.get("bytes_out", "")
        if bi: parts.append(f"bytes_in={bi}")
        if bo: parts.append(f"bytes_out={bo}")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_dns(row):
    """stream:dns — merge structured columns (IPs) with _raw JSON (query, reply)."""
    raw = row.get("_raw", "")
    parts = ["[dns]"]

    # Always try to get DNS-specific fields from _raw JSON
    query = ""
    qtype = ""
    reply = ""
    try:
        j = json.loads(raw[:5000])
        q = j.get("query", "") or j.get("name", "")
        if isinstance(q, list): q = q[0] if q else ""
        query = q
        qt = j.get("query_type", "")
        if isinstance(qt, list): qt = qt[0] if qt else ""
        qtype = qt
        reply = j.get("reply_code", "")
    except:
        pass

    # Add DNS query info first (most important for classification)
    if query: parts.append(f"query={query}")
    if qtype: parts.append(f"type={qtype}")
    if reply: parts.append(f"reply={reply}")

    # Add network tuple from structured columns
    src = row.get("src_ip", "")
    dst = row.get("dest_ip", "")
    if src: parts.append(f"src={src}")
    if dst: parts.append(f"dst={dst}")
    sport = row.get("src_port", "")
    dport = row.get("dest_port", "")
    if sport and dport: parts.append(f"port={sport}->{dport}")

    bi = row.get("bytes_in", "")
    bo = row.get("bytes_out", "")
    if bi: parts.append(f"bytes_in={bi}")
    if bo: parts.append(f"bytes_out={bo}")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_linux(row):
    """syslog — SSH, nvzFlow, or generic."""
    raw = row.get("_raw", "")

    if "sshd" in raw:
        parts = ["[identity]"]
        if "Invalid user" in raw or "invalid user" in raw:
            parts.append("SSH invalid user attempt.")
            u = re.search(r'[Ii]nvalid user\s+(\S+)', raw)
            if u: parts.append(f"user={u.group(1)}")
        elif "Failed password" in raw:
            parts.append("SSH failed password.")
            u = re.search(r'for\s+(\S+)', raw)
            if u and u.group(1) != "invalid": parts.append(f"user={u.group(1)}")
        elif "Accepted" in raw:
            parts.append("SSH login accepted.")
            u = re.search(r'for\s+(\S+)', raw)
            if u: parts.append(f"user={u.group(1)}")
        elif "Received disconnect" in raw:
            parts.append("SSH disconnect.")
        else:
            parts.append("SSH event.")
        ip = re.search(r'from\s+([\d.]+)', raw)
        if ip: parts.append(f"src={ip.group(1)}")
        port = re.search(r'port\s+(\d+)', raw)
        if port: parts.append(f"sport={port.group(1)}")
        host = row.get("host", "")
        if host: parts.append(f"agent={host}")
        return " ".join(parts)

    if "nvzFlow" in raw:
        parts = ["[firewall] Network flow."]
        for tag, key in [("sa","src"),("da","dst"),("sp","sport"),("dp","dport")]:
            m = re.search(rf'{tag}="([^"]+)"', raw)
            if m: parts.append(f"{key}={m.group(1)}")
        pr = re.search(r'pr="(\d+)"', raw)
        if pr:
            pmap = {"6":"tcp","17":"udp","1":"icmp"}
            parts.append(f"proto={pmap.get(pr.group(1), pr.group(1))}")
        user = re.search(r'liuidp="(\w+)"', raw)
        if user: parts.append(f"user={user.group(1)}")
        return " ".join(parts)

    parts = ["[system]"]
    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    proc = re.search(r'(\S+)\[\d+\]:', raw)
    if proc: parts.append(f"process={proc.group(1)}")
    msg = re.search(r'\[\d+\]:\s*(.+)', raw)
    if msg: parts.append(msg.group(1).strip()[:150])
    return " ".join(parts)


def handle_sysmon(row):
    """Sysmon XML from _raw."""
    raw = row.get("_raw", "")
    parts = ["[endpoint]"]
    eid = re.search(r'<EventID>(\d+)</EventID>', raw)
    if eid:
        eid_map = {"1":"Process created.","3":"Network connection.",
                   "5":"Process terminated.","7":"Image loaded.",
                   "8":"CreateRemoteThread.","11":"File created.",
                   "12":"Registry created/deleted.","13":"Registry value set."}
        parts.append(eid_map.get(eid.group(1), f"Sysmon EventID {eid.group(1)}."))
    for field, key in [("Image","process"),("ParentImage","parent"),
                       ("TargetFilename","file"),("DestinationIp","dst"),
                       ("DestinationPort","dport"),("SourceIp","src"),
                       ("User","user"),("Protocol","proto")]:
        m = re.search(rf"Name='{field}'[^>]*>([^<]+)<", raw)
        if m:
            val = m.group(1).strip()
            if key in ("process","parent","file","user"):
                val = val.split("\\")[-1]
            parts.append(f"{key}={val}")
    comp = re.search(r"<Computer>([^<]+)</Computer>", raw)
    if comp: parts.append(f"agent={comp.group(1)}")
    return " ".join(parts)


def handle_osquery(row):
    """osquery:results JSON from _raw."""
    raw = row.get("_raw", "")
    parts = ["[endpoint]"]
    try:
        j = json.loads(raw[:3000])
        name = j.get("name", "")
        if name:
            short = name.replace("pack_","").replace("process-monitoring_","").replace("incident-response_","IR:")
            parts.append(f"osquery: {short}.")
        host = j.get("hostIdentifier", "")
        if host: parts.append(f"agent={host}")
        user = j.get("decorations", {}).get("username", "")
        if user: parts.append(f"user={user}")
        cols = j.get("columns", {})
        for k in ["cmdline","path","name","target_path"]:
            if k in cols and cols[k]:
                v = str(cols[k])
                if "/" in v: v = v.split("/")[-1]
                elif "\\" in v: v = v.split("\\")[-1]
                parts.append(f"{k}={v[:60]}")
    except:
        host = row.get("host", "")
        if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_cisco_asa(row):
    """cisco:asa syslog from _raw."""
    raw = row.get("_raw", "")
    parts = ["[firewall]"]

    # Extract ASA message ID and severity
    asa = re.search(r'%ASA-(\d)-(\d+):', raw)
    if asa:
        severity = asa.group(1)
        msg_id = asa.group(2)
        # Get the message after the ASA code
        msg_text = raw.split(f"%ASA-{severity}-{msg_id}: ")[-1].strip()

        # Determine action from message
        if "Deny" in msg_text or "denied" in msg_text:
            parts.append("Connection denied.")
        elif "Teardown" in msg_text:
            parts.append("Connection teardown.")
        elif "Built" in msg_text:
            parts.append("Connection built.")
        else:
            parts.append(msg_text[:80])

    # Extract IPs from various ASA formats
    # Format 1: "from IP/port to IP/port"
    conn1 = re.search(r'from\s+(?:\w+:)?([\d.]+)/(\d+)\s+to\s+(?:\w+:)?([\d.]+)/(\d+)', raw)
    if conn1:
        parts.append(f"src={conn1.group(1)}")
        parts.append(f"dst={conn1.group(3)}")
        parts.append(f"port={conn1.group(2)}->{conn1.group(4)}")
    else:
        # Format 2: "src inside:IP/port dst outside:IP/port"
        conn2 = re.search(r'src\s+\w+:([\d.]+)/(\d+)\s+dst\s+\w+:([\d.]+)/(\d+)', raw)
        if conn2:
            parts.append(f"src={conn2.group(1)}")
            parts.append(f"dst={conn2.group(3)}")
            parts.append(f"port={conn2.group(2)}->{conn2.group(4)}")

    # Protocol
    proto = re.search(r'(tcp|udp|icmp)', raw, re.IGNORECASE)
    if proto: parts.append(f"proto={proto.group(1).lower()}")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")

    return " ".join(parts)


def handle_vpc_flow(row):
    """aws:cloudwatchlogs:vpcflow — space-delimited fields."""
    raw = row.get("_raw", "")
    parts = ["[firewall] VPC flow."]

    # Format: version account-id eni src-ip dst-ip src-port dst-port protocol packets bytes start end action log-status
    fields = raw.split()
    if len(fields) >= 14:
        parts.append(f"src={fields[3]}")
        parts.append(f"dst={fields[4]}")
        parts.append(f"port={fields[5]}->{fields[6]}")
        proto_map = {"6":"tcp","17":"udp","1":"icmp"}
        parts.append(f"proto={proto_map.get(fields[7], fields[7])}")
        action = fields[12]
        if action: parts.append(f"action={action}")
    elif len(fields) >= 6:
        # Partial format — extract what we can
        for f in fields:
            if re.match(r'\d+\.\d+\.\d+\.\d+', f):
                if "src=" not in " ".join(parts): parts.append(f"src={f}")
                elif "dst=" not in " ".join(parts): parts.append(f"dst={f}")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_access_log(row):
    """access_combined — Apache/Nginx access logs."""
    raw = row.get("_raw", "")
    parts = ["[web]"]

    # Format: IP - user [date] "METHOD path HTTP/ver" status size "referer" "ua"
    m = re.match(r'([\d.]+)\s+\S+\s+(\S+)\s+\[.*?\]\s+"(\w+)\s+(\S+)\s+', raw)
    if m:
        parts.append(f"src={m.group(1)}")
        if m.group(2) != "-": parts.append(f"user={m.group(2)}")
        parts.append(f"method={m.group(3)}")
        parts.append(f"url={m.group(4)[:80]}")

    status = re.search(r'"\s+(\d{3})\s+', raw)
    if status: parts.append(f"status={status.group(1)}")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_rds_audit(row):
    """aws:rds:audit — database audit logs."""
    raw = row.get("_raw", "")
    parts = ["[cloud] RDS audit."]

    # Comma-separated: timestamp,host,user,client,thread,query_id,type,db,query
    fields = raw.split(",")
    if len(fields) >= 8:
        if fields[2]: parts.append(f"user={fields[2]}")
        if fields[3]: parts.append(f"src={fields[3]}")
        if fields[6]: parts.append(f"action={fields[6]}")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


def handle_symantec(row):
    """symantec:ep:agent:file — Symantec endpoint events."""
    raw = row.get("_raw", "")
    parts = ["[antivirus]"]

    # Extract key fields from comma-separated format
    if "management server" in raw.lower() or "The management" in raw:
        parts.append("Symantec management event.")
    elif "virus" in raw.lower() or "malware" in raw.lower():
        parts.append("Symantec: Threat detected.")
    else:
        parts.append("Symantec event.")

    host = row.get("host", "")
    if host: parts.append(f"agent={host}")
    return " ".join(parts)


# ── WAZUH HANDLER ─────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    "windows":"endpoint","windows_security":"endpoint","windows_system":"endpoint",
    "WEF":"endpoint","win_evt_channel":"endpoint","sysmon":"endpoint",
    "osquery":"endpoint","osquery_result":"endpoint",
    "CrowdStrike":"edr","crowdstrike":"edr",
    "sophos-fw":"firewall","cisco":"firewall","cisco-asa":"firewall",
    "authentication_success":"identity","authentication_failed":"identity",
    "authentication_failures":"identity","invalid_login":"identity",
    "AzureActiveDirectoryStsLogon":"identity","AzureActiveDirectory":"identity",
    "sudo":"identity","pam":"identity","sshd":"identity",
    "syscheck":"fim","syscheck_entry_modified":"fim","syscheck_registry":"fim","rootcheck":"fim",
    "web":"web","accesslog":"web","apache":"web","sqlinjection":"web",
    "vipre":"antivirus",
    "office365":"cloud","amazon":"cloud","aws":"cloud","cloudtrail":"cloud",
    "attack":"threat_detection","attacks":"threat_detection",
    "recon":"threat_detection","ids":"threat_detection",
    "syslog":"system","errors":"system",
}
GENERIC = {"syslog","errors","system_error","ossec","wazuh","local","cron"}

def _get_category(groups):
    for g in groups:
        if g in CATEGORY_MAP and g not in GENERIC: return CATEGORY_MAP[g]
    for g in groups:
        if g in CATEGORY_MAP: return CATEGORY_MAP[g]
    return "unknown"

def handle_wazuh(alert):
    rule = alert.get("rule", {})
    agent = alert.get("agent", {})
    data = alert.get("data", {})
    level = rule.get("level", 0)
    category = _get_category(rule.get("groups", []))
    parts = [f"[{category}]"]
    desc = rule.get("description", "")
    if desc: parts.append(desc)
    parts.append(f"level={level}")
    if agent.get("name"): parts.append(f"agent={agent['name']}")
    for wf, of in [("srcip","src"),("src_ip","src"),("dstip","dst"),("dst_ip","dst"),
                    ("srcport","sport"),("dstport","dport"),("protocol","proto"),
                    ("srcuser","user"),("dstuser","user"),("user","user"),
                    ("action","action"),("url","url"),("process","process"),
                    ("parent_process","parent"),("cmdline","cmdline"),
                    ("direction","direction"),("query","query"),
                    ("query_type","type"),("reply_code","reply"),
                    ("api","api"),("eventName","api")]:
        val = data.get(wf, "")
        if val and of not in [p.split("=")[0] for p in parts if "=" in p]:
            if of in ("process","parent","user"):
                val = str(val).split("\\")[-1].split("/")[-1]
            if of == "url": val = val[:100]
            parts.append(f"{of}={val}")
    return " ".join(parts), category, level


# ── ROUTING ───────────────────────────────────────────────────────────────────

def route_row(row):
    """
    Route a CSV row to the correct handler.
    Checks sourcetype FIRST, then falls back to field detection.
    Returns standardized text or None to skip.
    """
    stype = row.get("sourcetype", "")

    # Skip non-security sourcetypes
    if stype in SKIP_SOURCETYPES:
        return None

    # Route by sourcetype (most reliable)
    if "WinEventLog:Security" in stype:
        return handle_windows(row)
    if "Sysmon" in stype or "sysmon" in stype:
        return handle_sysmon(row)
    if stype == "syslog":
        return handle_linux(row)
    if "cloudtrail" in stype:
        return handle_cloud(row)
    if "cisco:asa" in stype or "cisco_asa" in stype:
        return handle_cisco_asa(row)
    if stype == "osquery:results":
        return handle_osquery(row)
    if "stream:dns" in stype:
        return handle_dns(row)
    if "vpcflow" in stype:
        return handle_vpc_flow(row)
    if "access_combined" in stype or "access_common" in stype:
        return handle_access_log(row)
    if "rds:audit" in stype:
        return handle_rds_audit(row)
    if "symantec" in stype:
        return handle_symantec(row)

    # stream:* family — check if structured fields exist
    if "stream:" in stype:
        src = row.get("src_ip", "")
        if src:
            return handle_network_structured(row)
        else:
            return handle_stream_json(row)

    # Fallback: check populated fields
    if row.get("EventCode", "") and row.get("ComputerName", ""):
        return handle_windows(row)
    if row.get("eventName", "") and row.get("sourceIPAddress", ""):
        return handle_cloud(row)
    if row.get("src_ip", "") and row.get("dest_ip", ""):
        return handle_network_structured(row)

    # Last resort: use _raw if available
    raw = row.get("_raw", "")
    if raw:
        # Try to detect format from content
        if "%ASA-" in raw:
            return handle_cisco_asa(row)
        if raw.strip().startswith("{"):
            return handle_stream_json(row)
        if raw.strip().startswith("<Event"):
            return handle_sysmon(row)
        # Return raw truncated with unknown tag
        return f"[unknown] {raw[:200]}"

    return None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def process_file(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()
    results = []
    skipped = 0
    errors = 0

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    alert = json.loads(line)
                    text, category, level = handle_wazuh(alert)
                    if level >= 4:
                        results.append({"text": text, "source_category": category})
                    else:
                        skipped += 1
                except:
                    errors += 1

    elif ext == ".csv":
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            print(f"Columns: {columns}")

            from collections import Counter
            stype_counts = Counter()

            for row in reader:
                stype = row.get("sourcetype", "unknown")
                try:
                    text = route_row(row)
                    if text is None:
                        skipped += 1
                    else:
                        stype_counts[stype] += 1
                        results.append({"text": text, "source_category": stype})
                except Exception as e:
                    errors += 1

            print(f"\nRouting summary:")
            for s, c in stype_counts.most_common():
                sample = next((r["text"][:80] for r in results if r["source_category"] == s), "")
                print(f"  {s:>45}: {c:>5} | {sample}")

    # Write output
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text","label_int","label","source_category"])
        writer.writeheader()
        for r in results:
            writer.writerow({"text": r["text"], "label_int": -1, "label": "unknown",
                           "source_category": r["source_category"]})

    print(f"\n{'='*60}")
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Processed: {len(results)}")
    print(f"Skipped:   {skipped} (non-security or below threshold)")
    print(f"Errors:    {errors}")

    short = sum(1 for r in results if len(r["text"]) < 15)
    if short:
        print(f"Warning:   {short} alerts with very short text")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize SIEM alerts for SecureBERT")
    parser.add_argument("--input", required=True, help="Input CSV (Splunk) or JSON (Wazuh)")
    parser.add_argument("--output", required=True, help="Output CSV for inference")
    args = parser.parse_args()
    process_file(args.input, args.output)
