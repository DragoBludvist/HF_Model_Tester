"""
preprocess_pipeline.py — Wazuh alert preprocessing for SecureBERT classification.

Two input paths:
  1. Production (Wazuh JSON) -> preprocess_alert() -> standardized text
  2. Training (BOTS v3 CSV)  -> adapt_botsv3()     -> same standardized text

Output format: [category] description key=value key=value ... | raw_log_excerpt
"""

import re
import json

CATEGORY_MAP = {
    "windows": "endpoint", "windows_security": "endpoint",
    "windows_system": "endpoint", "windows_application": "endpoint",
    "WEF": "endpoint", "win_evt_channel": "endpoint",
    "sysmon": "endpoint", "windows_logs": "endpoint",
    "CrowdStrike": "edr", "crowdstrike": "edr",
    "sophos-fw": "firewall", "cisco": "firewall",
    "cisco-asa": "firewall", "asa": "firewall", "ipsec": "firewall",
    "authentication_success": "identity", "authentication_failed": "identity",
    "authentication_failures": "identity", "invalid_login": "identity",
    "AzureActiveDirectoryStsLogon": "identity", "AzureActiveDirectory": "identity",
    "login_time": "identity", "login_day": "identity",
    "account_changed": "identity", "access_denied": "identity",
    "sudo": "identity", "pam": "identity", "sshd": "identity",
    "syscheck": "fim", "syscheck_registry": "fim",
    "syscheck_entry_modified": "fim", "syscheck_entry_added": "fim",
    "syscheck_entry_deleted": "fim", "syscheck_file": "fim", "rootcheck": "fim",
    "web": "web", "accesslog": "web", "web_scan": "web",
    "apache": "web", "nginx": "web", "appsec": "web", "sqlinjection": "web",
    "vipre": "antivirus",
    "office365": "cloud_saas", "Office365": "cloud_saas",
    "SharePoint": "cloud_saas", "SharePointFileOperation": "cloud_saas",
    "SharePointSharingOperation": "cloud_saas", "SharePointListOperation": "cloud_saas",
    "MicrosoftTeams": "cloud_saas", "OneDrive": "cloud_saas",
    "ExchangeItem": "email", "ExchangeItemAggregated": "email",
    "Exchange": "email", "ExchangeItemGroup": "email",
    "Yammer": "cloud_saas", "CRM": "cloud_saas",
    "attack": "threat_detection", "attacks": "threat_detection",
    "recon": "threat_detection", "ids": "threat_detection",
    "firewall": "network",
    "syslog": "system", "errors": "system", "system_error": "system",
    "ossec": "wazuh_internal", "wazuh": "wazuh_internal",
    "DLPEndpoint": "dlp", "ComplianceDLPExchange": "dlp",
    "SensitivityLabeledFileAction": "dlp", "SensitivityLabelAction": "dlp",
    "MIPLabel": "dlp", "ComplianceDLPSharePoint": "dlp",
    "SecurityComplianceAlerts": "compliance",
    "policy_changed": "policy", "policy_violation": "policy",
}

PROCESS_CATEGORIES = {
    "endpoint", "edr", "firewall", "identity", "fim", "web",
    "antivirus", "cloud_saas", "email", "system", "network", "threat_detection",
}
AUTO_ESCALATE = {"threat_detection"}
SKIP_ML = {"dlp", "compliance", "policy", "wazuh_internal"}
MIN_RULE_LEVEL = 4


# ── WAZUH PATH ────────────────────────────────────────────────────────────────

def get_category(groups):
    generic = {"syslog", "errors", "system_error", "ossec", "wazuh",
               "local", "cron", "systemd", "dpkg"}
    for g in groups:
        if g in CATEGORY_MAP and g not in generic:
            return CATEGORY_MAP[g]
    for g in groups:
        if g in CATEGORY_MAP:
            return CATEGORY_MAP[g]
    return "unknown"


def standardize_wazuh(alert):
    rule = alert.get("rule", {})
    agent = alert.get("agent", {})
    data = alert.get("data", {})
    level = rule.get("level", 0)

    if level < MIN_RULE_LEVEL:
        return None, "filtered", False, f"Level {level} < {MIN_RULE_LEVEL}"

    category = get_category(rule.get("groups", []))

    if category in SKIP_ML:
        return None, category, False, f"'{category}' handled by Phase 1"

    parts = [f"[{category}]"]
    desc = rule.get("description", "")
    if desc: parts.append(desc)
    parts.append(f"level={level}")

    agent_name = agent.get("name", "")
    if agent_name: parts.append(f"agent={agent_name}")

    src = data.get("srcip", "") or data.get("src_ip", "")
    dst = data.get("dstip", "") or data.get("dst_ip", "")
    if src: parts.append(f"src={src}")
    if dst: parts.append(f"dst={dst}")

    sport = data.get("srcport", "") or data.get("src_port", "")
    dport = data.get("dstport", "") or data.get("dst_port", "")
    if sport or dport: parts.append(f"port={sport or '?'}->{dport or '?'}")

    proto = data.get("protocol", "")
    if proto: parts.append(f"proto={proto}")

    user = data.get("srcuser", "") or data.get("dstuser", "") or data.get("user", "")
    if user: parts.append(f"user={user}")

    action = data.get("action", "")
    if action: parts.append(f"action={action}")

    url = data.get("url", "")
    if url: parts.append(f"url={url[:100]}")

    full_log = alert.get("full_log", "")
    if full_log:
        parts.append(f"| {re.sub(r'[ ]+', ' ', full_log)[:400]}")

    return " ".join(parts), category, category in AUTO_ESCALATE, None


def preprocess_alert(alert):
    text, category, auto_esc, skip_reason = standardize_wazuh(alert)
    if text is None:
        return {"skip": True, "skip_reason": skip_reason,
                "category": category, "auto_escalate": False}
    return {
        "skip": False, "skip_reason": None,
        "auto_escalate": auto_esc, "category": category,
        "securebert_input": text,
        "metadata": {
            "rule_id": alert.get("rule", {}).get("id", ""),
            "rule_level": alert.get("rule", {}).get("level", 0),
            "rule_description": alert.get("rule", {}).get("description", ""),
            "agent_name": alert.get("agent", {}).get("name", ""),
            "mitre": alert.get("rule", {}).get("mitre", {}),
            "timestamp": alert.get("timestamp", ""),
        },
    }


def preprocess_batch(alerts):
    processed, auto_escalated, skipped = [], [], []
    for a in alerts:
        r = preprocess_alert(a)
        if r["skip"]: skipped.append(r)
        elif r["auto_escalate"]:
            auto_escalated.append(r)
            processed.append(r)
        else: processed.append(r)
    return {"processed": processed, "auto_escalated": auto_escalated,
            "skipped": skipped,
            "stats": {"total": len(alerts), "processed": len(processed),
                      "auto_escalated": len(auto_escalated), "skipped": len(skipped)}}


# ── BOTS v3 PATH ─────────────────────────────────────────────────────────────

def adapt_botsv3(text, source_category):
    if source_category == "windows": return _std_windows(text)
    elif source_category == "linux": return _std_linux(text)
    elif source_category == "network": return _std_network(text)
    elif source_category == "cloud": return _std_cloud(text)
    elif source_category == "endpoint": return _std_endpoint(text)
    elif source_category == "dns": return _std_dns(text)
    return f"[{source_category}] {text[:200]}"


def _std_windows(text):
    text = re.sub(r'^\[windows\]\s*', '', text)
    n = text.replace('\\n', ' ').replace('\\t', ' ')
    n = re.sub(r'\s{2,}', ' ', n)
    desc = n.split('.')[0].strip()
    if len(desc) < 10: desc = '. '.join(n.split('.')[:2]).strip()
    parts = [f"[endpoint] {desc[:100]}."]

    acc = re.search(r'Account Name:\s*(\S+)', n)
    if acc and acc.group(1) not in ('-', 'NULL', 'NULL SID'):
        parts.append(f"account={acc.group(1)}")

    sid = re.search(r'Security ID:\s*([\w\\\\]+)', n)
    if sid:
        s = sid.group(1)
        if 'SYSTEM' in s: parts.append("sid=SYSTEM")
        elif 'AzureAD' in s: parts.append(f"sid=AzureAD\\{s.split(chr(92))[-1]}")

    for pat in [r'Application Name:\s*(\S+)', r'Process Name:\s*(\S+)',
                r'New Process Name:\s*(\S+)']:
        m = re.search(pat, n)
        if m:
            exe = m.group(1).split('\\\\')[-1].split('\\')[-1]
            if exe and exe not in ('-', 'Name:'): parts.append(f"process={exe}"); break

    d = re.search(r'Direction:\s*(\w+)', n)
    if d: parts.append(f"direction={d.group(1)}")
    src = re.search(r'Source Address:\s*([\d.:a-fA-F]+)', n)
    dst = re.search(r'Destination Address:\s*([\d.:a-fA-F]+)', n)
    sp = re.search(r'Source Port:\s*(\d+)', n)
    dp = re.search(r'Destination Port:\s*(\d+)', n)
    proto = re.search(r'Protocol:\s*(\d+)', n)
    if src: parts.append(f"src={src.group(1)}")
    if dst: parts.append(f"dst={dst.group(1)}")
    if sp and dp: parts.append(f"port={sp.group(1)}->{dp.group(1)}")
    if proto:
        pmap = {'6': 'tcp', '17': 'udp'}
        parts.append(f"proto={pmap.get(proto.group(1), proto.group(1))}")

    if 'Scriptblock' in text or 'powershell' in text.lower():
        parts.append("powershell=encoded" if '-enc ' in text else "powershell=yes")
    svc = re.search(r'Service Name:\s*(\S+)', n)
    if svc and svc.group(1) != '-': parts.append(f"service={svc.group(1)}")
    return " ".join(parts)


def _std_linux(text):
    if "sshd" in text:
        parts = ["[identity]"]
        if "Invalid user" in text or "invalid user" in text:
            parts.append("SSH invalid user attempt.")
            u = re.search(r'[Ii]nvalid user\s+(\S+)', text)
            if u: parts.append(f"user={u.group(1)}")
        elif "Failed password" in text:
            parts.append("SSH failed password.")
            u = re.search(r'for\s+(\S+)', text)
            if u and u.group(1) != 'invalid': parts.append(f"user={u.group(1)}")
        elif "Accepted" in text: parts.append("SSH login accepted.")
        elif "Received disconnect" in text: parts.append("SSH disconnect.")
        else: parts.append("SSH event.")
        ip = re.search(r'from\s+([\d.]+)', text)
        if ip: parts.append(f"src={ip.group(1)}")
        port = re.search(r'port\s+(\d+)', text)
        if port: parts.append(f"sport={port.group(1)}")
        host = re.search(r'^\w+\s+\d+\s+[\d:]+\s+(\S+)', text)
        if host: parts.append(f"agent={host.group(1)}")
        return " ".join(parts)

    if "nvzFlow" in text:
        parts = ["[firewall] Network flow."]
        for tag, key in [('sa','src'),('da','dst'),('sp','sport'),('dp','dport')]:
            m = re.search(rf'{tag}="([^"]+)"', text)
            if m: parts.append(f"{key}={m.group(1)}")
        pr = re.search(r'pr="(\d+)"', text)
        if pr:
            pmap = {'6': 'tcp', '17': 'udp', '1': 'icmp'}
            parts.append(f"proto={pmap.get(pr.group(1), pr.group(1))}")
        user = re.search(r'liuidp="(\w+)"', text)
        if user: parts.append(f"user={user.group(1)}")
        return " ".join(parts)

    parts = ["[system]"]
    host = re.search(r'^\w+\s+\d+\s+[\d:]+\s+(\S+)', text)
    if host: parts.append(f"agent={host.group(1)}")
    proc = re.search(r'(\S+)\[\d+\]:', text)
    if proc: parts.append(f"process={proc.group(1)}")
    msg = re.search(r'\[\d+\]:\s*(.+)', text)
    if msg: parts.append(msg.group(1).strip()[:150])
    return " ".join(parts)


def _std_network(text):
    parts = ["[firewall]"]
    proto = re.search(r'stream:(\w+)', text)
    parts.append(f"{proto.group(1).upper()} connection." if proto else "Network connection.")
    conn = re.search(r'from\s+([\d.]+):?(\d+)?\s+to\s+([\d.]+):?(\d+)?', text)
    if conn:
        parts.append(f"src={conn.group(1)}")
        parts.append(f"dst={conn.group(3)}")
        s, d = conn.group(2), conn.group(4)
        if s and d: parts.append(f"port={s}->{d}")
        elif d: parts.append(f"dport={d}")
    for f in ['bytes_in', 'bytes_out']:
        m = re.search(rf'{f}=(\d+)', text)
        if m: parts.append(f"{f}={m.group(1)}")
    host = re.search(r'on host\s+(\S+)', text)
    if host: parts.append(f"agent={host.group(1)}")
    return " ".join(parts)


def _std_cloud(text):
    text = re.sub(r'^\[cloud\]\s*', '', text)
    parts = ["[cloud]"]
    api = re.search(r':\s*(\w+)\s*\|', text)
    if api: parts.append(f"api={api.group(1)}")
    ip = re.search(r'from\s+([\d.]+)', text)
    if ip: parts.append(f"src={ip.group(1)}")
    host = re.search(r'on host\s+(\S+)', text)
    if host: parts.append(f"agent={host.group(1)}")
    return " ".join(parts)


def _std_endpoint(text):
    text = re.sub(r'^\[endpoint\]\s*', '', text)
    if text.startswith("<Event") or "<EventID>" in text:
        parts = ["[endpoint]"]
        eid = re.search(r'<EventID>(\d+)</EventID>', text)
        if eid:
            m = {'1':'Process created','3':'Network connection','5':'Process terminated',
                 '8':'CreateRemoteThread','11':'File created','12':'Registry created/deleted',
                 '13':'Registry value set'}
            parts.append(f"Sysmon: {m.get(eid.group(1), f'EventID={eid.group(1)}')}")
        for field, key in [('Image','process'),('TargetFilename','file'),
                           ('DestinationIp','dst'),('DestinationPort','dport'),
                           ('SourceIp','src'),('User','user'),('ParentImage','parent')]:
            m = re.search(rf"Name='{field}'[^>]*>([^<]+)<", text)
            if m: parts.append(f"{key}={m.group(1).strip().split(chr(92))[-1]}")
        comp = re.search(r"<Computer>([^<]+)</Computer>", text)
        if comp: parts.append(f"agent={comp.group(1)}")
        return " ".join(parts)

    if text.startswith("{"):
        parts = ["[endpoint]"]
        try:
            j = json.loads(text[:3000])
            name = j.get("name", "")
            if name:
                short = name.replace("pack_","").replace("process-monitoring_","").replace("incident-response_","IR:")
                parts.append(f"osquery: {short}")
            host = j.get("hostIdentifier", "")
            if host: parts.append(f"agent={host}")
            user = j.get("decorations", {}).get("username", "")
            if user: parts.append(f"user={user}")
            cols = j.get("columns", {})
            for k in ["cmdline","path","name","target_path"]:
                if k in cols and cols[k]:
                    v = str(cols[k])
                    if '/' in v: v = v.split('/')[-1]
                    elif '\\' in v: v = v.split('\\')[-1]
                    parts.append(f"{k}={v[:60]}")
        except: parts.append("osquery: parse_error")
        return " ".join(parts)
    return f"[endpoint] {text[:200]}"


def _std_dns(text):
    text = re.sub(r'^\[dns\]\s*', '', text)
    parts = ["[dns]"]
    try:
        j = json.loads(text[:2000])
        q = j.get("query", "") or j.get("name", "")
        if isinstance(q, list): q = q[0] if q else ""
        if q: parts.append(f"query={q}")
        qt = j.get("query_type", "")
        if isinstance(qt, list): qt = qt[0] if qt else ""
        if qt: parts.append(f"type={qt}")
        r = j.get("reply_code", "")
        if r: parts.append(f"reply={r}")
        h = j.get("host_addr", "") or j.get("src_ip", "") or j.get("dest_ip", "")
        if isinstance(h, list): h = h[0] if h else ""
        if h: parts.append(f"host={h}")
    except: parts.append(text[:150])
    return " ".join(parts)


if __name__ == "__main__":
    tests = [
        {"timestamp": "2025-03-05T10:15:30.000+0000",
         "rule": {"level": 10, "description": "sshd: Multiple auth failures.",
                  "id": "5720", "groups": ["syslog", "sshd", "authentication_failed"]},
         "agent": {"name": "linux-prod-01"}, "decoder": {"name": "sshd"},
         "data": {"srcip": "203.0.113.50", "srcuser": "admin"},
         "full_log": "Failed password for invalid user admin from 203.0.113.50"},
        {"rule": {"level": 8, "description": "Sophos: Connection denied.",
                  "groups": ["sophos-fw"]},
         "agent": {"name": "fw-01"},
         "data": {"srcip": "10.0.0.50", "dstip": "203.0.113.25", "dstport": "443",
                  "protocol": "TCP", "action": "Denied"}, "full_log": ""},
        {"rule": {"level": 3, "groups": ["syslog"]}, "data": {}},
        {"rule": {"level": 5, "groups": ["DLPEndpoint"]}, "data": {}},
    ]

    print("=" * 60)
    print("PREPROCESSOR TEST (SecureBERT only)")
    print("=" * 60)
    for a in tests:
        r = preprocess_alert(a)
        if r["skip"]: print(f"  SKIP: {r['skip_reason']}")
        else: print(f"  [{r['category']}]: {r['securebert_input'][:100]}")

    print("\nBOTS v3:")
    for cat, txt in [("network", "Network event (stream:tcp) | from 217.61.6.175:34152 to 172.16.0.178:22 | bytes_in=54 | on host gx"),
                     ("linux", "Aug 20 11:57:37 srv sshd[9926]: Invalid user joyce from 117.240.199.84 port 44452"),
                     ("cloud", "Cloud event (aws:cloudtrail): RunInstances | from 139.198.18.205 | on host splunk.froth.ly")]:
        print(f"  [{cat}]: {adapt_botsv3(txt, cat)}")
    print("=" * 60)
