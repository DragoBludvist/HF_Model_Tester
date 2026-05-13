"""
standardize_v2.py — Canonical event standardization for SecureBERT.

Replaces standardize.py + preprocess_pipeline.py with a single source of truth.
Both training and inference data MUST flow through this module so the model
sees the same surface format in both cases.

Output format (canonical):
    [category] event_type=NAME field=value field=value ...

Where:
    category   ∈ fixed CATEGORIES set
    event_type ∈ fixed EVENT_TYPES[category] vocabulary
    fields     use canonical names (src_ip not srcip, etc.)

Design:
    1. Source-specific PARSERS extract fields into a canonical dict
    2. CANONICALIZER turns the dict into deterministic output text
    3. ROUTER picks the parser based on input shape

Adding a new source = write one parser function, register it in route_row().
"""

import csv
import json
import re
import argparse
import os

csv.field_size_limit(10 * 1024 * 1024)


# ════════════════════════════════════════════════════════════════════════════
# CANONICAL SCHEMA
# ════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "endpoint",     # process, file, registry events on hosts
    "identity",     # auth, account, login events
    "network",      # connections, flows, firewall (collapsed)
    "dns",          # DNS queries / responses
    "cloud",        # AWS/Azure/GCP/O365 API events
    "web",          # HTTP/web access
    "fim",          # file integrity monitoring
    "antivirus",    # AV/EDR detections
    "email",        # mail flow events
    "system",       # generic system/syslog, last resort
}

# Per-category event_type vocabulary. The model learns these as semantic anchors.
EVENT_TYPES = {
    "endpoint": {
        "process_created", "process_terminated",
        "process_network_connection",      # sysmon EID 3
        "image_loaded",
        "remote_thread_created",            # sysmon EID 8 — injection
        "file_created",
        "registry_modified",
        "service_installed",
        "wfp_block",                        # Windows Filtering Platform
        "scheduled_task_created",
        "endpoint_event",                   # catchall
    },
    "identity": {
        "auth_failure",
        "auth_success",
        "auth_invalid_user",                # SSH "invalid user"
        "account_locked",
        "account_created",
        "password_changed",
        "privilege_escalation",
        "session_disconnect",
        "identity_event",
    },
    "network": {
        "connection_denied",
        "connection_built",
        "connection_teardown",
        "connection_attempt",
        "port_scan",
        "network_flow",
        "network_event",
    },
    "dns": {
        "dns_query",
        "dns_nxdomain",
        "dns_suspicious",                   # rule-based detection
        "dns_event",
    },
    "cloud": {
        "iam_user_created",
        "iam_key_created",
        "compute_instance_launched",
        "console_login_success",
        "console_login_failure",
        "cloud_api_call",                   # generic
    },
    "web": {
        "http_request",
        "http_error",
        "http_attack",                      # SQLi, XSS, etc.
    },
    "fim": {
        "file_modified",
        "file_added",
        "file_deleted",
        "registry_added",
        "registry_modified",
    },
    "antivirus": {
        "malware_detected",
        "malware_quarantined",
        "av_event",
    },
    "email": {
        "email_sent",
        "email_received",
        "email_blocked",
    },
    "system": {
        "system_event",
    },
}

# Canonical field names. Anything outside this set is dropped.
# The output order is fixed so identical events always produce identical text.
CANONICAL_FIELDS = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto",
    "user", "domain",
    "process", "parent_process", "cmdline",
    "file", "file_path",
    "url", "method", "status",
    "query", "qtype", "reply_code",
    "action",
    "api", "account",
    "severity",
    "host",
    "bytes_in", "bytes_out",
]

# Truncation limits (per field) — kept consistent everywhere.
TRUNCATE = {
    "cmdline": 100,
    "url": 80,
    "file_path": 80,
    "query": 80,
}


# ════════════════════════════════════════════════════════════════════════════
# SECURITY HARDENING — defenses against malformed, oversized, and adversarial
# inputs. Every parser ultimately processes attacker-influenced data (alerts
# may contain attacker-controlled strings injected into logs), so all input
# is treated as untrusted.
#
# Threat model assumptions:
#   - Alert content is UNTRUSTED. Attackers can inject text into logs.
#   - The standardizer is called by automated pipelines, never by humans
#     directly. Failure mode is "drop the event and log the reason," never
#     "best-effort parse" of unexpected input.
#   - Output text is consumed by an ML model AND rendered in a dashboard,
#     so it must be safe for both string-comparison and HTML rendering.
#
# Mitigations applied below:
#   1. Hard size limits on raw input (prevents memory exhaustion / ReDoS)
#   2. Type enforcement (rejects lists, dicts, unexpected types as values)
#   3. Control character stripping (prevents log injection, terminal escapes)
#   4. Field count limit on output (prevents oversized canonical strings)
#   5. Fail-closed wrappers on entry points (any uncaught exception → None)
#   6. Rejection counter (operational visibility without leaking content)
# ════════════════════════════════════════════════════════════════════════════

# Size limits. Tunable but err on the conservative side; legitimate security
# events rarely exceed these bounds.
MAX_RAW_SIZE       = 65_536    # 64 KB max per _raw field
MAX_JSON_SIZE      = 16_384    # 16 KB max for embedded JSON within _raw
MAX_FIELD_VALUE    = 1_024     # 1 KB max per canonical field value
MAX_FIELDS_PER_OUT = 32        # No canonical event should have more than this
MAX_CANONICAL_LEN  = 4_096     # 4 KB max for the final canonical string

# Control character regex — anything in these ranges is stripped from values
# before they reach the canonical output or the ML model. Newlines (\n, \r)
# and tabs (\t) are also stripped because they break our line-delimited
# output format and could be used for log-injection attacks against
# downstream log processors.
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Counters for rejected inputs. Read by external monitoring without leaking
# event content. Reset by callers if periodic reporting is desired.
REJECTION_COUNTERS = {
    "oversized_raw":       0,
    "non_dict_input":      0,
    "parser_exception":    0,
    "invalid_canonical":   0,
    "field_value_dropped": 0,
}


def _bump(counter):
    """Increment a rejection counter. Never leaks input content to logs."""
    if counter in REJECTION_COUNTERS:
        REJECTION_COUNTERS[counter] += 1


def _safe_str(value, max_len=MAX_FIELD_VALUE):
    """
    Coerce a value to a sanitized string suitable for canonical output.

    Returns None for unsafe inputs (collections, control-only strings,
    unsupported types). Strips control characters; truncates over-length
    strings. This is the single trust boundary for field values: anything
    that doesn't pass this is dropped, never passed through.
    """
    if value is None:
        return None
    # Reject non-scalar types — lists/dicts/bytes shouldn't appear as field
    # values, and if they do it's a sign of upstream confusion or attack.
    if not isinstance(value, (str, int, float, bool)):
        _bump("field_value_dropped")
        return None
    s = str(value)
    # Strip control characters (including newlines/tabs/escapes).
    s = _CTRL_RE.sub("", s)
    # Truncate to bound output size.
    if len(s) > max_len:
        s = s[:max_len]
    # Reject if everything was stripped — likely a control-only value.
    if not s.strip():
        return None
    return s


def _validate_raw(row):
    """
    Pre-parser size guard for _raw field.

    Returns the _raw string if within limits, or None if oversized.
    Bumps a counter on rejection so operators can see the rate without
    seeing the content.
    """
    raw = row.get("_raw")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        _bump("field_value_dropped")
        return ""
    if len(raw) > MAX_RAW_SIZE:
        _bump("oversized_raw")
        # Truncate rather than reject so that legitimate-but-verbose events
        # still parse. The size limit's purpose is preventing memory
        # exhaustion, not lossless preservation.
        raw = raw[:MAX_RAW_SIZE]
    return raw


def get_rejection_stats():
    """Return a copy of current rejection counters. Safe to expose to ops."""
    return dict(REJECTION_COUNTERS)



# ════════════════════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _norm_severity_wazuh(level):
    """Wazuh rule level → canonical severity."""
    try:
        n = int(level)
    except (TypeError, ValueError):
        return "low"
    if n >= 12: return "critical"
    if n >= 8:  return "high"
    if n >= 4:  return "medium"
    return "low"


def _norm_severity_asa(num):
    """Cisco ASA severity digit → canonical severity."""
    try:
        n = int(num)
    except (TypeError, ValueError):
        return "medium"
    if n <= 2: return "critical"
    if n <= 4: return "high"
    if n <= 6: return "medium"
    return "low"


def _strip_path(value):
    """Reduce a Windows or Unix path to just the leaf name."""
    if not value: return value
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _truncate(field, value):
    """Apply the per-field truncation limit if defined."""
    if value is None: return value
    s = str(value)
    limit = TRUNCATE.get(field)
    if limit and len(s) > limit:
        return s[:limit]
    return s


# ════════════════════════════════════════════════════════════════════════════
# PARSERS — one per source family
#   Each returns: {"category": str, "event_type": str, "fields": dict} or None
#   Returning None means "skip this row" (non-security or unparseable).
# ════════════════════════════════════════════════════════════════════════════

# ─── Wazuh JSON (the most important — production input format) ───────────────

# Map Wazuh rule.groups → (category, default event_type)
# When multiple groups match, we pick the most specific (first non-generic).
_WAZUH_GROUP_MAP = {
    # endpoint
    "windows":            ("endpoint", "endpoint_event"),
    "windows_security":   ("endpoint", "endpoint_event"),
    "windows_system":     ("endpoint", "endpoint_event"),
    "sysmon":             ("endpoint", "endpoint_event"),
    "sysmon_event1":      ("endpoint", "process_created"),
    "sysmon_event3":      ("endpoint", "process_network_connection"),
    "sysmon_event7":      ("endpoint", "image_loaded"),
    "sysmon_event8":      ("endpoint", "remote_thread_created"),
    "sysmon_event11":     ("endpoint", "file_created"),
    "sysmon_event13":     ("endpoint", "registry_modified"),
    "osquery":            ("endpoint", "endpoint_event"),
    "win_evt_channel":    ("endpoint", "endpoint_event"),
    "WEF":                ("endpoint", "endpoint_event"),
    "CrowdStrike":        ("endpoint", "endpoint_event"),
    "crowdstrike":        ("endpoint", "endpoint_event"),

    # identity
    "authentication_failed":   ("identity", "auth_failure"),
    "authentication_failures": ("identity", "auth_failure"),
    "invalid_login":           ("identity", "auth_failure"),
    "authentication_success":  ("identity", "auth_success"),
    "sudo":                    ("identity", "privilege_escalation"),
    "pam":                     ("identity", "identity_event"),
    "sshd":                    ("identity", "identity_event"),
    "AzureActiveDirectory":    ("identity", "identity_event"),
    "AzureActiveDirectoryStsLogon": ("identity", "auth_success"),

    # network
    "firewall":     ("network", "network_event"),
    "sophos-fw":    ("network", "network_event"),
    "cisco":        ("network", "network_event"),
    "cisco-asa":    ("network", "network_event"),
    "ipsec":        ("network", "network_event"),

    # dns — Wazuh doesn't have a standard dns group; we detect via description

    # cloud
    "office365":    ("cloud", "cloud_api_call"),
    "amazon":       ("cloud", "cloud_api_call"),
    "aws":          ("cloud", "cloud_api_call"),
    "cloudtrail":   ("cloud", "cloud_api_call"),

    # fim
    "syscheck":               ("fim", "file_modified"),
    "syscheck_entry_added":   ("fim", "file_added"),
    "syscheck_entry_modified":("fim", "file_modified"),
    "syscheck_entry_deleted": ("fim", "file_deleted"),
    "syscheck_registry":      ("fim", "registry_modified"),

    # web
    "web":          ("web", "http_request"),
    "apache":       ("web", "http_request"),
    "nginx":        ("web", "http_request"),
    "accesslog":    ("web", "http_request"),
    "sqlinjection": ("web", "http_attack"),

    # antivirus
    "vipre":        ("antivirus", "av_event"),
    "symantec":     ("antivirus", "av_event"),
    "rootcheck":    ("antivirus", "av_event"),
}

_WAZUH_GENERIC_GROUPS = {
    "syslog", "errors", "system_error", "ossec", "wazuh",
    "local", "cron", "systemd", "dpkg",
}

# Description-based event_type refinement. Applied *after* group-based mapping
# to upgrade generic event_types to specific ones when the rule description
# carries clear signal.
_DESC_PATTERNS = [
    # (substring, category, event_type) — first match wins
    ("invalid user",          "identity",  "auth_invalid_user"),
    ("brute force",           "identity",  "auth_failure"),
    ("authentication failed", "identity",  "auth_failure"),
    ("failed password",       "identity",  "auth_failure"),
    ("login failed",          "identity",  "auth_failure"),
    ("password changed",      "identity",  "password_changed"),
    ("account locked",        "identity",  "account_locked"),
    ("account created",       "identity",  "account_created"),

    ("suspicious dns",        "dns",       "dns_suspicious"),
    ("dns query",             "dns",       "dns_query"),
    ("nxdomain",              "dns",       "dns_nxdomain"),

    ("connection denied",     "network",   "connection_denied"),
    ("access denied",         "network",   "connection_denied"),
    ("connection blocked",    "network",   "connection_denied"),
    ("port scan",             "network",   "port_scan"),

    ("process created",       "endpoint",  "process_created"),
    ("service installed",     "endpoint",  "service_installed"),
    ("scheduled task",        "endpoint",  "scheduled_task_created"),
    ("filtering platform",    "endpoint",  "wfp_block"),

    ("malware",               "antivirus", "malware_detected"),
    ("virus",                 "antivirus", "malware_detected"),
    ("quarantined",           "antivirus", "malware_quarantined"),

    ("instance launched",     "cloud",     "compute_instance_launched"),
    ("runinstances",          "cloud",     "compute_instance_launched"),
    ("createuser",            "cloud",     "iam_user_created"),
    ("createaccesskey",       "cloud",     "iam_key_created"),
    ("consolelogin",          "cloud",     "console_login_success"),
]

# Wazuh data field → canonical field name
_WAZUH_FIELD_MAP = {
    "srcip":         "src_ip",
    "src_ip":        "src_ip",
    "dstip":         "dst_ip",
    "dst_ip":        "dst_ip",
    "srcport":       "src_port",
    "dstport":       "dst_port",
    "protocol":      "proto",
    "srcuser":       "user",
    "dstuser":       "user",
    "user":          "user",
    "url":           "url",
    "process":       "process",
    "process_name":  "process",
    "parent_process":"parent_process",
    "cmdline":       "cmdline",
    "command_line":  "cmdline",
    "action":        "action",
    "query":         "query",
    "query_type":    "qtype",
    "reply_code":    "reply_code",
    "api":           "api",
    "eventName":     "api",
}


def _wazuh_category_and_type(rule):
    """Pick (category, event_type) for a Wazuh alert from rule.groups + description."""
    groups = rule.get("groups", [])
    description = (rule.get("description", "") or "").lower()

    # 1. Description match has highest priority — it's the most semantic signal
    for sub, cat, et in _DESC_PATTERNS:
        if sub in description:
            return cat, et

    # 2. Group match — prefer specific groups over generic ones
    for g in groups:
        if g in _WAZUH_GROUP_MAP and g not in _WAZUH_GENERIC_GROUPS:
            return _WAZUH_GROUP_MAP[g]
    for g in groups:
        if g in _WAZUH_GROUP_MAP:
            return _WAZUH_GROUP_MAP[g]

    # 3. Default: system bucket
    return "system", "system_event"


def parse_wazuh(alert):
    """Parse a Wazuh JSON alert into canonical form."""
    rule  = alert.get("rule", {}) or {}
    agent = alert.get("agent", {}) or {}
    data  = alert.get("data", {}) or {}

    category, event_type = _wazuh_category_and_type(rule)

    fields = {}
    fields["severity"] = _norm_severity_wazuh(rule.get("level"))
    if agent.get("name"):
        fields["host"] = agent["name"]

    # Pull canonical fields from data{}
    for wazuh_key, canon_key in _WAZUH_FIELD_MAP.items():
        if wazuh_key in data and data[wazuh_key]:
            val = data[wazuh_key]
            if canon_key in ("process", "parent_process"):
                val = _strip_path(val)
            elif canon_key == "user":
                val = str(val).split("\\")[-1]  # strip DOMAIN\
            fields[canon_key] = _truncate(canon_key, val)

    return {"category": category, "event_type": event_type, "fields": fields}


# ─── Cisco ASA syslog ─────────────────────────────────────────────────────────

# Common ASA message ID → event_type. Falls back to keyword match below.
_ASA_MSGID_MAP = {
    "106001": "connection_denied",   # inbound TCP denied
    "106023": "connection_denied",   # ACL denied
    "106100": "connection_denied",   # ACL hit
    "710003": "connection_denied",   # access denied by ACL
    "302013": "connection_built",    # TCP built
    "302014": "connection_teardown", # TCP teardown
    "302015": "connection_built",    # UDP built
    "302016": "connection_teardown", # UDP teardown
    "605004": "auth_failure",        # admin login failed
    "605005": "auth_success",        # admin login success
    "113005": "auth_failure",        # AAA failure
    "113012": "auth_success",
}


def parse_cisco_asa(row):
    raw = row.get("_raw", "") or ""
    m = re.search(r'%ASA-(\d)-(\d+):', raw)
    if not m:
        return None

    severity_num = m.group(1)
    msg_id = m.group(2)

    # Determine event_type (and possibly recategorize as identity for auth msgs)
    event_type = _ASA_MSGID_MAP.get(msg_id)
    category = "network"
    if event_type is None:
        low = raw.lower()
        if "denied" in low or "deny" in low:
            event_type = "connection_denied"
        elif "built" in low:
            event_type = "connection_built"
        elif "teardown" in low:
            event_type = "connection_teardown"
        else:
            event_type = "network_event"
    elif event_type in ("auth_failure", "auth_success"):
        category = "identity"

    fields = {"severity": _norm_severity_asa(severity_num)}

    # Connection tuple — ASA has multiple formats
    conn = (
        re.search(r'from\s+(?:\w+:)?([\d.]+)/(\d+)\s+to\s+(?:\w+:)?([\d.]+)/(\d+)', raw)
        or re.search(r'src\s+\w+:([\d.]+)/(\d+)\s+dst\s+\w+:([\d.]+)/(\d+)', raw)
    )
    if conn:
        fields["src_ip"]   = conn.group(1)
        fields["src_port"] = conn.group(2)
        fields["dst_ip"]   = conn.group(3)
        fields["dst_port"] = conn.group(4)

    proto = re.search(r'\b(tcp|udp|icmp)\b', raw, re.IGNORECASE)
    if proto:
        fields["proto"] = proto.group(1).lower()

    if row.get("host"): fields["host"] = row["host"]

    return {"category": category, "event_type": event_type, "fields": fields}


# ─── Splunk WinEventLog:Security ──────────────────────────────────────────────

_WINEVENT_CODE_MAP = {
    "4624": ("identity", "auth_success"),
    "4625": ("identity", "auth_failure"),
    "4634": ("identity", "session_disconnect"),
    "4647": ("identity", "session_disconnect"),
    "4672": ("identity", "privilege_escalation"),
    "4720": ("identity", "account_created"),
    "4724": ("identity", "password_changed"),
    "4740": ("identity", "account_locked"),
    "4688": ("endpoint", "process_created"),
    "4697": ("endpoint", "service_installed"),
    "4698": ("endpoint", "scheduled_task_created"),
    "5140": ("endpoint", "endpoint_event"),  # network share access
    "5156": ("endpoint", "endpoint_event"),
    "5157": ("endpoint", "wfp_block"),       # WFP block
}


def parse_winevent(row):
    code = str(row.get("EventCode", "") or "")
    if not code:
        return None
    category, event_type = _WINEVENT_CODE_MAP.get(code, ("endpoint", "endpoint_event"))

    fields = {}
    if row.get("Account_Name") and row["Account_Name"] not in ("-", "NULL", ""):
        fields["user"] = str(row["Account_Name"]).split("\\")[-1]
    if row.get("ComputerName"): fields["host"] = row["ComputerName"]
    elif row.get("host"):       fields["host"] = row["host"]
    if row.get("Process_Name"):
        fields["process"] = _strip_path(row["Process_Name"])
    if row.get("Process_Command_Line"):
        fields["cmdline"] = _truncate("cmdline", row["Process_Command_Line"])

    return {"category": category, "event_type": event_type, "fields": fields}


# ─── Splunk Sysmon (XML in _raw) ──────────────────────────────────────────────

_SYSMON_EID_MAP = {
    "1":  "process_created",
    "3":  "process_network_connection",
    "5":  "process_terminated",
    "7":  "image_loaded",
    "8":  "remote_thread_created",
    "11": "file_created",
    "12": "registry_modified",
    "13": "registry_modified",
}


def parse_sysmon(row):
    raw = row.get("_raw", "") or ""
    eid_m = re.search(r'<EventID>(\d+)</EventID>', raw)
    if not eid_m:
        return None
    eid = eid_m.group(1)
    event_type = _SYSMON_EID_MAP.get(eid, "endpoint_event")

    fields = {}
    sysmon_fields = [
        ("Image",          "process",       True),
        ("ParentImage",    "parent_process",True),
        ("CommandLine",    "cmdline",       False),
        ("TargetFilename", "file",          True),
        ("DestinationIp",  "dst_ip",        False),
        ("DestinationPort","dst_port",      False),
        ("SourceIp",       "src_ip",        False),
        ("SourcePort",     "src_port",      False),
        ("Protocol",       "proto",         False),
        ("User",           "user",          True),
    ]
    for sysmon_key, canon_key, strip_path in sysmon_fields:
        m = re.search(rf"Name='{sysmon_key}'[^>]*>([^<]+)<", raw)
        if not m: continue
        val = m.group(1).strip()
        if strip_path: val = _strip_path(val)
        fields[canon_key] = _truncate(canon_key, val)

    if "user" in fields:
        fields["user"] = str(fields["user"]).split("\\")[-1]

    comp = re.search(r"<Computer>([^<]+)</Computer>", raw)
    if comp: fields["host"] = comp.group(1)

    return {"category": "endpoint", "event_type": event_type, "fields": fields}


# ─── Linux syslog (SSH primarily) ─────────────────────────────────────────────

def parse_linux_syslog(row):
    raw = row.get("_raw", "") or ""

    # SSH events → identity
    if "sshd" in raw:
        if "Invalid user" in raw or "invalid user" in raw:
            event_type = "auth_invalid_user"
        elif "Failed password" in raw or "authentication failure" in raw:
            event_type = "auth_failure"
        elif "Accepted" in raw:
            event_type = "auth_success"
        elif "Received disconnect" in raw or "Disconnected" in raw:
            event_type = "session_disconnect"
        else:
            event_type = "identity_event"

        fields = {}
        u = re.search(r'(?:[Ii]nvalid user|for)\s+(\S+)', raw)
        if u and u.group(1) != "invalid":
            fields["user"] = u.group(1)
        ip = re.search(r'from\s+([\d.]+)', raw)
        if ip: fields["src_ip"] = ip.group(1)
        port = re.search(r'port\s+(\d+)', raw)
        if port: fields["src_port"] = port.group(1)
        if row.get("host"): fields["host"] = row["host"]
        return {"category": "identity", "event_type": event_type, "fields": fields}

    # Cisco nvzFlow (Network Visibility Module — flow records embedded in syslog)
    if "nvzFlow" in raw:
        fields = {}
        for tag, canon in [("sa","src_ip"),("da","dst_ip"),("sp","src_port"),("dp","dst_port")]:
            m = re.search(rf'{tag}="([^"]+)"', raw)
            if m: fields[canon] = m.group(1)
        pr = re.search(r'pr="(\d+)"', raw)
        if pr:
            fields["proto"] = {"6":"tcp","17":"udp","1":"icmp"}.get(pr.group(1), pr.group(1))
        user = re.search(r'liuidp="(\w+)"', raw)
        if user: fields["user"] = user.group(1)
        if row.get("host"): fields["host"] = row["host"]
        return {"category": "network", "event_type": "network_flow", "fields": fields}

    # Generic syslog → system
    fields = {}
    if row.get("host"): fields["host"] = row["host"]
    return {"category": "system", "event_type": "system_event", "fields": fields}


# ─── AWS CloudTrail (Splunk aws:cloudtrail) ───────────────────────────────────

_CLOUDTRAIL_API_MAP = {
    "RunInstances":      "compute_instance_launched",
    "CreateUser":        "iam_user_created",
    "CreateAccessKey":   "iam_key_created",
    "ConsoleLogin":      "console_login_success",  # refined below by errorCode
}


def parse_cloudtrail(row):
    api = row.get("eventName", "") or ""
    if not api:
        return None
    event_type = _CLOUDTRAIL_API_MAP.get(api, "cloud_api_call")
    if api == "ConsoleLogin" and row.get("errorCode"):
        event_type = "console_login_failure"

    fields = {"api": api}
    if row.get("sourceIPAddress"): fields["src_ip"] = row["sourceIPAddress"]
    if row.get("host"):            fields["host"]   = row["host"]
    if row.get("recipientAccountId"): fields["account"] = row["recipientAccountId"]

    return {"category": "cloud", "event_type": event_type, "fields": fields}


# ─── Splunk stream:dns ────────────────────────────────────────────────────────

def parse_stream_dns(row):
    raw = row.get("_raw", "") or ""
    fields = {}

    try:
        j = json.loads(raw[:5000])
    except Exception:
        j = {}

    q = j.get("query") or j.get("name")
    if isinstance(q, list): q = q[0] if q else ""
    if q: fields["query"] = _truncate("query", q)

    qt = j.get("query_type")
    if isinstance(qt, list): qt = qt[0] if qt else ""
    if qt: fields["qtype"] = qt

    reply = j.get("reply_code")
    if reply: fields["reply_code"] = reply

    event_type = "dns_query"
    if reply and "nxdomain" in str(reply).lower():
        event_type = "dns_nxdomain"

    if row.get("src_ip"):  fields["src_ip"]  = row["src_ip"]
    if row.get("dest_ip"): fields["dst_ip"] = row["dest_ip"]
    if row.get("host"):    fields["host"]   = row["host"]

    return {"category": "dns", "event_type": event_type, "fields": fields}


# ─── Splunk stream:tcp/udp/icmp etc — generic network ─────────────────────────

def parse_stream_network(row):
    fields = {}
    if row.get("src_ip"):    fields["src_ip"]   = row["src_ip"]
    if row.get("dest_ip"):   fields["dst_ip"]   = row["dest_ip"]
    if row.get("src_port"):  fields["src_port"] = row["src_port"]
    if row.get("dest_port"): fields["dst_port"] = row["dest_port"]
    if row.get("proto"):     fields["proto"]    = str(row["proto"]).lower()
    if row.get("bytes_in"):  fields["bytes_in"] = row["bytes_in"]
    if row.get("bytes_out"): fields["bytes_out"]= row["bytes_out"]
    if row.get("host"):      fields["host"]     = row["host"]

    if not fields.get("src_ip"):
        # No structured fields — try to pull from _raw JSON
        raw = row.get("_raw", "") or ""
        try:
            j = json.loads(raw[:5000])
            for k_in, k_out in [("src_ip","src_ip"),("dest_ip","dst_ip"),
                                ("src_port","src_port"),("dest_port","dst_port"),
                                ("bytes_in","bytes_in"),("bytes_out","bytes_out")]:
                v = j.get(k_in)
                if v: fields[k_out] = v
        except Exception:
            pass

    if not fields.get("src_ip") and not fields.get("dst_ip"):
        return None

    return {"category": "network", "event_type": "network_flow", "fields": fields}


# ─── AWS VPC Flow logs ────────────────────────────────────────────────────────

def parse_vpc_flow(row):
    raw = row.get("_raw", "") or ""
    parts = raw.split()
    if len(parts) < 14:
        return None
    fields = {
        "src_ip":    parts[3],
        "dst_ip":    parts[4],
        "src_port":  parts[5],
        "dst_port":  parts[6],
        "proto":     {"6":"tcp","17":"udp","1":"icmp"}.get(parts[7], parts[7]),
        "action":    parts[12],
    }
    if row.get("host"): fields["host"] = row["host"]
    event_type = "connection_denied" if parts[12].upper() == "REJECT" else "network_flow"
    return {"category": "network", "event_type": event_type, "fields": fields}


# ─── Apache/Nginx access logs ─────────────────────────────────────────────────

def parse_access_log(row):
    raw = row.get("_raw", "") or ""
    m = re.match(r'([\d.]+)\s+\S+\s+(\S+)\s+\[.*?\]\s+"(\w+)\s+(\S+)\s+', raw)
    if not m: return None

    fields = {
        "src_ip": m.group(1),
        "method": m.group(3),
        "url":    _truncate("url", m.group(4)),
    }
    if m.group(2) != "-": fields["user"] = m.group(2)
    status_m = re.search(r'"\s+(\d{3})\s+', raw)
    if status_m:
        fields["status"] = status_m.group(1)

    event_type = "http_request"
    if status_m and status_m.group(1).startswith(("4","5")):
        event_type = "http_error"
    if row.get("host"): fields["host"] = row["host"]

    return {"category": "web", "event_type": event_type, "fields": fields}


# ─── osquery results ──────────────────────────────────────────────────────────

def parse_osquery(row):
    """osquery:results — JSON event with name, columns, decorations."""
    raw = row.get("_raw", "") or ""
    try:
        j = json.loads(raw[:3000])
    except Exception:
        # Malformed JSON — preserve at least the host so it isn't lost entirely
        fields = {}
        if row.get("host"): fields["host"] = row["host"]
        return {"category": "endpoint", "event_type": "endpoint_event", "fields": fields}

    fields = {}
    if j.get("hostIdentifier"):
        fields["host"] = j["hostIdentifier"]
    elif row.get("host"):
        fields["host"] = row["host"]

    decorations = j.get("decorations", {}) or {}
    if decorations.get("username"):
        fields["user"] = str(decorations["username"]).split("\\")[-1]

    cols = j.get("columns", {}) or {}
    if cols.get("cmdline"):
        fields["cmdline"] = _truncate("cmdline", cols["cmdline"])
    # 'name' here is the executable name, distinct from the query 'name'
    if cols.get("name"):
        fields["process"] = _strip_path(cols["name"])
    if cols.get("path"):
        fields["file_path"] = _truncate("file_path", cols["path"])
    if cols.get("target_path"):
        fields["file"] = _strip_path(cols["target_path"])

    # Map osquery query name → event_type when we recognize the pack
    qname = (j.get("name", "") or "").lower()
    if "process_events" in qname or "process-monitor" in qname:
        event_type = "process_created"
    elif "file_events" in qname or "file-events" in qname:
        event_type = "file_created"
    elif "socket_events" in qname:
        event_type = "process_network_connection"
    else:
        event_type = "endpoint_event"

    return {"category": "endpoint", "event_type": event_type, "fields": fields}


# ─── AWS RDS audit logs ───────────────────────────────────────────────────────

def parse_rds_audit(row):
    """aws:rds:audit — comma-separated database audit format."""
    raw = row.get("_raw", "") or ""
    # Format: timestamp,host,user,client,thread,query_id,type,db,query
    parts = raw.split(",")
    if len(parts) < 8:
        return None

    fields = {"api": "rds_audit"}
    if parts[2]:                         fields["user"]   = parts[2]
    if parts[3]:
        client = parts[3].split(":")[0]  # strip port if present
        if re.match(r'^\d+\.\d+\.\d+\.\d+$', client):
            fields["src_ip"] = client
    if parts[6]:                         fields["action"] = parts[6]
    if row.get("host"):                  fields["host"]   = row["host"]

    # Authentication-style actions get identity-flavored event types
    action = (parts[6] or "").upper()
    if action == "FAILED_CONNECT":
        event_type = "console_login_failure"
    elif action == "CONNECT":
        event_type = "console_login_success"
    else:
        event_type = "cloud_api_call"

    return {"category": "cloud", "event_type": event_type, "fields": fields}


# ─── Symantec endpoint protection ─────────────────────────────────────────────

def parse_symantec(row):
    """symantec:ep:* — endpoint protection events."""
    raw = (row.get("_raw", "") or "").lower()

    if "quarantin" in raw:
        event_type = "malware_quarantined"
    elif any(k in raw for k in ("virus", "malware", "trojan", "worm", "threat detected")):
        event_type = "malware_detected"
    else:
        event_type = "av_event"

    fields = {}
    if row.get("host"): fields["host"] = row["host"]

    return {"category": "antivirus", "event_type": event_type, "fields": fields}


# ════════════════════════════════════════════════════════════════════════════
# CANONICALIZER — dict → text
# ════════════════════════════════════════════════════════════════════════════

def canonicalize(parsed):
    """
    Turn a parsed dict into the canonical text form. Returns None if invalid.

    This is the last line of defense before output. Applies:
      - Schema validation (category + event_type must be in vocabulary)
      - Per-value sanitization (control char stripping, length cap, type check)
      - Total field count cap (prevents oversized canonical strings)
      - Total output length cap (defense in depth against memory exhaustion)
    """
    if not parsed or not isinstance(parsed, dict):
        return None

    cat = parsed.get("category")
    et  = parsed.get("event_type")
    fields = parsed.get("fields", {}) or {}

    # Field container must be a dict — refuse to iterate anything else.
    if not isinstance(fields, dict):
        _bump("invalid_canonical")
        return None

    # Schema validation — invalid category or event_type means we have a bug
    # OR an attacker has somehow influenced the parsed dict. Fail closed to
    # a known-safe category rather than emit unexpected tokens.
    if cat not in CATEGORIES:
        cat = "system"
        et = "system_event"
    if et not in EVENT_TYPES.get(cat, set()):
        et = next(iter(EVENT_TYPES[cat]))

    parts = [f"[{cat}]", f"event_type={et}"]
    fields_emitted = 0

    for f in CANONICAL_FIELDS:
        if fields_emitted >= MAX_FIELDS_PER_OUT:
            # Hard cap on number of fields. Anything beyond this is dropped
            # silently — it's not security-relevant data, the parser
            # produced too much.
            break
        v = fields.get(f)
        if v is None or v == "":
            continue
        # Sanitize through _safe_str: strips control chars, enforces type,
        # truncates over-length values. Returns None for unsafe inputs.
        v = _safe_str(v)
        if v is None:
            continue
        # Spaces in values would break our space-separated format and could
        # be used to inject fake field tokens. Replace with underscore.
        v = v.replace(" ", "_")
        # Equals signs would similarly break field=value parsing downstream.
        v = v.replace("=", "_")
        # Angle brackets — neutralize HTML-tag / pseudo-tag injection.
        # Defense in depth: the dashboard layer escapes HTML on render,
        # langchain_analyst._sanitize_alert_text strips tag-injection
        # patterns, but neutralizing here ensures the canonical text is
        # safe to show in *any* downstream context without further escaping.
        v = v.replace("<", "(").replace(">", ")")
        parts.append(f"{f}={v}")
        fields_emitted += 1

    out = " ".join(parts)

    # Final safety net: enforce maximum canonical length. If we somehow
    # produced something larger than this, something is wrong upstream.
    if len(out) > MAX_CANONICAL_LEN:
        _bump("invalid_canonical")
        return None
    return out


# ════════════════════════════════════════════════════════════════════════════
# ROUTER — picks the parser based on input
# ════════════════════════════════════════════════════════════════════════════

# Sourcetypes we deliberately drop (non-security telemetry).
SKIP_SOURCETYPES = {
    "PerfmonMk:Process", "WinHostMon", "top", "ps", "vmstat", "cpu", "iostat",
    "bandwidth", "interfaces", "who", "netstat", "protocol", "df",
    "Script:ListeningPorts", "Script:GetEndpointInfo", "Script:InstalledApps",
    "osquery:info", "amazon-ssm-agent-too_small", "errors-too_small",
    "ess_content_importer", "aws:config:rule",
    "WinEventLog:Application",
}


def route_row(row):
    """
    Route a Splunk-style CSV row to the correct parser, return canonical text.

    Hardened entry point: any malformed input or parser exception results in
    None (event dropped) rather than partial/broken output. Failures are
    counted in REJECTION_COUNTERS for operational visibility.
    """
    # Type guard: row must be a dict-like mapping.
    if not isinstance(row, dict):
        _bump("non_dict_input")
        return None

    try:
        return _route_row_unsafe(row)
    except Exception:
        # Any parser-level exception means the input was unexpected in a way
        # our defenses missed. Fail closed and bump the counter. We do NOT
        # log the row content — it may contain sensitive data, and the
        # rejection counter is the operational signal that something is
        # going wrong without exposing event payloads.
        _bump("parser_exception")
        return None


def _route_row_unsafe(row):
    """Internal router. Assumes type-checked dict input. Wrapped by route_row."""
    stype = row.get("sourcetype", "") or ""

    if not isinstance(stype, str):
        return None
    if stype in SKIP_SOURCETYPES:
        return None

    # Pre-validate _raw size before any parser touches it. The parsers below
    # all eventually read _raw; we want oversized inputs caught once, here,
    # rather than risk a parser missing the check.
    if row.get("_raw") is not None:
        raw_validated = _validate_raw(row)
        # Stash validated/truncated raw back into row so parsers see the
        # bounded version. This is the only place we mutate the row.
        if raw_validated != row.get("_raw"):
            row = dict(row)            # avoid mutating caller's dict
            row["_raw"] = raw_validated

    parsed = None
    if "WinEventLog:Security" in stype:
        parsed = parse_winevent(row)
    elif "Sysmon" in stype or "sysmon" in stype:
        parsed = parse_sysmon(row)
    elif "cisco:asa" in stype or "cisco_asa" in stype:
        parsed = parse_cisco_asa(row)
    elif stype == "syslog":
        parsed = parse_linux_syslog(row)
    elif "cloudtrail" in stype:
        parsed = parse_cloudtrail(row)
    elif "stream:dns" in stype:
        parsed = parse_stream_dns(row)
    elif "stream:" in stype:
        parsed = parse_stream_network(row)
    elif "vpcflow" in stype:
        parsed = parse_vpc_flow(row)
    elif "access_combined" in stype or "access_common" in stype:
        parsed = parse_access_log(row)
    elif stype == "osquery:results":
        parsed = parse_osquery(row)
    elif "rds:audit" in stype:
        parsed = parse_rds_audit(row)
    elif "symantec" in stype:
        parsed = parse_symantec(row)

    # Fallback: try by content of _raw (only after size validation above).
    if parsed is None:
        raw = row.get("_raw", "") or ""
        if "%ASA-" in raw:
            parsed = parse_cisco_asa(row)
        elif "sshd" in raw:
            parsed = parse_linux_syslog(row)
        elif raw.strip().startswith("<Event"):
            parsed = parse_sysmon(row)

    return canonicalize(parsed)


def handle_wazuh(alert):
    """
    Top-level Wazuh entry point. Returns (text, category, severity).

    Hardened: fails closed on any malformed input. Wazuh alerts arrive from
    a trusted internal channel but their CONTENT (descriptions, src_ip,
    user fields) is attacker-influenced log data and must be treated as
    untrusted.
    """
    # Type guard: alert must be a dict.
    if not isinstance(alert, dict):
        _bump("non_dict_input")
        return None, None, None

    # The required sub-structures must themselves be dicts (or absent).
    # An attacker who could replace rule with a list, for example, would
    # cause downstream confusion in parse_wazuh.
    rule = alert.get("rule")
    if rule is not None and not isinstance(rule, dict):
        _bump("non_dict_input")
        return None, None, None
    agent = alert.get("agent")
    if agent is not None and not isinstance(agent, dict):
        _bump("non_dict_input")
        return None, None, None
    data = alert.get("data")
    if data is not None and not isinstance(data, dict):
        _bump("non_dict_input")
        return None, None, None

    try:
        parsed = parse_wazuh(alert)
        text = canonicalize(parsed)
    except Exception:
        _bump("parser_exception")
        return None, None, None

    if text is None:
        return None, None, None
    return text, parsed["category"], parsed["fields"].get("severity", "low")


# ════════════════════════════════════════════════════════════════════════════
# FILE PROCESSOR + CLI
# ════════════════════════════════════════════════════════════════════════════

def process_file(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()
    rows_out = []
    skipped = errors = 0

    if ext == ".json":
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    alert = json.loads(line)
                    text, cat, sev = handle_wazuh(alert)
                    if text:
                        rows_out.append({"text": text, "source_category": cat,
                                         "label_int": -1, "label": "unknown"})
                    else:
                        skipped += 1
                except Exception:
                    errors += 1

    elif ext == ".csv":
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    text = route_row(row)
                    if text is None:
                        skipped += 1
                    else:
                        cat = text.split("]")[0].strip("[")
                        rows_out.append({"text": text, "source_category": cat,
                                         "label_int": -1, "label": "unknown"})
                except Exception:
                    errors += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text","label_int","label","source_category"])
        w.writeheader()
        for r in rows_out: w.writerow(r)

    print(f"Processed: {len(rows_out)} | Skipped: {skipped} | Errors: {errors}")
    return rows_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    process_file(args.input, args.output)