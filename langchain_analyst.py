"""
langchain_analyst.py — SOC Analyst AI Assistant.

Takes SecureBERT classification results and generates analyst-ready reports.
Uses LangChain to orchestrate the LLM call with structured prompts.

Usage:
    from langchain_analyst import analyze_alert
    report = analyze_alert(alert_text, prediction, confidence)

Requirements:
    pip install langchain langchain-anthropic
"""

import os
from typing import Optional

# ── LLM SETUP ─────────────────────────────────────────────────────────────────
# Supports: Claude API, or falls back to a simple prompt template for offline use

SYSTEM_PROMPT = """You are a senior SOC analyst at Navitas Life Sciences reviewing security alerts.
You receive alerts that have been classified by an ML model (SecureBERT) as threat or benign.

For each alert, provide a concise analyst report with exactly these sections:

SUMMARY: One sentence — what happened.
ATTACK TYPE: The likely MITRE ATT&CK technique (e.g., T1110 Brute Force) or "N/A" if benign.
SEVERITY: Critical / High / Medium / Low / Informational.
ANALYSIS: 2-3 sentences explaining why this is or isn't a threat. Reference specific fields from the alert.
RECOMMENDED ACTIONS: 2-3 numbered steps the analyst should take.

Rules:
- Be direct and actionable. No filler.
- Reference actual IPs, ports, process names, and usernames from the alert.
- If the ML model confidence is below 70%, note the uncertainty.
- If classified as benign, still briefly explain what the alert is and why it's safe.
- Keep the total response under 150 words."""

def _build_prompt(alert_text, prediction, confidence, analyst_question=None):
    """Build the user prompt from alert data."""
    parts = [
        f"Alert classified as: {prediction.upper()} ({confidence:.1%} confidence)",
        f"Alert content: {alert_text}",
    ]

    if analyst_question:
        parts.append(f"\nAnalyst follow-up question: {analyst_question}")
        parts.append("Answer the question based on the alert data above.")

    return "\n".join(parts)


# ── LANGCHAIN PATH (when langchain + API key available) ───────────────────────

def _analyze_with_langchain(alert_text, prediction, confidence, analyst_question=None, api_key=None):
    """Use LangChain + Claude API for analysis."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=500,
        temperature=0,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(alert_text, prediction, confidence, analyst_question)),
    ]

    response = llm.invoke(messages)
    return response.content


# ── DIRECT API PATH (when langchain not installed) ────────────────────────────

def _analyze_with_api(alert_text, prediction, confidence, analyst_question=None, api_key=None):
    """Direct Anthropic API call — no LangChain dependency."""
    import json
    import urllib.request

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _analyze_offline(alert_text, prediction, confidence)

    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 500,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_prompt(alert_text, prediction, confidence, analyst_question)}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data["content"][0]["text"]


# ── OFFLINE PATH (no API key — rule-based fallback) ───────────────────────────

MITRE_PATTERNS = {
    "ssh": ("T1110", "Brute Force", "Credential Access"),
    "invalid user": ("T1110", "Brute Force", "Credential Access"),
    "failed password": ("T1110", "Brute Force", "Credential Access"),
    "brute": ("T1110", "Brute Force", "Credential Access"),
    "cmd.exe": ("T1059.003", "Windows Command Shell", "Execution"),
    "powershell": ("T1059.001", "PowerShell", "Execution"),
    "wmic": ("T1047", "WMI", "Execution"),
    "createremotethread": ("T1055", "Process Injection", "Defense Evasion"),
    "process created": ("T1059", "Command and Scripting Interpreter", "Execution"),
    "service installed": ("T1543.003", "Windows Service", "Persistence"),
    "denied": ("T1190", "Exploit Public-Facing Application", "Initial Access"),
    "connection denied": ("TA0001", "Network Scanning", "Reconnaissance"),
    "runinstances": ("T1578", "Modify Cloud Compute Infrastructure", "Defense Evasion"),
    "createuser": ("T1136", "Create Account", "Persistence"),
    "createaccesskey": ("T1098", "Account Manipulation", "Persistence"),
    "consolelogin": ("T1078", "Valid Accounts", "Initial Access"),
    "port scan": ("T1046", "Network Service Scanning", "Discovery"),
    "nxdomain": ("T1071", "Application Layer Protocol", "Command and Control"),
    "file created": ("T1105", "Ingress Tool Transfer", "Command and Control"),
    "registry": ("T1112", "Modify Registry", "Defense Evasion"),
}

SEVERITY_MAP = {
    "T1110": "High",
    "T1059": "High", "T1059.001": "Critical", "T1059.003": "High",
    "T1047": "High",
    "T1055": "Critical",
    "T1543.003": "High",
    "T1190": "Critical",
    "T1578": "High",
    "T1136": "High", "T1098": "High",
    "T1078": "High",
    "T1046": "Medium",
    "T1071": "Medium",
    "T1105": "High",
    "T1112": "Medium",
    "TA0001": "Medium",
}

def _analyze_offline(alert_text, prediction, confidence):
    """Rule-based analysis when no API is available."""
    lower = alert_text.lower()

    # Match MITRE technique
    technique_id, technique_name, tactic = "N/A", "N/A", "N/A"
    for pattern, (tid, tname, tac) in MITRE_PATTERNS.items():
        if pattern in lower:
            technique_id, technique_name, tactic = tid, tname, tac
            break

    severity = SEVERITY_MAP.get(technique_id, "Informational" if prediction == "benign" else "Medium")

    # Extract key fields
    fields = {}
    import re
    for match in re.finditer(r'(\w+)=([\w.:/-]+)', alert_text):
        fields[match.group(1)] = match.group(2)

    # Build summary
    category = alert_text.split("]")[0].strip("[") if "]" in alert_text else "unknown"

    if prediction == "threat":
        summary = f"{tactic} activity detected in {category} alert."
        if "src" in fields:
            summary = f"{tactic} activity from {fields['src']} detected in {category} alert."

        analysis = f"SecureBERT classified this as threat with {confidence:.1%} confidence."
        if "src" in fields and "dst" in fields:
            analysis += f" Connection from {fields['src']} to {fields.get('dst', 'unknown')}."
        if "process" in fields:
            analysis += f" Process {fields['process']} is associated with attack tooling."
        if "user" in fields:
            analysis += f" Targeted user: {fields['user']}."

        actions = []
        if "src" in fields:
            actions.append(f"Block source IP {fields['src']} at the firewall.")
        if "user" in fields:
            actions.append(f"Verify account '{fields['user']}' for unauthorized access.")
        if "agent" in fields:
            actions.append(f"Check host {fields['agent']} for signs of compromise.")
        if not actions:
            actions = ["Investigate the alert manually.", "Check related alerts in the timeframe."]
        actions.append("Document findings in the incident tracker.")

    else:
        summary = f"Normal {category} activity. No threat indicators."
        analysis = f"SecureBERT classified this as benign with {confidence:.1%} confidence. "
        analysis += "No known attack patterns match this alert."
        severity = "Informational"
        technique_id, technique_name = "N/A", "N/A"
        actions = ["No action required.", "Continue monitoring."]

    report = f"SUMMARY: {summary}\n"
    report += f"ATTACK TYPE: {technique_id} {technique_name}\n"
    report += f"SEVERITY: {severity}\n"
    report += f"ANALYSIS: {analysis}\n"
    report += f"RECOMMENDED ACTIONS:\n"
    for i, a in enumerate(actions[:3], 1):
        report += f"  {i}. {a}\n"

    return report


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def analyze_alert(alert_text, prediction, confidence, analyst_question=None, api_key=None):
    """
    Analyze an alert using the best available method.

    Tries in order:
      1. LangChain + Claude API (best quality)
      2. Direct Claude API call (no LangChain dependency)
      3. Offline rule-based analysis (no API needed)

    Args:
        alert_text: standardized alert text
        prediction: "threat" or "benign"
        confidence: float 0.0-1.0
        analyst_question: optional follow-up question
        api_key: optional API key override

    Returns:
        str: analyst report
    """
    # Try LangChain first
    try:
        return _analyze_with_langchain(alert_text, prediction, confidence, analyst_question, api_key)
    except ImportError:
        pass
    except Exception as e:
        print(f"LangChain failed: {e}")

    # Try direct API
    try:
        return _analyze_with_api(alert_text, prediction, confidence, analyst_question, api_key)
    except Exception as e:
        print(f"API failed: {e}")

    # Offline fallback
    return _analyze_offline(alert_text, prediction, confidence)


# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_alerts = [
        ("[identity] SSH invalid user attempt. user=joyce src=117.240.199.84 sport=44452 agent=ip-172-31-12-76",
         "threat", 0.97),
        ("[endpoint] Process created. process=cmd.exe parent=cmd.exe user=SYSTEM agent=BSTOLL-L.froth.ly",
         "threat", 0.99),
        ("[firewall] Connection denied. src=185.220.101.45 dst=10.0.0.50 dport=3389 proto=TCP action=Denied",
         "threat", 0.85),
        ("[cloud] api=RunInstances src=139.198.18.205 agent=splunk.froth.ly",
         "threat", 0.95),
        ("[firewall] UDP connection. src=10.0.0.50 dst=8.8.8.8 dport=53 proto=udp",
         "benign", 0.99),
        ("[dns] query=splunk.froth.ly type=A reply=NoError src=172.16.133.131",
         "benign", 0.98),
    ]

    print("=" * 60)
    print("LANGCHAIN ANALYST — TEST (offline mode)")
    print("=" * 60)

    for alert, pred, conf in test_alerts:
        print(f"\n{'─' * 60}")
        print(f"Alert: {alert}")
        print(f"Classification: {pred} ({conf:.0%})")
        print()
        report = analyze_alert(alert, pred, conf)
        print(report)
