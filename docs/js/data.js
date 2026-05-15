/* ============================================================
   NAVITAS SOC // DATA
   ------------------------------------------------------------
   Section index:
     1. Mock alert templates (used when no file is loaded)
     2. Source registry + mock counts
     3. extractAlerts()  — find the alerts array in any JSON shape
     4. normalizeAlert() — map arbitrary field names to our schema
     5. Field-name dictionaries for flexible matching
   ============================================================ */


/* ============ 1. MOCK ALERT TEMPLATES ============ */
const ALERT_TEMPLATES = [
  { source: 'CrowdStrike', cls: 'threat', conf: 0.998, mitre: 'T1059.001',
    title: 'Encoded PowerShell command executed by user jsmith',
    host: 'EXEC-23', user: 'jsmith', sev: 12,
    raw: 'powershell.exe -nop -w hidden -enc JABjAGwAaQBlAG4AdAA9AE4AZQB3AC0ATwBiAGoAZQBjAHQA...' },
  { source: 'Trellix EDR', cls: 'threat', conf: 0.997, mitre: 'T1003.001',
    title: 'Credential dumping attempt on LSASS.exe',
    host: 'DC-01', user: 'SYSTEM', sev: 14,
    raw: 'process=mimikatz.exe parent=cmd.exe user=SYSTEM hash=a1b2c3d4...' },
  { source: 'Wazuh', cls: 'threat', conf: 0.994, mitre: 'T1110.001',
    title: 'Multiple failed SSH logins from TOR exit node',
    host: 'jump-01', user: 'root', sev: 10,
    raw: 'src_ip=185.220.101.45 attempts=23 result=failed user=root' },
  { source: 'CrowdStrike', cls: 'threat', conf: 0.987, mitre: 'T1055',
    title: 'Process injection: explorer.exe -> rundll32.exe',
    host: 'WS-2034', user: 'pjohnson', sev: 13,
    raw: 'parent=explorer.exe child=rundll32.exe technique=process_injection' },
  { source: 'Purview DLP', cls: 'threat', conf: 0.964, mitre: 'T1567.002',
    title: 'PHI document uploaded to personal Gmail address',
    host: 'WS-1842', user: 'akumar', sev: 11,
    raw: 'destination=mail.google.com classification=PHI file=patient_records.xlsx' },
  { source: 'CloudTrail', cls: 'threat', conf: 0.999, mitre: 'T1098',
    title: 'Root account API key created',
    host: 'aws-prod', user: 'root', sev: 15,
    raw: 'eventName=CreateAccessKey userIdentity.type=Root sourceIPAddress=104.x.x.x' },
  { source: 'Wazuh', cls: 'threat', conf: 0.991, mitre: 'T1021.002',
    title: 'Lateral movement: PsExec to DC-01',
    host: 'WS-1908', user: 'svc_backup', sev: 13,
    raw: 'tool=psexec.exe target=DC-01 service_install=true' },
  { source: 'Trellix EDR', cls: 'threat', conf: 0.989, mitre: 'T1071.004',
    title: 'Outbound DNS to known C2 domain *.evilcorp.tld',
    host: 'EXEC-09', user: 'mrodriguez', sev: 12,
    raw: 'query=api.evilcorp.tld type=A reputation=malicious_c2' },
  { source: 'Netskope', cls: 'threat', conf: 0.953, mitre: 'T1567',
    title: 'Bulk file download to unsanctioned cloud storage',
    host: 'WS-3201', user: 'tlee', sev: 11,
    raw: 'destination=mega.nz file_count=247 total_size=1.2GB' },
  { source: 'CrowdStrike', cls: 'threat', conf: 0.974, mitre: 'T1047',
    title: 'WMI command-line execution from suspicious parent',
    host: 'EXEC-15', user: 'rgupta', sev: 12,
    raw: 'wmic process call create cmd.exe /c whoami parent=winword.exe' },

  { source: 'Wazuh', cls: 'benign', conf: 0.981, mitre: null,
    title: 'SSH login from authorized user (corp VPN)',
    host: 'jump-02', user: 'mthomas', sev: 5,
    raw: 'src_ip=10.118.4.22 user=mthomas auth=publickey' },
  { source: 'Wazuh', cls: 'benign', conf: 0.997, mitre: null,
    title: 'Scheduled service restart on db-prod-01',
    host: 'db-prod-01', user: 'systemd', sev: 4,
    raw: 'unit=postgresql.service action=restart trigger=cron' },
  { source: 'Netskope', cls: 'benign', conf: 0.995, mitre: null,
    title: 'Sanctioned Slack file upload by jdoe',
    host: 'WS-1102', user: 'jdoe', sev: 4,
    raw: 'destination=slack.com app_status=sanctioned file=design.pdf' },
  { source: 'Trellix EDR', cls: 'benign', conf: 0.989, mitre: null,
    title: 'Patch installation via SCCM on WS-2034',
    host: 'WS-2034', user: 'SYSTEM', sev: 4,
    raw: 'agent=ccmexec.exe package=KB5028166 result=success' },
  { source: 'CloudTrail', cls: 'benign', conf: 0.998, mitre: null,
    title: 'API call from monitoring service (datadog)',
    host: 'aws-prod', user: 'datadog-agent', sev: 4,
    raw: 'eventName=DescribeInstances userIdentity.userName=datadog-agent' },
  { source: 'CrowdStrike', cls: 'benign', conf: 0.994, mitre: null,
    title: 'Browser cache cleared via group policy',
    host: 'WS-1456', user: 'gpolicy', sev: 4,
    raw: 'process=gpupdate.exe action=cache_clear scope=user' },
  { source: 'Wazuh', cls: 'benign', conf: 0.996, mitre: null,
    title: 'Backup script ran successfully on file-server-01',
    host: 'fs-01', user: 'svc_backup', sev: 4,
    raw: 'script=/opt/backup/run.sh exit_code=0 duration=42m' },
  { source: 'Purview DLP', cls: 'benign', conf: 0.992, mitre: null,
    title: 'User logged in to M365 from registered device',
    host: 'WS-2287', user: 'lwilliams', sev: 4,
    raw: 'app=M365 auth=SSO device_compliant=true risk=low' },
  { source: 'Trellix EDR', cls: 'benign', conf: 0.999, mitre: null,
    title: 'AV signature update on EXEC-12',
    host: 'EXEC-12', user: 'SYSTEM', sev: 4,
    raw: 'service=trellix_av update_type=signature version=2026.05.14' },
];


/* ============ 2. SOURCE REGISTRY + MOCK COUNTS ============ */
const SOURCES = ['CrowdStrike', 'Wazuh', 'Trellix EDR', 'Purview DLP', 'CloudTrail', 'Netskope'];

const SOURCE_COUNTS = {
  'CrowdStrike': 3142,
  'Wazuh': 2891,
  'Trellix EDR': 1456,
  'CloudTrail': 387,
  'Purview DLP': 245,
  'Netskope': 126
};


/* ============ 5. FIELD-NAME DICTIONARIES ============ */
/* Defined before extractAlerts / normalizeAlert so they can use them. */
const FIELD_ALIASES = {
  classification: ['classification', 'predicted_label', 'pred_label', 'predicted',
                   'prediction', 'pred', 'label', 'y_pred', 'class'],
  confidence:     ['confidence', 'score', 'probability', 'prob', 'conf', 'pred_prob',
                   'predicted_prob', 'p_threat', 'threat_prob'],
  title:          ['title', 'alert', 'alert_text', 'text', 'message', 'msg',
                   'description', 'event', 'event_text', 'full_log', 'rule_description'],
  source:         ['source', 'tool', 'product', 'detector', 'source_tool', 'origin',
                   'data_source', 'sensor'],
  host:           ['host', 'hostname', 'computer', 'machine', 'asset', 'agent_name',
                   'system', 'srcip', 'src_ip'],
  user:           ['user', 'username', 'account', 'user_name', 'srcuser', 'src_user',
                   'subject'],
  severity:       ['severity', 'sev', 'level', 'rule_level', 'risk_score', 'priority'],
  mitre:          ['mitre', 'mitre_id', 'attack_id', 'technique', 'technique_id',
                   'mitre_attack', 'mitre_technique'],
  raw:            ['raw', 'raw_alert', 'raw_log', 'payload', 'log', 'full_log'],
  truth:          ['true_label', 'truth', 'gt', 'ground_truth', 'actual', 'y_true', 'y'],
  timestamp:      ['timestamp', 'ts', 'time', 'event_time', '@timestamp', 'created_at',
                   'date'],
};


/* ============ 3. extractAlerts() ============ */
/* Given a parsed JSON object, return the alerts array no matter where it sits. */
function extractAlerts(json) {
  if (Array.isArray(json)) return json;
  if (json == null || typeof json !== 'object') return [];

  // Try common wrapper keys first
  const wrappers = ['alerts', 'results', 'data', 'predictions', 'classifications',
                    'records', 'items', 'rows', 'samples', 'entries'];
  for (const key of wrappers) {
    if (Array.isArray(json[key])) return json[key];
  }

  // Last resort: first array property found
  for (const key in json) {
    if (Array.isArray(json[key]) && json[key].length > 0
        && typeof json[key][0] === 'object') {
      return json[key];
    }
  }
  return [];
}


/* ============ 4. normalizeAlert() ============ */
/* Map an arbitrary alert object to our internal schema. */
function pick(obj, aliases, fallback) {
  for (const key of aliases) {
    if (obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return fallback;
}

function normalizeClassification(value) {
  if (value == null) return 'benign';
  if (typeof value === 'number') return value >= 0.5 ? 'threat' : 'benign';
  if (typeof value === 'boolean') return value ? 'threat' : 'benign';
  const s = String(value).toLowerCase().trim();
  if (['1', 'true', 'yes', 'pos', 'positive'].includes(s)) return 'threat';
  if (['0', 'false', 'no', 'neg', 'negative'].includes(s)) return 'benign';
  if (s.includes('threat') || s.includes('malicious') || s.includes('attack')
      || s.includes('alert') || s.includes('positive') || s.includes('anomal'))
    return 'threat';
  return 'benign';
}

function normalizeConfidence(value) {
  if (value == null) return 0;
  let n = parseFloat(value);
  if (isNaN(n)) return 0;
  if (n > 1) n = n / 100;          // looks like a percent
  return Math.max(0, Math.min(1, n));
}

function normalizeTimestamp(value, index, total) {
  if (value != null) {
    const t = new Date(value).getTime();
    if (!isNaN(t)) return t;
  }
  // Spread synthetic timestamps across the last 24 hours
  const now = Date.now();
  const span = 24 * 60 * 60 * 1000;
  return now - span * (1 - (index / Math.max(1, total - 1)));
}

function normalizeAlert(raw, index, total) {
  if (typeof raw !== 'object' || raw === null) raw = { text: String(raw) };

  const cls = normalizeClassification(pick(raw, FIELD_ALIASES.classification));
  const conf = normalizeConfidence(pick(raw, FIELD_ALIASES.confidence, 0.9));
  const title = String(pick(raw, FIELD_ALIASES.title, '(no title)')).slice(0, 200);
  const source = String(pick(raw, FIELD_ALIASES.source, 'Unknown'));
  const host = String(pick(raw, FIELD_ALIASES.host, '-'));
  const user = String(pick(raw, FIELD_ALIASES.user, '-'));
  const sev = parseInt(pick(raw, FIELD_ALIASES.severity, 5)) || 5;
  const mitre = pick(raw, FIELD_ALIASES.mitre, null);
  const truth = pick(raw, FIELD_ALIASES.truth, null);
  const tsRaw = pick(raw, FIELD_ALIASES.timestamp);
  const timestamp = normalizeTimestamp(tsRaw, index, total);

  // Raw signal: explicit field, else compact JSON dump of the original record
  let rawSignal = pick(raw, FIELD_ALIASES.raw);
  if (!rawSignal) {
    try {
      rawSignal = JSON.stringify(raw, null, 2);
      if (rawSignal.length > 500) rawSignal = rawSignal.slice(0, 500) + '\n...';
    } catch (e) {
      rawSignal = String(raw);
    }
  }

  return {
    id: 'd-' + String(index + 1).padStart(6, '0'),
    timestamp: timestamp,
    source: source,
    classification: cls,
    confidence: conf,
    mitre: mitre ? String(mitre) : null,
    title: title,
    host: host,
    user: user,
    severity: sev,
    raw: String(rawSignal),
    trueLabel: truth ? normalizeClassification(truth) : null,
    fresh: false,
    expanded: false,
  };
}