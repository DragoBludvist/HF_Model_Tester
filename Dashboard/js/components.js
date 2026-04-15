/* components.js — UI components for SOC dashboard */

const CAT_COLORS = {
  windows: "#818cf8",
  linux: "#34d399",
  network: "#fbbf24",
  cloud: "#60a5fa",
};

const CATEGORIES = ["all", "windows", "linux", "network", "cloud"];

/* ── Stat Cards ───────────────────────────────────────────── */

function StatCards({ data }) {
  const meta = data.metadata;
  const total = meta.total_alerts;
  const cards = [
    { cls: "threat", label: "Threats detected", value: meta.threats_found.toLocaleString(), sub: `of ${total.toLocaleString()} total` },
    { cls: "benign", label: "Benign filtered", value: meta.benign_filtered.toLocaleString(), sub: `${(meta.benign_filtered / total * 100).toFixed(0)}% noise removed` },
    { cls: "accuracy", label: meta.accuracy != null ? "Model accuracy" : "Threat rate", value: meta.accuracy != null ? `${meta.accuracy}%` : `${meta.threat_rate}%`, sub: meta.accuracy != null ? "on test set" : "of total alerts" },
    { cls: "speed", label: "Avg inference", value: `${meta.avg_time_per_alert_ms}ms`, sub: "per alert" },
  ];

  return React.createElement("div", { className: "stats-grid" },
    cards.map((s, i) =>
      React.createElement("div", { key: i, className: `stat-card ${s.cls}` },
        React.createElement("div", { className: "label" }, s.label),
        React.createElement("div", { className: "value" }, s.value),
        React.createElement("div", { className: "sub" }, s.sub)
      )
    )
  );
}

/* ── Filters ──────────────────────────────────────────────── */

function Filters({ filter, setFilter, catFilter, setCatFilter, search, setSearch }) {
  const predFilters = [
    { label: "All", value: "all" },
    { label: "Threats", value: "threat" },
    { label: "Benign", value: "benign" },
  ];

  return React.createElement("div", { className: "filters" },
    predFilters.map((f) =>
      React.createElement("button", {
        key: f.value,
        className: `filter-btn ${filter === f.value ? "active" : ""}`,
        onClick: () => setFilter(f.value),
      }, f.label)
    ),
    React.createElement("div", { className: "divider" }),
    CATEGORIES.map((cat) =>
      React.createElement("button", {
        key: cat,
        className: `filter-btn ${catFilter === cat ? "active" : ""}`,
        onClick: () => setCatFilter(cat),
        style: catFilter === cat && cat !== "all" ? {
          borderColor: CAT_COLORS[cat],
          color: CAT_COLORS[cat],
          background: `${CAT_COLORS[cat]}18`,
        } : {},
      }, cat === "all" ? "All categories" : cat.charAt(0).toUpperCase() + cat.slice(1))
    ),
    React.createElement("input", {
      className: "search-input",
      placeholder: "Search alerts...",
      value: search,
      onChange: (e) => setSearch(e.target.value),
    })
  );
}

/* ── Alert Table ──────────────────────────────────────────── */

function AlertTable({ alerts, selected, setSelected, sortBy, sortDir, toggleSort }) {
  const headers = [
    { label: "ID", col: "id" },
    { label: "Category", col: "source_category" },
    { label: "Alert", col: "alert_text" },
    { label: "Confidence", col: "confidence" },
    { label: "Verdict", col: "prediction" },
  ];

  return React.createElement("div", { className: "alert-table" },
    React.createElement("div", { className: "table-header" },
      headers.map((h) =>
        React.createElement("span", { key: h.col, onClick: () => toggleSort(h.col) },
          h.label, " ", sortBy === h.col ? (sortDir === "desc" ? "\u2193" : "\u2191") : ""
        )
      )
    ),
    React.createElement("div", { className: "table-body" },
      alerts.map((a) =>
        React.createElement("div", {
          key: a.id,
          className: `alert-row ${selected?.id === a.id ? "selected" : ""} ${a.prediction === "threat" ? "is-threat" : ""}`,
          onClick: () => setSelected(a),
        },
          React.createElement("span", { className: "id" }, `#${a.id}`),
          React.createElement("span", {
            className: "cat-badge",
            style: { background: `${CAT_COLORS[a.source_category]}18`, color: CAT_COLORS[a.source_category] },
          }, a.source_category),
          React.createElement("span", { className: "alert-text" },
            a.alert_text.replace(/\[(windows|linux|network|cloud)\]\s*/, "")
          ),
          React.createElement("div", { className: "conf-cell" },
            React.createElement("div", {
              className: "conf-value",
              style: { color: a.prediction === "threat" ? "#fca5a5" : "#86efac" },
            }, `${(a.confidence * 100).toFixed(1)}%`),
            React.createElement("div", { className: "conf-bar" },
              React.createElement("div", {
                className: "conf-bar-fill",
                style: { width: `${a.confidence * 100}%`, background: a.prediction === "threat" ? "#ef4444" : "#10b981" },
              })
            )
          ),
          React.createElement("span", {
            className: "verdict-badge",
            style: {
              background: a.prediction === "threat" ? "#ef44441a" : "#10b9811a",
              color: a.prediction === "threat" ? "#fca5a5" : "#86efac",
            },
          }, a.prediction === "threat" ? "THREAT" : "BENIGN")
        )
      )
    )
  );
}

/* ── Detail Panel ─────────────────────────────────────────── */

function DetailPanel({ alert, onClose, llmAnalysis, llmLoading, onRequestLlm }) {
  const a = alert;
  const hasLabel = a.actual_label && a.actual_label !== "unknown";
  const labelMatch = hasLabel && a.actual_label === a.prediction;

  return React.createElement("div", { className: "detail-panel" },
    /* Top bar */
    React.createElement("div", { className: "detail-top" },
      React.createElement("div", { style: { display: "flex", gap: 8 } },
        React.createElement("span", {
          className: "verdict-badge",
          style: { background: a.prediction === "threat" ? "#ef44441a" : "#10b9811a", color: a.prediction === "threat" ? "#fca5a5" : "#86efac", fontSize: 12, fontWeight: 700 },
        }, a.prediction.toUpperCase()),
        React.createElement("span", {
          className: "cat-badge",
          style: { background: `${CAT_COLORS[a.source_category]}18`, color: CAT_COLORS[a.source_category] },
        }, a.source_category)
      ),
      React.createElement("button", { className: "close-btn", onClick: onClose }, "\u00d7")
    ),

    /* Alert text */
    React.createElement("div", { className: "detail-label" }, `Alert #${a.id}`),
    React.createElement("div", { className: "detail-alert-text" }, a.alert_text),

    /* Metrics */
    React.createElement("div", { className: "detail-metrics" },
      React.createElement("div", { className: "detail-metric" },
        React.createElement("div", { className: "dm-label" }, "Confidence"),
        React.createElement("div", {
          className: "dm-value",
          style: { color: a.prediction === "threat" ? "#fca5a5" : "#86efac" },
        }, `${(a.confidence * 100).toFixed(2)}%`)
      ),
      React.createElement("div", { className: "detail-metric" },
        React.createElement("div", { className: "dm-label" }, hasLabel ? "Actual label" : "Threat probability"),
        hasLabel
          ? React.createElement("div", {
              className: "dm-value",
              style: { fontSize: 14, fontWeight: 600, color: labelMatch ? "#86efac" : "#fca5a5" },
            }, `${a.actual_label} ${labelMatch ? "\u2713" : "\u2717 mismatch"}`)
          : React.createElement("div", {
              className: "dm-value",
              style: { color: a.threat_probability > 0.5 ? "#fca5a5" : "#86efac" },
            }, `${(a.threat_probability * 100).toFixed(2)}%`)
      )
    ),

    /* Key fields */
    React.createElement("div", { className: "detail-label" }, "Key fields"),
    React.createElement("div", { style: { marginBottom: 16 } },
      Object.entries(a.rule_summary.key_fields).map(([k, v]) =>
        React.createElement("div", { key: k, className: "key-field" },
          React.createElement("span", { className: "kf-label" }, `${k}:`),
          React.createElement("span", { className: "kf-value" }, v)
        )
      )
    ),

    /* Rule-based assessment */
    React.createElement("div", { className: "detail-label" }, "Rule-based assessment"),
    React.createElement("div", { className: `assessment-box ${a.prediction}` }, a.rule_summary.analyst_note),

    /* AI analysis */
    React.createElement("div", { className: "detail-label" }, "AI detailed analysis"),
    llmAnalysis[a.id]
      ? React.createElement("div", { className: "assessment-box ai" }, llmAnalysis[a.id])
      : React.createElement("button", {
          className: "ai-btn",
          onClick: () => onRequestLlm(a),
          disabled: llmLoading[a.id],
        }, llmLoading[a.id] ? "Analyzing..." : "Request AI analysis")
  );
}
