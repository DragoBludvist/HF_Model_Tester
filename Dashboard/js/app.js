/* app.js — Main application, state management, data loading */

const { useState, useMemo, useCallback, useRef } = React;

function App() {
  const [data, setData] = useState(SAMPLE_DATA);
  const [filter, setFilter] = useState("all");
  const [catFilter, setCatFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [llmAnalysis, setLlmAnalysis] = useState({});
  const [llmLoading, setLlmLoading] = useState({});
  const [sortBy, setSortBy] = useState("threat_probability");
  const [sortDir, setSortDir] = useState("desc");
  const [customLoaded, setCustomLoaded] = useState(false);
  const fileRef = useRef();

  /* ── File upload ──────────────────────────────────────── */
  const handleUpload = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const json = JSON.parse(ev.target.result);
        if (!json.metadata || !json.alerts) {
          alert("Invalid format. Expected {metadata, alerts}.");
          return;
        }
        setData(json);
        setCustomLoaded(true);
        setSelected(null);
        setLlmAnalysis({});
      } catch {
        alert("Invalid JSON file.");
      }
    };
    reader.readAsText(file);
  }, []);

  /* ── Filtering & sorting ─────────────────────────────── */
  const filtered = useMemo(() => {
    let items = data.alerts;
    if (filter !== "all") items = items.filter((a) => a.prediction === filter);
    if (catFilter !== "all") items = items.filter((a) => a.source_category === catFilter);
    if (search) {
      const q = search.toLowerCase();
      items = items.filter((a) => a.alert_text.toLowerCase().includes(q));
    }
    return [...items].sort((a, b) => {
      const av = a[sortBy], bv = b[sortBy];
      if (typeof av === "number") return sortDir === "desc" ? bv - av : av - bv;
      return sortDir === "desc" ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    });
  }, [data, filter, catFilter, search, sortBy, sortDir]);

  const toggleSort = useCallback((col) => {
    if (sortBy === col) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else { setSortBy(col); setSortDir("desc"); }
  }, [sortBy]);

  /* ── LLM request ─────────────────────────────────────── */
  const handleRequestLlm = useCallback(async (al) => {
    if (llmAnalysis[al.id] || llmLoading[al.id]) return;
    setLlmLoading((p) => ({ ...p, [al.id]: true }));
    try {
      const text = await requestAnalysis(al);
      setLlmAnalysis((p) => ({ ...p, [al.id]: text }));
    } catch {
      setLlmAnalysis((p) => ({ ...p, [al.id]: "Error connecting to analysis service." }));
    }
    setLlmLoading((p) => ({ ...p, [al.id]: false }));
  }, [llmAnalysis, llmLoading]);

  /* ── Render ──────────────────────────────────────────── */
  return React.createElement("div", { className: "container" },

    /* Header */
    React.createElement("div", { className: "header" },
      React.createElement("div", null,
        React.createElement("h1", null, "Alert classification"),
        React.createElement("span", { className: "model-tag" }, data.metadata.model)
      ),
      React.createElement("p", { className: "subtitle" },
        customLoaded ? "Custom data loaded" : "Sample data",
        ` \u2014 ${data.alerts.length.toLocaleString()} alerts classified`
      )
    ),

    /* Upload bar */
    React.createElement("div", { className: "upload-bar" },
      React.createElement("p", null, "Load classified_alerts.json from the inference script, or browse with sample data"),
      React.createElement("input", { type: "file", accept: ".json", ref: fileRef, onChange: handleUpload, style: { display: "none" } }),
      React.createElement("button", { className: "upload-btn", onClick: () => fileRef.current.click() }, "Upload JSON"),
      customLoaded && React.createElement("span", { className: "upload-status" }, "\u2713 Loaded")
    ),

    /* Stats */
    React.createElement(StatCards, { data }),

    /* Filters */
    React.createElement(Filters, { filter, setFilter, catFilter, setCatFilter, search, setSearch }),
    React.createElement("div", { className: "result-count" }, `${filtered.length} alerts shown`),

    /* Main layout */
    React.createElement("div", { className: "main-layout" },
      React.createElement("div", { className: `table-section ${selected ? "with-detail" : ""}` },
        React.createElement(AlertTable, { alerts: filtered, selected, setSelected, sortBy, sortDir, toggleSort })
      ),
      selected && React.createElement("div", { className: "detail-section" },
        React.createElement(DetailPanel, {
          alert: selected,
          onClose: () => setSelected(null),
          llmAnalysis,
          llmLoading,
          onRequestLlm: handleRequestLlm,
        })
      )
    )
  );
}

ReactDOM.render(React.createElement(App), document.getElementById("app"));
