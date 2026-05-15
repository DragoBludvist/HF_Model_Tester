/* ============================================================
   NAVITAS SOC // DASHBOARD
   ------------------------------------------------------------
   Section index:
     1. State
     2. Helpers (formatting, randomness, toast)
     3. Alert factory + feed seeding (live mode)
     4. Feed rendering (filter, search, expand)
     5. Filter / search / play / refresh handlers
     6. Upload handling (button, drag-drop, parsing)
     7. Mode switching (live <-> dataset)
     8. KPI + chart recomputation from loaded data
     9. Clock
    10. Live-mode counters
    11. Volume chart (SVG)
    12. Source bars
    13. Init
   ------------------------------------------------------------
   Depends on: js/data.js
     - ALERT_TEMPLATES, SOURCES, SOURCE_COUNTS
     - extractAlerts(), normalizeAlert()
   ============================================================ */


/* ============ 1. STATE ============ */
let alerts = [];
let nextId = 1;
let playing = true;
let activeFilter = 'all';
let searchTerm = '';
let mode = 'live';              // 'live' | 'dataset'
let datasetName = null;
let liveAlertTimer = null;


/* ============ 2. HELPERS ============ */
function fmtTime(d) {
  const diff = Math.floor((Date.now() - d) / 1000);
  if (diff < 0) return 'in future';
  if (diff < 5) return 'just now';
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  return Math.floor(diff / 86400) + 'd ago';
}

function fmtClock(d) {
  return d.toTimeString().slice(0, 8);
}

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function showToast(message, isError) {
  const t = document.getElementById('toast');
  t.textContent = message;
  t.classList.toggle('error', !!isError);
  t.classList.add('visible');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => t.classList.remove('visible'), 3600);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}


/* ============ 3. ALERT FACTORY + FEED SEEDING (LIVE MODE) ============ */
function makeAlert() {
  const tpl = pickRandom(ALERT_TEMPLATES);
  return {
    id: 'a-' + String(nextId++).padStart(6, '0'),
    timestamp: Date.now(),
    source: tpl.source,
    classification: tpl.cls,
    confidence: tpl.conf,
    mitre: tpl.mitre,
    title: tpl.title,
    host: tpl.host,
    user: tpl.user,
    severity: tpl.sev,
    raw: tpl.raw,
    fresh: true,
    expanded: false,
  };
}

function seedLiveFeed() {
  alerts = [];
  nextId = 1;
  for (let i = 0; i < 12; i++) {
    const a = makeAlert();
    a.timestamp = Date.now() - (12 - i) * (1500 + Math.random() * 4000);
    a.fresh = false;
    alerts.unshift(a);
  }
  alerts.reverse();
}


/* ============ 4. FEED RENDERING ============ */
function alertMatchesFilter(a) {
  if (activeFilter === 'threat' && a.classification !== 'threat') return false;
  if (activeFilter === 'benign' && a.classification !== 'benign') return false;
  if (activeFilter === 'high' && a.confidence < 0.99) return false;
  if (searchTerm) {
    const term = searchTerm.toLowerCase();
    if (!a.title.toLowerCase().includes(term) &&
        !a.host.toLowerCase().includes(term) &&
        !a.user.toLowerCase().includes(term) &&
        !(a.mitre && a.mitre.toLowerCase().includes(term)) &&
        !a.source.toLowerCase().includes(term)) {
      return false;
    }
  }
  return true;
}

function renderFeed() {
  const list = document.getElementById('feed-list');
  const visible = alerts.filter(alertMatchesFilter);

  if (!visible.length) {
    list.innerHTML = '<div class="empty-feed">No alerts match the current filter</div>';
    return;
  }

  list.innerHTML = visible.map(renderAlert).join('');
  visible.forEach(a => {
    const el = document.getElementById(a.id);
    if (el) el.addEventListener('click', () => toggleAlert(a.id));
  });
}

function renderAlert(a) {
  const confPct = (a.confidence * 100).toFixed(1);
  const mitre = a.mitre ? `<span class="mitre">${escapeHtml(a.mitre)}</span>` : '';
  const truthBadge = a.trueLabel && a.trueLabel !== a.classification
    ? `<span class="mitre" style="color:var(--amber)">MISCLASSIFIED (truth: ${escapeHtml(a.trueLabel)})</span>`
    : '';
  return `
    <div class="alert ${a.classification} ${a.fresh ? 'fresh' : ''} ${a.expanded ? 'expanded' : ''}" id="${a.id}">
      <span class="badge ${a.classification}">${a.classification === 'threat' ? 'Threat' : 'Benign'}</span>
      <span class="badge source">${escapeHtml(a.source)}</span>
      <div class="alert-content">
        <div class="alert-title">${escapeHtml(a.title)}</div>
        <div class="alert-meta">
          ${mitre}
          <span class="confidence">Confidence <strong>${confPct}%</strong></span>
          <span>Sev ${a.severity}</span>
          ${truthBadge}
        </div>
      </div>
      <div class="alert-time">${fmtTime(a.timestamp)}</div>
      <div class="alert-detail">
        <div class="detail-field"><div class="detail-label">Alert ID</div><div class="detail-value">${escapeHtml(a.id)}</div></div>
        <div class="detail-field"><div class="detail-label">Host</div><div class="detail-value">${escapeHtml(a.host)}</div></div>
        <div class="detail-field"><div class="detail-label">User</div><div class="detail-value">${escapeHtml(a.user)}</div></div>
        <div class="detail-field"><div class="detail-label">Severity</div><div class="detail-value">${a.severity}</div></div>
        ${a.trueLabel ? `<div class="detail-field"><div class="detail-label">True label</div><div class="detail-value">${escapeHtml(a.trueLabel)}</div></div>` : ''}
        <div class="detail-field full"><div class="detail-label">Raw signal</div><div class="detail-value raw">${escapeHtml(a.raw)}</div></div>
      </div>
    </div>
  `;
}

function toggleAlert(id) {
  const a = alerts.find(x => x.id === id);
  if (a) {
    a.expanded = !a.expanded;
    renderFeed();
  }
}

function addNewAlert() {
  if (!playing || mode !== 'live') return;
  const a = makeAlert();
  alerts.unshift(a);
  if (alerts.length > 50) alerts.pop();
  setTimeout(() => { a.fresh = false; }, 600);
  renderFeed();
  bumpLiveKpi();
}


/* ============ 5. FILTER / SEARCH / PLAY / REFRESH HANDLERS ============ */
document.querySelectorAll('.filter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.filter-chip').forEach(c => {
      c.classList.remove('active', 'threats');
    });
    chip.classList.add('active');
    if (chip.dataset.filter === 'threat') chip.classList.add('threats');
    activeFilter = chip.dataset.filter;
    renderFeed();
  });
});

document.getElementById('search').addEventListener('input', (e) => {
  searchTerm = e.target.value;
  renderFeed();
});

document.getElementById('play-toggle').addEventListener('click', (e) => {
  playing = !playing;
  e.target.textContent = playing ? '\u258C\u258C' : '\u25B6';
  e.target.title = playing ? 'Pause feed' : 'Resume feed';
  e.target.classList.toggle('active', playing);
});

document.getElementById('refresh').addEventListener('click', () => {
  if (mode === 'live') {
    seedLiveFeed();
    renderFeed();
  } else {
    showToast('In dataset mode — click ' + '\u00D7' + ' to return to live');
  }
});


/* ============ 6. UPLOAD HANDLING ============ */
function handleFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.json')) {
    showToast('Only .json files are supported', true);
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const json = JSON.parse(e.target.result);
      const raw = extractAlerts(json);
      if (!raw.length) {
        showToast('No alert records found in this file', true);
        return;
      }
      const normalized = raw.map((r, i) => normalizeAlert(r, i, raw.length));
      loadDataset(file.name, normalized);
    } catch (err) {
      showToast('Failed to parse JSON: ' + err.message, true);
    }
  };
  reader.onerror = () => showToast('Failed to read file', true);
  reader.readAsText(file);
}

document.getElementById('upload-btn').addEventListener('click', () => {
  document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
  e.target.value = '';   // allow re-uploading the same file
});

document.getElementById('clear-data').addEventListener('click', () => {
  switchToLive();
  showToast('Returned to live mode');
});

// Drag-and-drop anywhere on the page
let dragCounter = 0;
document.addEventListener('dragenter', (e) => {
  if (e.dataTransfer && e.dataTransfer.types.includes('Files')) {
    dragCounter++;
    document.getElementById('drop-overlay').classList.add('active');
  }
});
document.addEventListener('dragover', (e) => {
  if (e.dataTransfer && e.dataTransfer.types.includes('Files')) e.preventDefault();
});
document.addEventListener('dragleave', () => {
  dragCounter = Math.max(0, dragCounter - 1);
  if (dragCounter === 0) document.getElementById('drop-overlay').classList.remove('active');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  dragCounter = 0;
  document.getElementById('drop-overlay').classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});


/* ============ 7. MODE SWITCHING ============ */
function loadDataset(name, normalizedAlerts) {
  mode = 'dataset';
  datasetName = name;
  alerts = normalizedAlerts.sort((a, b) => b.timestamp - a.timestamp);
  playing = false;

  // Pause the play button visually
  const playBtn = document.getElementById('play-toggle');
  playBtn.textContent = '\u25B6';
  playBtn.classList.remove('active');
  playBtn.title = 'Replay mode (not active in static dataset view)';

  // Status pill -> dataset mode
  const status = document.getElementById('status-indicator');
  status.classList.add('dataset');
  document.getElementById('status-text').textContent = 'Dataset';

  // Show dataset bar
  const bar = document.getElementById('dataset-bar');
  bar.classList.add('visible');
  document.getElementById('dataset-name').textContent = name;

  // Show clear button
  document.getElementById('clear-data').classList.add('visible');

  // Update panel and chart titles
  document.getElementById('feed-title').textContent = 'Classified alerts';
  document.getElementById('kpi-volume-meta').textContent = 'in this dataset';
  document.getElementById('chart-volume-title').textContent = 'Alert distribution';
  document.getElementById('chart-donut-sub').textContent = 'this dataset';
  document.getElementById('chart-sources-sub').textContent = 'this dataset';

  recomputeAll();
  renderFeed();

  const threats = alerts.filter(a => a.classification === 'threat').length;
  showToast(`Loaded ${alerts.length.toLocaleString()} alerts (${threats.toLocaleString()} threats)`);
}

function switchToLive() {
  mode = 'live';
  datasetName = null;
  seedLiveFeed();
  playing = true;

  // Play button back to active
  const playBtn = document.getElementById('play-toggle');
  playBtn.textContent = '\u258C\u258C';
  playBtn.classList.add('active');
  playBtn.title = 'Pause feed';

  // Status pill -> live
  const status = document.getElementById('status-indicator');
  status.classList.remove('dataset');
  document.getElementById('status-text').textContent = 'Live';

  // Hide dataset bar + clear button
  document.getElementById('dataset-bar').classList.remove('visible');
  document.getElementById('clear-data').classList.remove('visible');

  // Restore titles
  document.getElementById('feed-title').textContent = 'Live Alert Feed';
  document.getElementById('kpi-volume-meta').textContent = 'vs 24h avg';
  document.getElementById('chart-volume-title').textContent = 'Alert volume / 24h';
  document.getElementById('chart-donut-sub').textContent = 'last 24h';
  document.getElementById('chart-sources-sub').textContent = 'today';

  // Reset to mock KPIs
  document.getElementById('kpi-volume').textContent = '24,431';
  document.getElementById('kpi-threats').textContent = '142';
  document.getElementById('kpi-confidence').textContent = '99.2%';
  document.getElementById('kpi-confidence-meta').textContent = 'across 8,247 classifications';
  document.getElementById('kpi-saved').textContent = '87%';
  document.getElementById('legend-threat-count').textContent = '1,154';
  document.getElementById('legend-benign-count').textContent = '7,093';
  document.getElementById('donut-pct').textContent = '14%';
  document.getElementById('donut-ring').setAttribute('stroke-dashoffset', '237');

  drawVolumeChart(null);   // synthetic series
  drawSources(SOURCE_COUNTS);
  renderFeed();
}


/* ============ 8. KPI + CHART RECOMPUTATION FROM LOADED DATA ============ */
function recomputeAll() {
  if (mode !== 'dataset') return;

  const total = alerts.length;
  const threats = alerts.filter(a => a.classification === 'threat').length;
  const benigns = total - threats;
  const avgConf = total ? alerts.reduce((s, a) => s + a.confidence, 0) / total : 0;
  const threatPct = total ? Math.round((threats / total) * 100) : 0;

  document.getElementById('kpi-volume').textContent = total.toLocaleString();
  document.getElementById('kpi-threats').textContent = threats.toLocaleString();
  document.getElementById('kpi-confidence').textContent = (avgConf * 100).toFixed(1) + '%';
  document.getElementById('kpi-confidence-meta').textContent = `across ${total.toLocaleString()} classifications`;
  document.getElementById('kpi-saved').textContent = Math.round((benigns / Math.max(1, total)) * 100) + '%';

  // Donut
  document.getElementById('legend-threat-count').textContent = threats.toLocaleString();
  document.getElementById('legend-benign-count').textContent = benigns.toLocaleString();
  document.getElementById('donut-pct').textContent = threatPct + '%';
  const circ = 276.46;
  document.getElementById('donut-ring').setAttribute('stroke-dashoffset', (circ * (1 - threats / Math.max(1, total))).toFixed(2));

  // Source bars
  const counts = {};
  alerts.forEach(a => {
    counts[a.source] = (counts[a.source] || 0) + 1;
  });
  drawSources(counts);

  // Volume chart from data timestamps
  drawVolumeChart(alerts);

  // Dataset stats line
  const accNote = alerts.some(a => a.trueLabel)
    ? ' \u00B7 has truth labels'
    : '';
  document.getElementById('dataset-stats').textContent =
    `${total.toLocaleString()} records \u00B7 ${threats.toLocaleString()} threats \u00B7 ${(avgConf * 100).toFixed(1)}% avg conf${accNote}`;
}


/* ============ 9. CLOCK ============ */
function tickClock() {
  document.getElementById('clock').textContent = fmtClock(new Date());
}
setInterval(tickClock, 1000);
tickClock();

// Re-render "Xs ago" timestamps periodically
setInterval(() => { if (mode === 'live') renderFeed(); }, 10000);


/* ============ 10. LIVE-MODE COUNTERS ============ */
let liveVolume = 24431;
let liveThreats = 142;

function bumpLiveKpi() {
  liveVolume += Math.floor(Math.random() * 6) + 1;
  document.getElementById('kpi-volume').textContent = liveVolume.toLocaleString();
  if (Math.random() < 0.18) {
    liveThreats += 1;
    document.getElementById('kpi-threats').textContent = liveThreats;
  }
}


/* ============ 11. VOLUME CHART (SVG) ============ */
/* If alertData is passed, bin by hour. Otherwise synthesize a diurnal curve. */
function drawVolumeChart(alertData) {
  const N = 48;            // 30-min buckets across 24h
  const W = 320;
  const H = 70;
  const pad = 4;
  const vol = new Array(N).fill(0);
  const threats = new Array(N).fill(0);

  if (alertData && alertData.length) {
    // Bin by 30-min interval relative to most recent
    const maxTs = Math.max(...alertData.map(a => a.timestamp));
    const minTs = Math.min(...alertData.map(a => a.timestamp));
    const span = Math.max(1, maxTs - minTs);
    for (const a of alertData) {
      const idx = Math.min(N - 1, Math.floor(((a.timestamp - minTs) / span) * N));
      vol[idx] += 1;
      if (a.classification === 'threat') threats[idx] += 1;
    }
  } else {
    // Synthetic
    for (let i = 0; i < N; i++) {
      const hour = i / 2;
      const diurnal = 0.55 + 0.35 * Math.sin(((hour - 8) / 24) * Math.PI * 2);
      const v = Math.max(0.15, diurnal + (Math.random() - 0.5) * 0.15);
      vol[i] = v;
      threats[i] = v * (0.08 + Math.random() * 0.12);
    }
  }

  const max = Math.max(0.0001, ...vol);
  const xs = i => pad + (i / (N - 1)) * (W - pad * 2);
  const ys = v => H - (v / max) * (H - 8);

  const linePts = vol.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');
  const areaD = `M${xs(0)},${H} L` + linePts.split(' ').join(' L') + ` L${xs(N - 1)},${H} Z`;
  const threatPts = threats.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');

  document.getElementById('volume-area').setAttribute('d', areaD);
  document.getElementById('volume-line').setAttribute('points', linePts);
  document.getElementById('threat-line').setAttribute('points', threatPts);

  // Axis labels for dataset mode
  if (alertData && alertData.length) {
    const minTs = Math.min(...alertData.map(a => a.timestamp));
    const maxTs = Math.max(...alertData.map(a => a.timestamp));
    document.getElementById('volume-x0').textContent = new Date(minTs).toISOString().slice(11, 16);
    document.getElementById('volume-x2').textContent = new Date(maxTs).toISOString().slice(11, 16);
    document.getElementById('volume-x1').textContent = '\u2014';
  } else {
    document.getElementById('volume-x0').textContent = '00:00';
    document.getElementById('volume-x1').textContent = '12:00';
    document.getElementById('volume-x2').textContent = 'now';
  }
}


/* ============ 12. SOURCE BARS ============ */
function drawSources(counts) {
  const container = document.getElementById('sources');
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  if (!entries.length) {
    container.innerHTML = '<div class="empty-feed" style="padding: 20px 0">No source data</div>';
    return;
  }
  const max = Math.max(...entries.map(e => e[1]));
  container.innerHTML = entries.map(([name, v]) => {
    const pct = (v / max) * 100;
    return `
      <div class="source-row">
        <div class="source-name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
        <div class="source-bar"><div class="source-fill" style="width:${pct.toFixed(0)}%"></div></div>
        <div class="source-value">${v.toLocaleString()}</div>
      </div>
    `;
  }).join('');
}


/* ============ 13. INIT ============ */
seedLiveFeed();
renderFeed();
drawVolumeChart(null);
drawSources(SOURCE_COUNTS);
liveAlertTimer = setInterval(addNewAlert, 4000);