/* ============================================================
   NAVITAS SOC // DASHBOARD
   ------------------------------------------------------------
   Section index:
     1. State
     2. Helpers (formatting, randomness)
     3. Alert factory + feed seeding
     4. Feed rendering (filtering, search, expand)
     5. Filter / search / play / refresh handlers
     6. Clock
     7. KPI counters
     8. Volume chart (SVG)
     9. Source bars
    10. Init
   ------------------------------------------------------------
   Depends on: js/data.js (ALERT_TEMPLATES, SOURCES, SOURCE_COUNTS)
   ============================================================ */


/* ============ 1. STATE ============ */
let alerts = [];
let nextId = 1;
let playing = true;
let activeFilter = 'all';
let searchTerm = '';


/* ============ 2. HELPERS ============ */
function fmtTime(d) {
  const now = Date.now();
  const diff = Math.floor((now - d) / 1000);
  if (diff < 5) return 'just now';
  if (diff < 60) return diff + 's ago';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  return Math.floor(diff / 3600) + 'h ago';
}

function fmtClock(d) {
  return d.toTimeString().slice(0, 8);
}

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}


/* ============ 3. ALERT FACTORY + FEED SEEDING ============ */
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

function seedFeed() {
  for (let i = 0; i < 12; i++) {
    const a = makeAlert();
    a.timestamp = Date.now() - (12 - i) * (1500 + Math.random() * 4000);
    a.fresh = false;
    alerts.unshift(a);
  }
  alerts.reverse();
  renderFeed();
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
        !(a.mitre && a.mitre.toLowerCase().includes(term))) {
      return false;
    }
  }
  return true;
}

function renderFeed() {
  const list = document.getElementById('feed-list');
  const visible = alerts.filter(alertMatchesFilter);
  list.innerHTML = visible.map(renderAlert).join('');
  visible.forEach(a => {
    const el = document.getElementById(a.id);
    if (el) {
      el.addEventListener('click', () => toggleAlert(a.id));
    }
  });
}

function renderAlert(a) {
  const confPct = (a.confidence * 100).toFixed(1);
  const mitre = a.mitre ? `<span class="mitre">${a.mitre}</span>` : '';
  return `
    <div class="alert ${a.classification} ${a.fresh ? 'fresh' : ''} ${a.expanded ? 'expanded' : ''}" id="${a.id}">
      <span class="badge ${a.classification}">${a.classification === 'threat' ? 'Threat' : 'Benign'}</span>
      <span class="badge source">${a.source}</span>
      <div class="alert-content">
        <div class="alert-title">${a.title}</div>
        <div class="alert-meta">
          ${mitre}
          <span class="confidence">Confidence <strong>${confPct}%</strong></span>
          <span>Sev ${a.severity}</span>
        </div>
      </div>
      <div class="alert-time">${fmtTime(a.timestamp)}</div>
      <div class="alert-detail">
        <div class="detail-field"><div class="detail-label">Alert ID</div><div class="detail-value">${a.id}</div></div>
        <div class="detail-field"><div class="detail-label">Host</div><div class="detail-value">${a.host}</div></div>
        <div class="detail-field"><div class="detail-label">User</div><div class="detail-value">${a.user}</div></div>
        <div class="detail-field"><div class="detail-label">Wazuh severity</div><div class="detail-value">${a.severity} / 15</div></div>
        <div class="detail-field full"><div class="detail-label">Raw signal</div><div class="detail-value raw">${a.raw}</div></div>
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
  if (!playing) return;
  const a = makeAlert();
  alerts.unshift(a);
  if (alerts.length > 50) alerts.pop();
  setTimeout(() => { a.fresh = false; }, 600);
  renderFeed();
  bumpKpi();
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
  alerts = [];
  nextId = 1;
  seedFeed();
});


/* ============ 6. CLOCK ============ */
function tickClock() {
  document.getElementById('clock').textContent = fmtClock(new Date());
}
setInterval(tickClock, 1000);
tickClock();

// re-render times every 10s so "Xs ago" stays accurate
setInterval(renderFeed, 10000);


/* ============ 7. KPI COUNTERS ============ */
let kpiVolume = 24431;
let kpiThreats = 142;

function bumpKpi() {
  kpiVolume += Math.floor(Math.random() * 6) + 1;
  document.getElementById('kpi-volume').textContent = kpiVolume.toLocaleString();
  if (Math.random() < 0.18) {
    kpiThreats += 1;
    document.getElementById('kpi-threats').textContent = kpiThreats;
  }
}


/* ============ 8. VOLUME CHART (SVG, synthetic 24h series) ============ */
function drawVolumeChart() {
  const N = 48;            // 48 points = every 30 min over 24h
  const W = 320;
  const H = 70;
  const pad = 4;
  const vol = [];
  const threats = [];
  for (let i = 0; i < N; i++) {
    const hour = (i / 2);
    // diurnal pattern + noise
    const diurnal = 0.55 + 0.35 * Math.sin(((hour - 8) / 24) * Math.PI * 2);
    const v = Math.max(0.15, diurnal + (Math.random() - 0.5) * 0.15);
    vol.push(v);
    threats.push(v * (0.08 + Math.random() * 0.12));
  }
  const max = Math.max(...vol);
  const xs = i => pad + (i / (N - 1)) * (W - pad * 2);
  const ys = v => H - (v / max) * (H - 8);

  const linePts = vol.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');
  const areaD = `M${xs(0)},${H} L` + linePts.split(' ').join(' L') + ` L${xs(N - 1)},${H} Z`;
  const threatPts = threats.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');

  document.getElementById('volume-area').setAttribute('d', areaD);
  document.getElementById('volume-line').setAttribute('points', linePts);
  document.getElementById('threat-line').setAttribute('points', threatPts);
}


/* ============ 9. SOURCE BARS ============ */
function drawSources() {
  const container = document.getElementById('sources');
  const max = Math.max(...Object.values(SOURCE_COUNTS));
  container.innerHTML = SOURCES.map(s => {
    const v = SOURCE_COUNTS[s];
    const pct = (v / max) * 100;
    return `
      <div class="source-row">
        <div class="source-name">${s}</div>
        <div class="source-bar"><div class="source-fill" style="width:${pct.toFixed(0)}%"></div></div>
        <div class="source-value">${v.toLocaleString()}</div>
      </div>
    `;
  }).join('');
}


/* ============ 10. INIT ============ */
seedFeed();
drawVolumeChart();
drawSources();
setInterval(addNewAlert, 4000);
