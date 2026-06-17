/* =========================================================================
   AGV Rail Inspection — dashboard frontend (vanilla JS).
   Live over Socket.IO; Leaflet map (OpenStreetMap); Chart.js charts.
   Seeds from REST (/api/defects) on load, then updates live.
   ========================================================================= */
"use strict";

const SEV_RANK = { High: 3, Medium: 2, Low: 1 };
const DEFAULT_CENTER = [3.1390, 101.6869]; // Kuala Lumpur (auto-fits to data)
const INSPECTED_COLOR = "#3D9BE0";         // inspection-progress gauge — kept distinct from the severity palette
const GAUGE_ARC_LEN = 188.5;               // px length of a full 270° gauge sweep (2·π·r · 270/360, r = 40)

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const sevColor = (s) => css(s === "High" ? "--high" : s === "Medium" ? "--med" : "--low") || "#888";
const $ = (id) => document.getElementById(id);
const setText = (id, v) => { const el = $(id); if (el) el.textContent = v; };

// ---- state -------------------------------------------------------------
const defects = new Map();          // key -> normalised defect
const markers = new Map();          // key -> Leaflet circleMarker
const trackPts = [];                // AGV path (from telemetry)
let map, trackLine, agvMarker, sevChart, classChart;
let sortKey = "urgency_score", sortDir = -1;
let didFit = false;
let lastChainage = null, stripRenderedMax = null;

// ---- normalise (live MQTT shape vs REST row shape) ---------------------
function norm(d) {
  const loc = d.location || { lat: d.lat, lng: d.lng, chainage_m: d.chainage_m };
  return {
    detection_id: d.detection_id || d.defect_key || ("k" + d.track_id),
    track_id: (d.track_id ?? -1),
    defect_class: d.defect_class,
    confidence: Number(d.confidence) || 0,
    severity: d.severity,
    urgency_score: Number(d.urgency_score) || 0,
    recommended_action: d.recommended_action || "",
    lat: Number(loc.lat), lng: Number(loc.lng), chainage_m: Number(loc.chainage_m) || 0,
    timestamp: d.timestamp || d.last_seen || d.first_seen || "",
    image_ref: d.image_ref || "",
    frame_count: d.frame_count,
  };
}
const keyOf = (d) => (d.track_id != null && d.track_id >= 0) ? "t" + d.track_id : "d" + d.detection_id;

// ---- map ---------------------------------------------------------------
function initMap() {
  map = L.map("map", { zoomControl: true, attributionControl: true }).setView(DEFAULT_CENTER, 15);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: '&copy; OpenStreetMap contributors',
  }).addTo(map);
  trackLine = L.polyline([], { color: css("--accent"), weight: 4, opacity: .85 }).addTo(map);
}

function popupHtml(d) {
  return `<div class="popup-h">${escapeHtml(d.defect_class)} <span style="color:${sevColor(d.severity)}">&#9679; ${d.severity}</span></div>
    <div class="popup-row">confidence: <b>${d.confidence.toFixed(2)}</b> &middot; urgency: <b>${d.urgency_score}</b></div>
    <div class="popup-row">${escapeHtml(d.recommended_action)}</div>
    <div class="popup-row">${d.lat.toFixed(5)}, ${d.lng.toFixed(5)} &middot; chainage ${d.chainage_m.toFixed(1)} m</div>
    <div class="popup-row" style="color:#777">${fmtTime(d.timestamp)}</div>`;
}

function upsertMarker(key, d) {
  if (!Number.isFinite(d.lat) || !Number.isFinite(d.lng)) return;
  const color = sevColor(d.severity);
  if (markers.has(key)) {
    const m = markers.get(key);
    m.setLatLng([d.lat, d.lng]); m.setStyle({ color, fillColor: color });
    m.setPopupContent(popupHtml(d));
  } else {
    const m = L.circleMarker([d.lat, d.lng], { radius: 7, color, fillColor: color, fillOpacity: .85, weight: 2 });
    m.addTo(map).bindPopup(popupHtml(d));
    markers.set(key, m);
  }
}

function maybeFit() {
  if (didFit || markers.size === 0) return;
  const grp = L.featureGroup([...markers.values()]);
  try { map.fitBounds(grp.getBounds().pad(0.3)); didFit = true; } catch (e) {}
}

// ---- ingest defect -----------------------------------------------------
function ingest(raw) {
  const d = norm(raw);
  if (!d.defect_class) return;
  defects.set(keyOf(d), d);
  upsertMarker(keyOf(d), d);
  renderCards();
  renderTable();
  refreshClassFilter();
  renderStrip();
}

// ---- KPI gauges --------------------------------------------------------
const TRACK_TOTAL_M = 80;   // demo corridor length for the "inspected" gauge
let lastBattery = 100;      // last known battery % (simulated); lets gauges seed without live telemetry
function gaugeLevel(pct, goodHigh) {
  const p = goodHigh ? 100 - pct : pct;   // goodHigh: a high value is GOOD (e.g. battery)
  return p >= 66 ? css("--high") : p >= 33 ? css("--med") : css("--low");
}
function setGauge(arcId, valId, pct, color, text) {
  const len = Math.max(0, Math.min(100, pct)) / 100 * GAUGE_ARC_LEN;
  const arc = $(arcId), v = $(valId);
  if (arc) { arc.style.stroke = color; arc.style.color = color; arc.setAttribute("stroke-dasharray", len.toFixed(1) + " 999"); }
  if (v) v.textContent = text;
}

// ---- extra panels: triage tiles / work queue / track histogram --------
let trackChart = null;
function bandOf(u) { return u >= 75 ? "imm" : u >= 50 ? "sch" : u >= 25 ? "rou" : "mon"; }
function renderExtras(arr) {
  // urgency triage counts
  const c = { imm: 0, sch: 0, rou: 0, mon: 0 };
  arr.forEach((d) => { c[bandOf(Number(d.urgency_score) || 0)]++; });
  setText("t-imm", c.imm); setText("t-sch", c.sch); setText("t-rou", c.rou); setText("t-mon", c.mon);

  // priority work queue (highest urgency first)
  const q = $("queue");
  if (q) {
    if (!arr.length) { q.innerHTML = '<div class="q-empty">No defects yet.</div>'; }
    else {
      const top = [...arr].sort((a, b) => (Number(b.urgency_score) || 0) - (Number(a.urgency_score) || 0)).slice(0, 10);
      q.innerHTML = top.map((d) => {
        const name = escapeHtml(String(d.defect_class || "").replace(/_/g, " "));
        const act = escapeHtml(d.recommended_action || "");
        const ref = String(d.image_ref || "");
        const thumb = ref ? "/crops/" + ref.split(/[\\/]/).pop() : "";
        return '<div class="q-item">'
          + (thumb ? '<img class="q-thumb" src="' + thumb + '" alt="" onerror="this.style.visibility=\'hidden\'">'
                   : '<span class="q-sev" style="background:' + sevColor(d.severity) + '"></span>')
          + '<div class="q-main"><div class="q-top"><b>' + name + '</b><span class="q-ch mono">'
          + (Number(d.chainage_m) || 0).toFixed(1) + ' m</span></div>'
          + '<div class="q-act" title="' + act + '">' + act + '</div></div>'
          + '<span class="badge ' + d.severity + '" style="align-self:center">' + d.severity + '</span>'
          + '<span class="q-urg mono">' + Math.round(Number(d.urgency_score) || 0) + '</span></div>';
      }).join("");
    }
    setText("q-count", arr.length + " open");
  }

  // defects along the track (count per 10 m)
  const BIN = 10;
  const maxCh = arr.length ? Math.max(20, ...arr.map((d) => Number(d.chainage_m) || 0)) : 20;
  const n = Math.max(1, Math.ceil(maxCh / BIN));
  const bins = new Array(n).fill(0);
  arr.forEach((d) => { bins[Math.min(n - 1, Math.floor((Number(d.chainage_m) || 0) / BIN))]++; });
  const labels = bins.map((_, i) => (i * BIN) + "–" + ((i + 1) * BIN));
  if (!trackChart) {
    trackChart = new Chart($("trackChart"), {
      type: "bar",
      data: { labels, datasets: [{ data: bins, backgroundColor: css("--chart"), borderRadius: 3 }] },
      options: { plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: css("--text"), font: { size: 9 } }, grid: { display: false } },
                  y: { beginAtZero: true, ticks: { color: css("--text"), precision: 0, font: { size: 10 } }, grid: { color: css("--border") } } },
        responsive: true, maintainAspectRatio: false },
    });
  } else {
    trackChart.data.labels = labels;
    trackChart.data.datasets[0].data = bins;
    trackChart.data.datasets[0].backgroundColor = css("--chart");
    trackChart.options.scales.x.ticks.color = css("--text");
    trackChart.options.scales.y.ticks.color = css("--text");
    trackChart.update("none");
    return;
  }
}

// ---- stat cards + charts ----------------------------------------------
function renderCards() {
  const arr = [...defects.values()];
  const c = { High: 0, Medium: 0, Low: 0 };
  arr.forEach((d) => { c[d.severity] = (c[d.severity] || 0) + 1; });
  setText("c-total", arr.length);
  setText("c-high", c.High); setText("c-med", c.Medium); setText("c-low", c.Low);
  // track health score: severity-weighted condition index (documented, honest)
  const T = arr.length || 1;
  const health = Math.max(0, Math.round(100 * (1 - 0.6 * c.High / T - 0.3 * c.Medium / T - 0.1 * c.Low / T)));
  const hv = $("c-health");
  if (hv) { hv.textContent = arr.length ? health + "%" : "—"; hv.style.color = health >= 75 ? css("--low") : health >= 50 ? css("--med") : css("--high"); }
  setText("c-health-l", arr.length ? (health >= 75 ? "Good" : health >= 50 ? "Fair" : "Poor") : "");
  [["c-high", c.High], ["c-med", c.Medium], ["c-low", c.Low]].forEach(([id, n]) => {
    const card = $(id) && $(id).closest(".card");
    if (card) card.classList.toggle("is-zero", (Number(n) || 0) === 0);
  });
  updateCharts(c, arr);
  const urg = arr.length ? arr.reduce((s, d) => s + (Number(d.urgency_score) || 0), 0) / arr.length : 0;
  setGauge("g-urg-arc", "g-urg-v", urg, gaugeLevel(urg, false), String(Math.round(urg)));
  // seed battery + inspected from reliable data (so they don't blank when telemetry is absent)
  const maxCh = arr.length ? Math.max(...arr.map((d) => Number(d.chainage_m) || 0)) : 0;
  const prog = Math.min(100, maxCh / TRACK_TOTAL_M * 100);
  setGauge("g-prog-arc", "g-prog-v", prog, INSPECTED_COLOR, Math.round(prog) + "%");
  setGauge("g-batt-arc", "g-batt-v", lastBattery, gaugeLevel(lastBattery, true), Math.round(lastBattery) + "%");
  renderExtras(arr);
}

// center "total" readout drawn inside the severity doughnut -> hero gauge
const gaugeCenter = {
  id: "gaugeCenter",
  afterDatasetsDraw(chart) {
    const ds = chart.data.datasets[0];
    const total = (ds.data || []).reduce((a, b) => a + (Number(b) || 0), 0);
    const area = chart.chartArea; if (!area) return;
    const cx = (area.left + area.right) / 2, cy = (area.top + area.bottom) / 2;
    const ctx = chart.ctx;
    ctx.save();
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = css("--text");
    ctx.font = "700 26px ui-monospace, Consolas, monospace";
    ctx.fillText(String(total), cx, cy - 6);
    ctx.fillStyle = css("--muted");
    ctx.font = "600 9px Inter, system-ui, sans-serif";
    ctx.fillText("DEFECTS", cx, cy + 13);
    ctx.restore();
  },
};

function updateCharts(sev, arr) {
  const colors = [sevColor("High"), sevColor("Medium"), sevColor("Low")];
  if (!sevChart) {
    sevChart = new Chart($("sevChart"), {
      type: "doughnut",
      data: { labels: ["High", "Medium", "Low"], datasets: [{ data: [0, 0, 0], backgroundColor: colors, borderWidth: 0 }] },
      options: { plugins: { legend: { labels: { color: css("--muted"), boxWidth: 12 }, position: "bottom" } }, cutout: "68%", responsive: true, maintainAspectRatio: false },
      plugins: [gaugeCenter],
    });
  }
  sevChart.data.datasets[0].data = [sev.High, sev.Medium, sev.Low];
  sevChart.data.datasets[0].backgroundColor = colors;
  sevChart.options.plugins.legend.labels.color = css("--text");
  sevChart.update("none");

  // defects by class — all seven contract classes (zeros included), horizontal bars
  const byClass = {};
  arr.forEach((d) => { byClass[d.defect_class] = (byClass[d.defect_class] || 0) + 1; });
  const CLASS_ORDER = ["broken_fastener", "missing_fastener", "loose_fastener", "crack", "spalling", "squat", "corrugation"];
  const CLASS_LABELS = { broken_fastener: "Broken fastener", missing_fastener: "Missing fastener", loose_fastener: "Loose fastener", crack: "Crack", spalling: "Spalling", squat: "Squat", corrugation: "Corrugation" };
  const labels = CLASS_ORDER.map((c) => CLASS_LABELS[c]);
  if (!classChart) {
    classChart = new Chart($("classChart"), {
      type: "bar",
      data: { labels, datasets: [{ data: [], backgroundColor: css("--chart"), borderRadius: 3, barThickness: 16 }] },
      options: {
        indexAxis: "y",
        layout: { padding: { left: 2, right: 16 } },
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true, grace: "12%", ticks: { color: css("--text"), precision: 0, font: { size: 11 } }, grid: { color: css("--border") } },
                  y: { ticks: { color: css("--text"), font: { size: 10 }, padding: 0, autoSkip: false }, grid: { display: false } } },
        responsive: true, maintainAspectRatio: false,
      },
    });
  }
  classChart.data.labels = labels;
  classChart.data.datasets[0].data = CLASS_ORDER.map((c) => byClass[c] || 0);
  classChart.data.datasets[0].backgroundColor = css("--chart");
  classChart.options.scales.x.ticks.color = css("--text");
  classChart.options.scales.x.grid.color = css("--border");
  classChart.options.scales.y.ticks.color = css("--text");
  classChart.update("none");
}

// ---- telemetry ---------------------------------------------------------
function onTelemetry(t) {
  setText("c-fps", (Number(t.fps) || 0).toFixed(1));
  setText("c-latency", Math.round(Number(t.inference_ms) || 0));
  setText("c-distance", (Number(t.chainage_m) || 0).toFixed(1));
  setText("hb-fps", (Number(t.fps) || 0).toFixed(1));
  setText("hb-dist", (Number(t.chainage_m) || 0).toFixed(1) + " m");
  setText("hb-batt", Math.round(Number(t.battery_pct) || 0) + "%");
  const bp = Number(t.battery_pct) || 0; lastBattery = bp;
  setGauge("g-batt-arc", "g-batt-v", bp, gaugeLevel(bp, true), Math.round(bp) + "%");
  const prog = Math.min(100, (Number(t.chainage_m) || 0) / TRACK_TOTAL_M * 100);
  setGauge("g-prog-arc", "g-prog-v", prog, INSPECTED_COLOR, Math.round(prog) + "%");
  if (Number.isFinite(t.lat) && Number.isFinite(t.lng)) {
    trackPts.push([t.lat, t.lng]);
    trackLine.setLatLngs(trackPts);
    if (agvMarker) { agvMarker.setLatLng([t.lat, t.lng]); }
    else { agvMarker = L.circleMarker([t.lat, t.lng], { radius: 6, color: css("--accent"), fillColor: css("--bg"), fillOpacity: 1, weight: 3 }).addTo(map).bindTooltip("AGV"); }
    maybeFit();
  }
  lastChainage = Number(t.chainage_m) || 0;
  if (stripMax() !== stripRenderedMax) renderStrip(); else updatePlayhead();
}

// ---- status ------------------------------------------------------------
function onStatus(s) {
  const el = $("agv-state");
  if (el) el.textContent = (s.state || "—").toUpperCase();
}

// ---- live frame --------------------------------------------------------
function updateFrame(d) {
  const img = $("liveframe"), empty = $("frame-empty"), frame = $("live-frame");
  img.style.display = ""; if (empty) empty.style.display = "none";
  img.src = "/latest_frame.jpg?t=" + Date.now();
  frame.style.borderColor = sevColor(d.severity);
  $("frame-tag").innerHTML = `<span class="sw" style="background:${sevColor(d.severity)}"></span>${escapeHtml(d.defect_class)} &middot; ${d.confidence.toFixed(2)}`;
  $("frame-tag").style.display = "";
  setText("frame-meta-class", d.defect_class);
  setText("frame-meta-sev", d.severity);
  setText("frame-meta-urg", d.urgency_score);
  setText("frame-meta-chain", d.chainage_m.toFixed(1) + " m");
}

// ---- table -------------------------------------------------------------
function fmtTime(ts) {
  if (!ts) return "—";
  const dt = new Date(ts);
  return isNaN(dt) ? ts : dt.toLocaleTimeString();
}
function sortVal(d, key) {
  if (key === "severity") return SEV_RANK[d.severity] || 0;
  if (key === "timestamp") { const t = Date.parse(d.timestamp); return isNaN(t) ? 0 : t; }
  const v = d[key];
  return typeof v === "number" ? v : String(v ?? "").toLowerCase();
}
function renderTable() {
  const fSev = $("filter-sev").value;
  const fClass = $("filter-class").value;
  let arr = [...defects.values()];
  if (fSev !== "all") arr = arr.filter((d) => d.severity === fSev);
  if (fClass !== "all") arr = arr.filter((d) => d.defect_class === fClass);
  arr.sort((a, b) => {
    const x = sortVal(a, sortKey), y = sortVal(b, sortKey);
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  setText("row-count", arr.length + " row" + (arr.length === 1 ? "" : "s"));

  const tb = $("tbody");
  if (arr.length === 0) { tb.innerHTML = `<tr class="empty-row"><td colspan="8">No detections yet — start the pipeline (<span class="mono">python run.py</span>).</td></tr>`; return; }
  tb.innerHTML = arr.map((d) => `
    <tr>
      <td class="mono">${fmtTime(d.timestamp)}</td>
      <td>${escapeHtml(d.defect_class)}</td>
      <td class="mono">${d.confidence.toFixed(2)}</td>
      <td><span class="badge ${d.severity}">${d.severity}</span></td>
      <td class="urg mono">${d.urgency_score}</td>
      <td class="action">${escapeHtml(d.recommended_action)}</td>
      <td class="mono">${Number.isFinite(d.lat) ? d.lat.toFixed(5) + ", " + d.lng.toFixed(5) : "—"}</td>
      <td class="mono">${d.chainage_m.toFixed(1)}</td>
    </tr>`).join("");
}
function escapeHtml(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function refreshClassFilter() {
  const sel = $("filter-class");
  const have = new Set([...sel.options].map((o) => o.value));
  [...new Set([...defects.values()].map((d) => d.defect_class))].sort().forEach((cls) => {
    if (!have.has(cls)) { const o = document.createElement("option"); o.value = o.textContent = cls; sel.appendChild(o); }
  });
}

// ---- chainage strip (linear track diagram) -----------------------------
function stripMax() {
  let m = 10;
  defects.forEach((d) => { if (Number.isFinite(d.chainage_m) && d.chainage_m > m) m = d.chainage_m; });
  if (lastChainage != null && lastChainage > m) m = lastChainage;
  const step = m <= 50 ? 10 : m <= 200 ? 25 : m <= 1000 ? 100 : 500;
  return Math.max(step, Math.ceil(m / step) * step);
}
function pctOf(v, max) { return Math.max(0, Math.min(100, (v / max) * 100)); }
function renderStrip() {
  const max = stripMax();
  stripRenderedMax = max;
  const marks = $("strip-marks");
  if (!marks) return;
  marks.innerHTML = "";
  [...defects.values()].forEach((d) => {
    if (!Number.isFinite(d.chainage_m)) return;
    const el = document.createElement("div");
    el.className = "strip-mark";
    el.style.left = pctOf(d.chainage_m, max) + "%";
    el.style.setProperty("--c", sevColor(d.severity));
    el.dataset.key = keyOf(d);
    el.title = `${d.defect_class} · ${d.severity} · ${d.chainage_m.toFixed(1)} m`;
    el.addEventListener("mouseenter", () => showTip(el, d));
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("click", () => selectDefect(keyOf(d), d));
    marks.appendChild(el);
  });
  const ax = $("strip-axis");
  if (ax) {
    ax.innerHTML = "";
    for (let i = 0; i <= 5; i++) {
      const s = document.createElement("span");
      s.textContent = Math.round((max * i) / 5) + " m";
      ax.appendChild(s);
    }
  }
  updatePlayhead(max);
}
function updatePlayhead(max) {
  max = max || stripRenderedMax || stripMax();
  const agv = $("strip-agv");
  if (!agv) return;
  if (lastChainage == null) { agv.style.display = "none"; return; }
  agv.style.display = "";
  agv.style.left = pctOf(lastChainage, max) + "%";
  agv.title = "AGV @ " + lastChainage.toFixed(1) + " m";
}
function showTip(el, d) {
  const tip = $("strip-tip");
  if (!tip) return;
  tip.innerHTML = popupHtml(d);
  tip.style.left = el.style.left;
  tip.style.display = "block";
}
function hideTip() { const t = $("strip-tip"); if (t) t.style.display = "none"; }
function selectDefect(key, d) {
  const m = markers.get(key);
  if (m) { try { map.setView(m.getLatLng(), Math.max(map.getZoom(), 17)); m.openPopup(); } catch (e) {} }
  const det = $("strip-detail");
  if (det) det.innerHTML = popupHtml(d);
  document.querySelectorAll(".strip-mark.sel").forEach((x) => x.classList.remove("sel"));
  const marks = $("strip-marks");
  const el = marks && marks.querySelector('.strip-mark[data-key="' + key + '"]');
  if (el) el.classList.add("sel");
}

// ---- theme toggle ------------------------------------------------------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("agv-theme-v3", theme);
  $("theme-btn").textContent = theme === "dark" ? "☀ Light" : "☾ Dark";
  // recolour live elements
  if (trackLine) trackLine.setStyle({ color: css("--accent") });
  if (sevChart || classChart) renderCards();
  renderStrip();
}

// ---- wire up -----------------------------------------------------------
function wireControls() {
  $("filter-sev").addEventListener("change", renderTable);
  $("filter-class").addEventListener("change", renderTable);
  document.querySelectorAll("thead th[data-key]").forEach((th) => {
    th.addEventListener("click", () => {
      const k = th.getAttribute("data-key");
      if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = (k === "severity" || k === "urgency_score" || k === "confidence" || k === "timestamp") ? -1 : 1; }
      document.querySelectorAll("thead th .arr").forEach((a) => a.textContent = "");
      const arr = th.querySelector(".arr"); if (arr) arr.textContent = sortDir < 0 ? "▼" : "▲";
      renderTable();
    });
  });
  $("theme-btn").addEventListener("click", () => applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));
}

function start() {
  applyTheme(localStorage.getItem("agv-theme-v3") || "dark");
  initMap();
  wireControls();

  const socket = io();
  const conn = $("conn");
  socket.on("connect", () => { conn.className = "pill live"; conn.querySelector(".lbl").textContent = "LIVE"; });
  socket.on("disconnect", () => { conn.className = "pill off"; conn.querySelector(".lbl").textContent = "OFFLINE"; });
  socket.on("detection", (d) => { ingest(d); updateFrame(norm(d)); });
  socket.on("telemetry", onTelemetry);
  socket.on("status", onStatus);

  // Seed from the persisted register so the page isn't empty on load.
  fetch("/api/defects").then((r) => r.json()).then((rows) => {
    rows.forEach(ingest);
    maybeFit();
    if (rows.length) { const last = norm(rows[0]); updateFrame(last); }
  }).catch(() => {});
  fetch("/api/state").then((r) => r.json()).then((s) => { if (s.telemetry) onTelemetry(s.telemetry); if (s.status) onStatus(s.status); }).catch(() => {});

  // keep the live frame fresh even between detections
  setInterval(() => { const img = $("liveframe"); if (img && img.style.display !== "none") img.src = "/latest_frame.jpg?t=" + Date.now(); }, 4000);
}

document.addEventListener("DOMContentLoaded", start);
