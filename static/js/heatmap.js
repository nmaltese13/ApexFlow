/* =========================================================================
   GEX tab — single graph, clean diverging horizontal bars
   - Call γ green, Put γ red
   - King strike highlighted (yellow + glow)
   - Spot / Gamma-flip / Call-wall / Put-wall lines (only when in range)
   - Hover: full breakdown per strike
   ========================================================================= */

const STORAGE_KEY = 'gx_state_v3';
const C = {
  call:     'rgba(41, 216, 150, 0.85)',
  callEdge: 'rgba(41, 216, 150, 1)',
  put:      'rgba(255, 82, 111, 0.85)',
  putEdge:  'rgba(255, 82, 111, 1)',
  king:     'rgba(255, 213, 64, 0.95)',
  kingEdge: 'rgba(255, 213, 64, 1)',
  spot:     '#4cc9ff',
  flip:     '#ffb547',
  cw:       '#5be0b0',
  pw:       '#ff7a90',
};

let state = {
  sym: 'SPY',
  dte_min: 0,
  dte_max: 2,
  n_strikes: 25,
  auto: false,
};
try { state = { ...state, ...(JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') || {}) }; } catch {}
function persist() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch {} }

let chart = null;
let lastData = null;
let autoTimer = null;

/* ---------------- formatters ---------------- */
const fmtMoney = v => (v == null || !isFinite(v)) ? '—' : '$' + v.toFixed(2);
function fmtGex(v) {
  if (!v || !isFinite(v)) return '0';
  const a = Math.abs(v);
  const s = v < 0 ? '−' : '+';
  if (a >= 1e9) return s + '$' + (a / 1e9).toFixed(2) + 'B';
  if (a >= 1e6) return s + '$' + (a / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return s + '$' + (a / 1e3).toFixed(0) + 'K';
  return s + '$' + a.toFixed(0);
}
const fmtPct = v => (v == null || !isFinite(v)) ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

/* ---------------- toolbar ---------------- */
function applyState() {
  document.getElementById('gx-sym').value = state.sym;
  document.getElementById('gx-nstrikes').value = String(state.n_strikes);
  document.getElementById('gx-auto').checked = !!state.auto;
  document.querySelectorAll('#gx-band button').forEach(b => {
    b.classList.toggle('active',
      parseInt(b.dataset.min, 10) === state.dte_min &&
      parseInt(b.dataset.max, 10) === state.dte_max);
  });
  document.querySelectorAll('#gx-presets button').forEach(b => {
    b.classList.toggle('active', b.dataset.sym === state.sym);
  });
}

function wireToolbar() {
  const symInp = document.getElementById('gx-sym');
  symInp.addEventListener('change', () => {
    const v = symInp.value.trim().toUpperCase();
    if (v) { state.sym = v; persist(); applyState(); load(); }
  });
  symInp.addEventListener('keydown', e => { if (e.key === 'Enter') symInp.blur(); });

  document.querySelectorAll('#gx-presets button').forEach(b => {
    b.addEventListener('click', () => { state.sym = b.dataset.sym; persist(); applyState(); load(); });
  });
  document.querySelectorAll('#gx-band button').forEach(b => {
    b.addEventListener('click', () => {
      state.dte_min = parseInt(b.dataset.min, 10);
      state.dte_max = parseInt(b.dataset.max, 10);
      persist(); applyState(); load();
    });
  });
  document.getElementById('gx-nstrikes').addEventListener('change', e => {
    state.n_strikes = parseInt(e.target.value, 10) || 25; persist(); load();
  });
  document.getElementById('gx-refresh').addEventListener('click', load);
  document.getElementById('gx-auto').addEventListener('change', e => {
    state.auto = e.target.checked; persist();
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    if (state.auto) {
      autoTimer = setInterval(load, 60_000);
      if (typeof toast === 'function') toast('Auto-refresh on (60s)', 'ok');
    }
  });
  document.addEventListener('keydown', e => {
    if (e.target.matches('input,textarea,select')) return;
    if (e.key === 'r' || e.key === 'R') load();
  });
}

/* ---------------- API ---------------- */
async function load() {
  const btn = document.getElementById('gx-refresh');
  btn.disabled = true;
  document.getElementById('gx-meta').textContent = 'loading…';
  try {
    const url = `/api/heatseeker?symbols=${encodeURIComponent(state.sym)}` +
                `&dte_min=${state.dte_min}&dte_max=${state.dte_max}` +
                `&n_strikes=${state.n_strikes}&king_count=1&gatekeeper_count=0`;
    const r = await fetch(url);
    if (!r.ok) throw new Error('http ' + r.status);
    const j = await r.json();
    const sym = (j.symbols || [])[0];
    const layer = sym && (sym.layers || [])[0];
    if (!sym || !layer || !(layer.strikes || []).length) {
      if (typeof toast === 'function') toast('No chain in selected DTE', 'err');
      document.getElementById('gx-meta').textContent = 'no data';
      return;
    }
    lastData = {
      symbol: sym.symbol,
      spot: sym.spot,
      change_pct: sym.change_pct,
      strikes: layer.strikes,
      summary: layer.summary || {},
      levels: layer.levels || {},
      max_abs_gex: layer.max_abs_gex || 0,
      expiries_used: layer.expiries_used || [],
    };
    render(lastData);
  } catch (e) {
    if (typeof toast === 'function') toast('GEX load failed', 'err');
    console.error(e);
  } finally { btn.disabled = false; }
}

/* ---------------- render ---------------- */
function render(d) { renderMetrics(d); renderChart(d); }

function renderMetrics(d) {
  const setT = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  const setC = (id, cls) => { const el = document.getElementById(id); if (el) el.className = cls; };

  const spot = d.spot || 0;
  setT('gx-spot', fmtMoney(spot));
  const ch = d.change_pct || 0;
  setT('gx-chg', (ch >= 0 ? '▲ ' : '▼ ') + fmtPct(ch));
  setC('gx-chg', 'gx-stat-s mono ' + (ch >= 0 ? 'up' : 'dn'));

  const total = d.summary?.total_gex || 0;
  setT('gx-total', fmtGex(total));
  setC('gx-total', 'gx-stat-v mono ' + (total >= 0 ? 'up' : 'dn'));
  if (!isFinite(total) || !total) { setT('gx-regime', '—'); setC('gx-regime', 'gx-stat-s'); }
  else if (total > 0) { setT('gx-regime', 'PINNING regime'); setC('gx-regime', 'gx-stat-s up'); }
  else { setT('gx-regime', 'AMPLIFYING regime'); setC('gx-regime', 'gx-stat-s dn'); }

  const flip = d.summary?.gamma_flip ?? d.levels?.zero_gamma;
  setT('gx-flip', flip != null ? fmtMoney(flip) : '—');
  if (flip != null && spot) {
    const dist = ((spot - flip) / spot) * 100;
    setT('gx-flip-dist', (dist >= 0 ? '+' : '') + dist.toFixed(2) + '% from spot');
    setC('gx-flip-dist', 'gx-stat-s mono ' + (dist >= 0 ? 'up' : 'dn'));
  } else { setT('gx-flip-dist', '—'); setC('gx-flip-dist', 'gx-stat-s mono'); }

  const king = d.levels?.king;
  setT('gx-king', king != null ? fmtMoney(king) : '—');
  if (king != null) {
    const row = d.strikes.find(s => Math.abs(s.strike - king) < 0.01);
    setT('gx-king-gex', row ? fmtGex(row.total_gex) : '—');
  } else setT('gx-king-gex', '—');

  const cw = d.levels?.call_wall;
  setT('gx-cw', cw != null ? fmtMoney(cw) : '—');
  setT('gx-cw-dist', (cw != null && spot) ? fmtPct(((cw - spot) / spot) * 100) : '—');

  const pw = d.levels?.put_wall;
  setT('gx-pw', pw != null ? fmtMoney(pw) : '—');
  setT('gx-pw-dist', (pw != null && spot) ? fmtPct(((pw - spot) / spot) * 100) : '—');

  const exps = d.expiries_used || [];
  const meta = exps.length
    ? `${d.symbol} · ${exps.length} expir${exps.length > 1 ? 'ies' : 'y'} · ${exps[0]}${exps.length > 1 ? ' → ' + exps[exps.length - 1] : ''}`
    : `${d.symbol}`;
  document.getElementById('gx-meta').textContent = meta;
}

function renderChart(d) {
  const canvas = document.getElementById('gx-canvas');
  const ctx = canvas.getContext('2d');
  // Sort ascending; reverse Y axis so high strikes are at top.
  const strikes = [...d.strikes].sort((a, b) => a.strike - b.strike);
  const labels = strikes.map(s => s.strike);
  const minK = labels[0];
  const maxK = labels[labels.length - 1];

  // Per-bar colors — King = yellow; otherwise green/red gradient by side.
  // (We deliberately don't dim air pockets or outline gatekeepers — that just
  //  added noise. The King is special.)
  const callBg = strikes.map(s => s.node_type === 'king' ? C.king : C.call);
  const putBg  = strikes.map(s => s.node_type === 'king' ? C.king : C.put);
  const borderColor = strikes.map(s => s.node_type === 'king' ? C.kingEdge : 'transparent');
  const borderWidth = strikes.map(s => s.node_type === 'king' ? 2 : 0);

  // Y-axis tick density: with > 20 strikes, only label every Nth (and always spot/king).
  const tickStep = labels.length > 24 ? 3 : labels.length > 16 ? 2 : 1;
  const spot = d.spot;
  const spotIdx = closestIndex(labels, spot);
  const kingIdx = strikes.findIndex(s => s.node_type === 'king');

  const datasets = [
    {
      label: 'Call γ',
      data: strikes.map(s => s.call_gex / 1e6),
      backgroundColor: callBg,
      borderColor, borderWidth,
      barPercentage: 0.92, categoryPercentage: 0.96,
      stack: 'gex',
    },
    {
      label: 'Put γ',
      data: strikes.map(s => s.put_gex / 1e6),
      backgroundColor: putBg,
      borderColor, borderWidth,
      barPercentage: 0.92, categoryPercentage: 0.96,
      stack: 'gex',
    },
  ];

  // Resolve which level lines to draw — only those WITHIN visible strike range.
  // Dedupe levels that land on the same strike row to avoid label overlap.
  const flip = d.summary?.gamma_flip ?? d.levels?.zero_gamma;
  const cw   = d.levels?.call_wall;
  const pw   = d.levels?.put_wall;
  const inRange = v => v != null && isFinite(v) && v >= minK - 0.5 && v <= maxK + 0.5;
  const lines = [];
  if (inRange(spot)) lines.push({ price: spot, color: C.spot, label: 'SPOT  ' + fmtMoney(spot), dash: [6, 4], side: 'right', priority: 0 });
  if (inRange(flip)) lines.push({ price: flip, color: C.flip, label: 'FLIP  ' + fmtMoney(flip), dash: [3, 3], side: 'left',  priority: 1 });
  if (inRange(cw))   lines.push({ price: cw,   color: C.cw,   label: 'CW  '  + fmtMoney(cw),   dash: [2, 4], side: 'right', priority: 2 });
  if (inRange(pw))   lines.push({ price: pw,   color: C.pw,   label: 'PW  '  + fmtMoney(pw),   dash: [2, 4], side: 'right', priority: 2 });
  // Drop a line if a higher-priority line is within 1 strike (avoid stacked labels).
  lines.sort((a, b) => a.priority - b.priority);
  const finalLines = [];
  for (const L of lines) {
    if (finalLines.some(K => Math.abs(K.price - L.price) < 0.6)) continue;
    finalLines.push(L);
  }
  // "Off-chart" badges for levels that exist but are out of view.
  const offChart = [];
  if (flip != null && !inRange(flip)) offChart.push({ label: 'FLIP', price: flip, color: C.flip });
  if (cw != null && !inRange(cw))     offChart.push({ label: 'CW',   price: cw,   color: C.cw });
  if (pw != null && !inRange(pw))     offChart.push({ label: 'PW',   price: pw,   color: C.pw });

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      indexAxis: 'y',
      maintainAspectRatio: false,
      responsive: true,
      animation: { duration: 240 },
      interaction: { mode: 'index', intersect: false, axis: 'y' },
      layout: { padding: { top: 6, right: 6, bottom: 6, left: 4 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0c111c',
          borderColor: '#283149', borderWidth: 1,
          titleColor: '#7adcff', bodyColor: '#e6ebf5',
          padding: 10, cornerRadius: 6, displayColors: true,
          callbacks: {
            title: ctxs => 'Strike  ' + fmtMoney(+ctxs[0].label),
            label: c => {
              const sgn = c.parsed.x >= 0 ? '+' : '−';
              return c.dataset.label + '  ' + sgn + Math.abs(c.parsed.x).toFixed(2) + 'M';
            },
            afterBody: ctxs => {
              const idx = ctxs[0].dataIndex;
              const s = strikes[idx];
              const lines = ['Net  ' + fmtGex(s.total_gex)];
              if (s.node_type === 'king') lines.push('♛ KING NODE');
              if (spot) {
                const dist = ((s.strike - spot) / spot) * 100;
                lines.push('Δ from spot  ' + (dist >= 0 ? '+' : '') + dist.toFixed(2) + '%');
              }
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          grid: {
            color: ctx => ctx.tick.value === 0 ? '#3a4258' : '#161c2a',
            lineWidth: ctx => ctx.tick.value === 0 ? 1.5 : 1,
            drawTicks: false,
          },
          border: { display: false },
          ticks: {
            color: '#5a6479',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: v => (v >= 0 ? '+' : '') + v + 'M',
          },
          title: { display: true, text: 'Per-strike GEX  ($M)', color: '#5a6479', font: { size: 11 } },
        },
        y: {
          stacked: true,
          reverse: true,
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: ctx => {
              if (ctx.index === kingIdx)  return C.kingEdge;
              if (ctx.index === spotIdx)  return C.spot;
              return '#7d879c';
            },
            font: ctx => ({
              family: 'JetBrains Mono',
              size: 10,
              weight: (ctx.index === kingIdx || ctx.index === spotIdx) ? 700 : 500,
            }),
            callback: (val, idx) => {
              if (idx === spotIdx || idx === kingIdx) return labels[idx].toFixed(0);
              if (idx % tickStep !== 0) return '';
              return labels[idx].toFixed(0);
            },
            autoSkip: false,
          },
          title: { display: true, text: 'Strike', color: '#5a6479', font: { size: 11 } },
        },
      },
    },
    plugins: [{
      id: 'gx-overlays',
      afterDatasetsDraw: chart => {
        const x = chart.scales.x;
        const y = chart.scales.y;
        const c = chart.ctx;

        // Interpolate y-pixel for a price, between two adjacent strike rows
        const yForPrice = (price) => {
          if (price <= labels[0]) return y.getPixelForValue(labels[0]);
          if (price >= labels[labels.length - 1]) return y.getPixelForValue(labels[labels.length - 1]);
          for (let i = 0; i < labels.length - 1; i++) {
            if (price >= labels[i] && price <= labels[i + 1]) {
              const y0 = y.getPixelForValue(labels[i]);
              const y1 = y.getPixelForValue(labels[i + 1]);
              const t = (price - labels[i]) / (labels[i + 1] - labels[i]);
              return y0 + (y1 - y0) * t;
            }
          }
          return y.getPixelForValue(labels[0]);
        };

        finalLines.forEach(L => {
          const yPx = yForPrice(L.price);
          // line
          c.save();
          c.strokeStyle = L.color;
          c.lineWidth = 1.4;
          c.setLineDash(L.dash);
          c.beginPath();
          c.moveTo(x.left, yPx);
          c.lineTo(x.right, yPx);
          c.stroke();
          // label badge (filled chip for legibility)
          const text = L.label;
          c.font = 'bold 10px "JetBrains Mono", monospace';
          c.setLineDash([]);
          const padX = 6, padY = 3;
          const w = c.measureText(text).width + padX * 2;
          const h = 16;
          const tx = L.side === 'left' ? x.left + 6 : x.right - 6 - w;
          const ty = yPx - h / 2;
          c.fillStyle = 'rgba(8, 12, 20, 0.92)';
          c.strokeStyle = L.color;
          c.lineWidth = 1;
          c.beginPath();
          if (c.roundRect) c.roundRect(tx, ty, w, h, 3); else c.rect(tx, ty, w, h);
          c.fill();
          c.stroke();
          c.fillStyle = L.color;
          c.textBaseline = 'middle';
          c.textAlign = 'left';
          c.fillText(text, tx + padX, ty + h / 2 + 0.5);
          c.restore();
        });

        // Off-chart level pills (top-right corner)
        if (offChart.length) {
          c.save();
          c.font = 'bold 10px "JetBrains Mono", monospace';
          let cx = x.right - 6;
          const cy = y.top + 8;
          for (let i = offChart.length - 1; i >= 0; i--) {
            const o = offChart[i];
            const text = `${o.label} ${fmtMoney(o.price)} ↗`;
            const w = c.measureText(text).width + 14;
            const h = 18;
            cx -= w;
            c.fillStyle = 'rgba(8, 12, 20, 0.85)';
            c.strokeStyle = o.color;
            c.lineWidth = 1;
            c.beginPath();
            if (c.roundRect) c.roundRect(cx, cy, w, h, 4); else c.rect(cx, cy, w, h);
            c.fill(); c.stroke();
            c.fillStyle = o.color;
            c.textBaseline = 'middle';
            c.textAlign = 'left';
            c.fillText(text, cx + 7, cy + h / 2 + 0.5);
            cx -= 6;
          }
          c.restore();
        }
      },
    }],
  });
}

function closestIndex(arr, target) {
  if (!arr.length || target == null) return -1;
  let best = 0, bestD = Math.abs(arr[0] - target);
  for (let i = 1; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target);
    if (d < bestD) { best = i; bestD = d; }
  }
  return best;
}

/* ---------------- init ---------------- */
applyState();
wireToolbar();
load();
if (state.auto) autoTimer = setInterval(load, 60_000);
