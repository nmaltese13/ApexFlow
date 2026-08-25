/* =========================================================================
   VOLATILITY SURFACE — term structure + skew

   Reads /api/iv_surface/{sym}. The confidence flag is the reason this page
   exists: a single quoted implied vol looks equally authoritative whether
   it came from a hundred agreeing strikes or from a penny-wide 0DTE
   contract whose neighbours disagree by a factor of four. Points that
   failed that test are hidden rather than plotted alongside good ones.
   ========================================================================= */

const VL_KEY = 'vl_state_v1';
let vlState = { sym: 'SPY', count: 8, showPoor: false };
try {
  vlState = { ...vlState, ...(JSON.parse(localStorage.getItem(VL_KEY) || 'null') || {}) };
} catch { /* private mode — defaults are fine */ }
const vlPersist = () => { try { localStorage.setItem(VL_KEY, JSON.stringify(vlState)); } catch {} };

let vlChart = null;
let vlData = null;
const $vl = id => document.getElementById(id);

const pct = v => (v == null || !isFinite(v)) ? '—' : (v * 100).toFixed(1) + '%';
const CONF = {
  good: { label: 'good', cls: 'conf-good' },
  fair: { label: 'fair', cls: 'conf-fair' },
  poor: { label: 'low', cls: 'conf-poor' },
  none: { label: 'none', cls: 'conf-poor' },
};

function vlApply() {
  $vl('vl-sym').value = vlState.sym;
  $vl('vl-count').value = String(vlState.count);
  $vl('vl-showpoor').checked = !!vlState.showPoor;
  document.querySelectorAll('#vl-presets button').forEach(b =>
    b.classList.toggle('active', b.dataset.sym === vlState.sym));
}

function vlWire() {
  const sym = $vl('vl-sym');
  sym.addEventListener('change', () => {
    const v = sym.value.trim().toUpperCase();
    if (v) { vlState.sym = v; vlPersist(); vlApply(); vlLoad(); }
  });
  sym.addEventListener('keydown', e => { if (e.key === 'Enter') sym.blur(); });

  document.querySelectorAll('#vl-presets button').forEach(b =>
    b.addEventListener('click', () => {
      vlState.sym = b.dataset.sym; vlPersist(); vlApply(); vlLoad();
    }));

  $vl('vl-count').addEventListener('change', e => {
    vlState.count = parseInt(e.target.value, 10); vlPersist(); vlLoad();
  });
  $vl('vl-showpoor').addEventListener('change', e => {
    vlState.showPoor = e.target.checked; vlPersist();
    if (vlData) vlRender(vlData);      // filter only — no refetch needed
  });
  $vl('vl-refresh').addEventListener('click', vlLoad);
}

async function vlLoad() {
  try {
    const r = await fetch(`/api/iv_surface/${encodeURIComponent(vlState.sym)}?max_expiries=${vlState.count}`);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    vlData = await r.json();
  } catch (e) {
    vlEmpty(`Could not load ${vlState.sym}: ${e.message}`);
    return;
  }
  if (!vlData.points || !vlData.points.length) {
    vlEmpty(`No option chain for ${vlState.sym}.`);
    return;
  }
  vlRender(vlData);
}

function vlEmpty(msg) {
  const el = $vl('vl-shape');
  el.className = 'dg-regime dg-regime-unknown';
  el.querySelector('.dg-regime-text').textContent = msg;
  ['vl-atm', 'vl-front', 'vl-back', 'vl-rr'].forEach(i => { $vl(i).textContent = '—'; });
  if (vlChart) { vlChart.destroy(); vlChart = null; }
  $vl('vl-table').querySelector('tbody').innerHTML =
    `<tr><td colspan="5" class="empty-sm">${msg}</td></tr>`;
}

function vlRender(d) {
  const usable = d.points.filter(p => p.quality === 'good' || p.quality === 'fair');
  const shown = vlState.showPoor ? d.points : usable;

  // --- shape banner ---
  const el = $vl('vl-shape');
  const shape = d.shape || 'unknown';
  el.className = 'dg-regime ' + (
    shape === 'backwardation' ? 'dg-regime-negative-gamma'
      : shape === 'contango' ? 'dg-regime-positive-gamma' : 'dg-regime-unknown');
  el.querySelector('.dg-regime-text').innerHTML =
    shape === 'backwardation'
      ? `<strong>Backwardation.</strong> Near-dated vol is bid over far-dated — the options market is pricing an event inside the front window.`
      : shape === 'contango'
        ? `<strong>Contango.</strong> Far-dated vol above near-dated, the normal resting state.`
        : shape === 'flat'
          ? `<strong>Flat.</strong> No meaningful slope across the curve.`
          : `Shape unknown — too few trustworthy expiries to say.`;

  // --- stats ---
  const atm = d.atm || {};
  $vl('vl-atm').textContent = pct(atm.iv);
  const c = CONF[atm.quality] || CONF.none;
  $vl('vl-atm-sub').innerHTML =
    `<span class="conf-chip ${c.cls}">${c.label}</span> from ${atm.expiry || '—'}`;
  $vl('vl-front').textContent = pct(d.front_iv);
  $vl('vl-back').textContent = pct(d.back_iv);

  const rr = d.skew || {};
  const hasRR = rr.quality === 'good' && isFinite(rr.rr_25d);
  $vl('vl-rr').textContent = hasRR ? (rr.rr_25d >= 0 ? '+' : '') + rr.rr_25d.toFixed(2) : '—';
  $vl('vl-rr').className = 'dg-stat-v mono ' + (hasRR ? (rr.rr_25d >= 0 ? 'pos' : 'neg') : '');
  $vl('vl-rr-sub').textContent = hasRR
    ? (rr.rr_25d >= 0 ? 'puts bid over calls' : 'calls bid over puts')
    : 'no 25Δ strikes available';

  vlChart_draw(shown);
  vlTable(d.points);
}

function vlChart_draw(points) {
  const canvas = $vl('vl-chart');
  if (!canvas || typeof Chart === 'undefined') return;
  if (vlChart) vlChart.destroy();
  if (!points.length) return;

  vlChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: points.map(p => p.dte.toFixed(0) + 'd'),
      datasets: [{
        label: 'ATM IV',
        data: points.map(p => p.atm_iv * 100),
        borderColor: '#4cc9ff',
        backgroundColor: 'rgba(76,201,255,0.12)',
        borderWidth: 2,
        fill: true,
        tension: 0.25,
        pointRadius: 4,
        // Colour each point by confidence so a shaky reading is visibly
        // shaky rather than just another dot on a smooth curve.
        pointBackgroundColor: points.map(p =>
          p.quality === 'good' ? '#29d896'
            : p.quality === 'fair' ? '#ffb547' : '#ff526f'),
        pointBorderColor: 'transparent',
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => points[items[0].dataIndex].expiry,
            label: item => {
              const p = points[item.dataIndex];
              return [`ATM IV ${(p.atm_iv * 100).toFixed(2)}%`,
                      `${p.dte.toFixed(1)} days · ${p.n_samples} strikes`,
                      `confidence: ${(CONF[p.quality] || CONF.none).label}`];
            },
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { font: { size: 10 } } },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { callback: v => v + '%', font: { size: 10 } },
        },
      },
    },
  });
}

function vlTable(points) {
  const rows = points.map(p => {
    const c = CONF[p.quality] || CONF.none;
    const dim = (p.quality === 'poor' || p.quality === 'none') ? ' class="row-dim"' : '';
    return `<tr${dim}>
      <td class="mono">${p.expiry}</td>
      <td class="mono">${p.dte.toFixed(1)}</td>
      <td class="mono">${(p.atm_iv * 100).toFixed(2)}%</td>
      <td class="mono">${p.n_samples}</td>
      <td><span class="conf-chip ${c.cls}">${c.label}</span></td>
    </tr>`;
  }).join('');
  $vl('vl-table').querySelector('tbody').innerHTML =
    rows || '<tr><td colspan="5" class="empty-sm">no expiries</td></tr>';
}

vlApply();
vlWire();
vlLoad();
