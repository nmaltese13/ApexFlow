/* =========================================================================
   DEALER EXPOSURE — DEX / GEX / VEX / charm / vanna by strike

   Reads /api/dealer_greeks/{sym}. The convention selector is the point of
   the page: every number here is conditional on an assumption about which
   side of the open interest the market maker holds, and flipping it flips
   every sign. Making that switchable in the UI is the honest way to present
   a number that most dashboards quote as if it were observed.
   ========================================================================= */

const DG_KEY = 'dg_state_v1';
const DG_COLORS = {
  pos: 'rgba(41, 216, 150, 0.85)',
  posEdge: 'rgba(41, 216, 150, 1)',
  neg: 'rgba(255, 82, 111, 0.85)',
  negEdge: 'rgba(255, 82, 111, 1)',
  spot: '#4cc9ff',
  flip: '#ffb547',
};

let dgState = {
  sym: 'SPY', dte: 30, convention: 'naive', basis: 'openInterest', metric: 'gex',
};
try {
  dgState = { ...dgState, ...(JSON.parse(localStorage.getItem(DG_KEY) || 'null') || {}) };
} catch { /* private mode — defaults are fine */ }

function dgPersist() {
  try { localStorage.setItem(DG_KEY, JSON.stringify(dgState)); } catch {}
}

let dgChart = null;
let dgData = null;

const $dg = id => document.getElementById(id);

/* ---------------- formatters ---------------- */
function fmtBig(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  const s = v < 0 ? '−' : '+';
  if (a >= 1e12) return s + '$' + (a / 1e12).toFixed(2) + 'T';
  if (a >= 1e9) return s + '$' + (a / 1e9).toFixed(2) + 'B';
  if (a >= 1e6) return s + '$' + (a / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return s + '$' + (a / 1e3).toFixed(0) + 'K';
  return s + '$' + a.toFixed(0);
}
const fmtPx = v => (v == null || !isFinite(v)) ? '—' : '$' + v.toFixed(2);

const METRIC_LABEL = {
  gex: 'GEX  ($ delta per +1% spot)',
  dex: 'DEX  ($ of stock)',
  vex: 'VEX  ($ P&L per +1 vol point)',
  charm_exp: 'Charm  ($ delta per day)',
  vanna_exp: 'Vanna  ($ delta per +1 vol point)',
};

/* ---------------- toolbar ---------------- */
function dgApplyState() {
  $dg('dg-sym').value = dgState.sym;
  $dg('dg-dte').value = String(dgState.dte);
  $dg('dg-convention').value = dgState.convention;
  $dg('dg-basis').value = dgState.basis;
  document.querySelectorAll('#dg-presets button').forEach(b =>
    b.classList.toggle('active', b.dataset.sym === dgState.sym));
  document.querySelectorAll('#dg-metric-tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.metric === dgState.metric));
}

function dgWire() {
  const sym = $dg('dg-sym');
  sym.addEventListener('change', () => {
    const v = sym.value.trim().toUpperCase();
    if (v) { dgState.sym = v; dgPersist(); dgApplyState(); dgLoad(); }
  });
  sym.addEventListener('keydown', e => { if (e.key === 'Enter') sym.blur(); });

  document.querySelectorAll('#dg-presets button').forEach(b =>
    b.addEventListener('click', () => {
      dgState.sym = b.dataset.sym; dgPersist(); dgApplyState(); dgLoad();
    }));

  ['dg-dte', 'dg-convention', 'dg-basis'].forEach(id =>
    $dg(id).addEventListener('change', e => {
      const key = id === 'dg-dte' ? 'dte' : id.replace('dg-', '');
      dgState[key] = id === 'dg-dte' ? parseInt(e.target.value, 10) : e.target.value;
      dgPersist(); dgLoad();
    }));

  document.querySelectorAll('#dg-metric-tabs button').forEach(b =>
    b.addEventListener('click', () => {
      dgState.metric = b.dataset.metric;
      dgPersist(); dgApplyState();
      if (dgData) dgDrawChart(dgData);   // no refetch — same payload
    }));

  $dg('dg-refresh').addEventListener('click', dgLoad);
}

/* ---------------- load ---------------- */
async function dgLoad() {
  setLoading(true);
  const url = `/api/dealer_greeks/${encodeURIComponent(dgState.sym)}`
    + `?dte_max=${dgState.dte}&convention=${dgState.convention}&basis=${dgState.basis}`;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    dgData = await r.json();
  } catch (e) {
    setLoading(false);
    dgEmpty(`Could not load ${dgState.sym}: ${e.message}`);
    if (typeof toast === 'function') toast('Dealer exposure failed: ' + e.message, 'err');
    return;
  }
  setLoading(false);

  if (!dgData.strikes || !dgData.strikes.length) {
    dgEmpty(`No option chain for ${dgState.sym} within ${dgState.dte} days.`);
    return;
  }
  dgRenderTotals(dgData);
  dgDrawChart(dgData);
  dgRenderVex(dgData);
  dgRenderTable(dgData);
}

function setLoading(on) {
  document.querySelector('.dealer-page')?.classList.toggle('is-loading', !!on);
}

function dgEmpty(msg) {
  const t = $dg('dg-regime');
  t.className = 'dg-regime dg-regime-unknown';
  t.querySelector('.dg-regime-text').textContent = msg;
  ['dg-dex', 'dg-gex', 'dg-vex', 'dg-charm', 'dg-vanna', 'dg-flip']
    .forEach(id => { $dg(id).textContent = '—'; });
  if (dgChart) { dgChart.destroy(); dgChart = null; }
  $dg('dg-vex-panel').innerHTML = `<div class="empty-sm">${msg}</div>`;
  $dg('dg-table').querySelector('tbody').innerHTML =
    `<tr><td colspan="4" class="empty-sm">—</td></tr>`;
}

/* ---------------- totals + regime ---------------- */
function dgRenderTotals(d) {
  const s = d.summary || {};
  const set = (id, v) => { $dg(id).textContent = v; };

  set('dg-dex', fmtBig(s.total_dex));
  set('dg-gex', fmtBig(s.total_gex));
  set('dg-vex', fmtBig(s.total_vex));
  set('dg-charm', fmtBig(s.total_charm));
  set('dg-vanna', fmtBig(s.total_vanna));

  const flip = s.gamma_flip;
  set('dg-flip', flip == null ? 'none' : fmtPx(flip));
  $dg('dg-flip-sub').textContent = flip == null
    ? 'no crossing within ±25%'
    : (d.spot >= flip ? `spot is ${fmtPx(d.spot)} — above` : `spot is ${fmtPx(d.spot)} — below`);

  ['dg-dex', 'dg-gex', 'dg-vex', 'dg-charm', 'dg-vanna'].forEach(id => {
    const raw = { 'dg-dex': s.total_dex, 'dg-gex': s.total_gex, 'dg-vex': s.total_vex,
                  'dg-charm': s.total_charm, 'dg-vanna': s.total_vanna }[id];
    $dg(id).classList.toggle('pos', raw > 0);
    $dg(id).classList.toggle('neg', raw < 0);
  });

  const regime = s.regime || 'unknown';
  const el = $dg('dg-regime');
  el.className = 'dg-regime dg-regime-' + regime.replace('_', '-');
  el.querySelector('.dg-regime-text').innerHTML = regime === 'positive_gamma'
    ? `<strong>Positive gamma.</strong> Dealer hedging sells strength and buys weakness, which damps realised volatility around these levels.`
    : regime === 'negative_gamma'
      ? `<strong>Negative gamma.</strong> Dealer hedging runs with the move rather than against it, so realised volatility expands.`
      : 'Regime unknown.';
}

/* ---------------- chart ---------------- */
function dgDrawChart(d) {
  const canvas = $dg('dg-chart');
  if (!canvas || typeof Chart === 'undefined') return;
  const metric = dgState.metric;

  const rows = [...d.strikes].sort((a, b) => a.strike - b.strike);
  const labels = rows.map(r => r.strike);
  const values = rows.map(r => r[metric] ?? 0);

  if (dgChart) dgChart.destroy();
  dgChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: METRIC_LABEL[metric] || metric,
        data: values,
        backgroundColor: values.map(v => v >= 0 ? DG_COLORS.pos : DG_COLORS.neg),
        borderColor: values.map(v => v >= 0 ? DG_COLORS.posEdge : DG_COLORS.negEdge),
        borderWidth: 1,
        borderRadius: 2,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => 'Strike ' + items[0].label,
            label: item => {
              const r = rows[item.dataIndex];
              return [
                `${METRIC_LABEL[metric]}: ${fmtBig(r[metric])}`,
                `GEX ${fmtBig(r.gex)}   DEX ${fmtBig(r.dex)}`,
                `VEX ${fmtBig(r.vex)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { callback: v => fmtBig(v), font: { size: 10 } },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 }, autoSkip: true, maxTicksLimit: 26 },
        },
      },
    },
  });
}

/* ---------------- vega concentration ---------------- */
function dgRenderVex(d) {
  const v = d.vex || {};
  const near = v.pct_near_spot != null ? v.pct_near_spot : 0;
  const short = !!v.net_short_vega;
  const pct = Math.round(near * 100);

  $dg('dg-vex-panel').innerHTML = `
    <div class="dg-vex-headline ${short ? 'neg' : 'pos'}">
      Dealers are net <strong>${short ? 'short' : 'long'}</strong> vega
    </div>
    <div class="dg-vex-bar" role="img"
         aria-label="${pct}% of vega exposure sits within 5% of spot">
      <span style="width:${Math.min(Math.max(pct, 0), 100)}%"></span>
    </div>
    <div class="dg-vex-sub">
      <strong>${pct}%</strong> of vega exposure sits within 5% of spot
    </div>
    <div class="dg-vex-note">
      ${short
        ? 'A rise in implied vol costs dealers money and pushes them to buy vega back — typically options near spot, which also buys them gamma and can change the hedging regime mid-move.'
        : 'A rise in implied vol benefits dealers here, so a vol spike does not force them to buy options back.'}
    </div>`;
}

/* ---------------- table ---------------- */
function dgRenderTable(d) {
  const rows = [...d.strikes]
    .sort((a, b) => Math.abs(b.gex) - Math.abs(a.gex))
    .slice(0, 12);
  const body = rows.map(r => {
    const near = Math.abs(r.strike - d.spot) / d.spot < 0.005 ? ' class="near-spot"' : '';
    return `<tr${near}>
      <td class="mono">${r.strike.toFixed(2)}</td>
      <td class="mono ${r.gex >= 0 ? 'pos' : 'neg'}">${fmtBig(r.gex)}</td>
      <td class="mono ${r.dex >= 0 ? 'pos' : 'neg'}">${fmtBig(r.dex)}</td>
      <td class="mono ${r.vex >= 0 ? 'pos' : 'neg'}">${fmtBig(r.vex)}</td>
    </tr>`;
  }).join('');
  $dg('dg-table').querySelector('tbody').innerHTML =
    body || '<tr><td colspan="4" class="empty-sm">no strikes</td></tr>';
}

/* ---------------- init ---------------- */
dgApplyState();
dgWire();
dgLoad();


/* =========================================================================
   POSITION SIZING

   Pure arithmetic on the user's own inputs. Kept visually separate from the
   exposure surface above because it is a different kind of thing: the
   exposure numbers are a model of the market, these are a constraint on
   your own account, and conflating the two is how a description becomes a
   recommendation.
   ========================================================================= */
(function sizing() {
  const ids = ['rk-equity', 'rk-risk', 'rk-entry', 'rk-stop', 'rk-target', 'rk-unit'];
  const els = Object.fromEntries(ids.map(i => [i, document.getElementById(i)]));
  const out = document.getElementById('rk-result');
  if (!out || ids.some(i => !els[i])) return;

  const num = el => {
    const v = parseFloat(el.value);
    return Number.isFinite(v) ? v : null;
  };

  let timer = null;
  const debounce = fn => (...a) => { clearTimeout(timer); timer = setTimeout(() => fn(...a), 220); };

  async function compute() {
    const equity = num(els['rk-equity']);
    const risk = num(els['rk-risk']);
    const entry = num(els['rk-entry']);
    const stop = num(els['rk-stop']);
    const target = num(els['rk-target']);
    if (equity == null || risk == null || entry == null || stop == null) {
      out.innerHTML = '<div class="empty-sm">Enter equity, risk %, entry and stop.</div>';
      return;
    }
    const params = new URLSearchParams({
      equity, risk_pct: risk, entry, stop, unit: els['rk-unit'].value,
    });
    if (target != null) params.set('target', target);

    try {
      const r = await fetch('/api/size?' + params.toString());
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
      render(d);
    } catch (e) {
      out.innerHTML = `<div class="rk-refuse">${escapeHtml(e.message)}</div>`;
    }
  }

  function render(d) {
    if (!d.ok) {
      out.innerHTML = `<div class="rk-refuse"><strong>Not sized.</strong> ${escapeHtml(d.detail)}</div>`;
      return;
    }
    const warn = (d.warnings || []).map(w =>
      `<li>${escapeHtml(w)}</li>`).join('');
    const rw = d.reward
      ? `<div class="rk-row"><span>Reward / risk</span><strong>${d.reward.r}R</strong></div>
         <div class="rk-note">${escapeHtml(d.reward.detail)}</div>`
      : '';
    out.innerHTML = `
      <div class="rk-headline">
        <span class="rk-qty">${d.quantity.toLocaleString()}</span>
        <span class="rk-unit">${d.unit}</span>
      </div>
      <div class="rk-row"><span>Risk if stopped</span><strong>$${d.risk_amount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong></div>
      <div class="rk-row"><span>Position value</span><strong>$${d.position_value.toLocaleString(undefined, {maximumFractionDigits: 0})}</strong> <em>(${(d.position_pct_of_equity * 100).toFixed(1)}% of equity)</em></div>
      <div class="rk-row"><span>Stop distance</span><strong>$${d.stop_distance.toFixed(2)}</strong> <em>(${(d.stop_distance_pct * 100).toFixed(2)}%)</em></div>
      ${rw}
      ${warn ? `<ul class="rk-warnings">${warn}</ul>` : ''}`;
  }

  ids.forEach(i => {
    els[i].addEventListener('input', debounce(compute));
    els[i].addEventListener('change', debounce(compute));
  });
  compute();
})();
