/* =========================================================================
   Dashboard · regime · scans · top cards · scanner blocks
   ========================================================================= */

const SCANNER_LABELS = {
  'options-flow': 'OPTIONS FLOW',
  'pre-breakout': 'PRE-BREAKOUT',
  'momentum':     'MOMENTUM IGNITION',
  'squeeze':      'SHORT SQUEEZE',
  'earnings':     'EARNINGS ALPHA',
};

let autoTimer = null;

function getEnabledScanners() {
  return Array.from(document.querySelectorAll('#scanner-toggles input:checked'))
    .map(i => i.dataset.scanner);
}

/* ---------- Market regime ---------- */
function heatColor(pct) {
  // -3% red → 0 neutral → +3% green
  const clamped = Math.max(-3, Math.min(3, pct));
  if (clamped >= 0) {
    const a = Math.min(0.55, clamped / 3 * 0.55 + 0.08);
    return `rgba(41, 216, 150, ${a.toFixed(2)})`;
  } else {
    const a = Math.min(0.55, -clamped / 3 * 0.55 + 0.08);
    return `rgba(255, 82, 111, ${a.toFixed(2)})`;
  }
}

async function loadRegime() {
  try {
    const r = await fetch('/api/regime').then(x => x.json());
    document.getElementById('r-vix').textContent = (r.vix || 0).toFixed(2);
    const vixCls = r.vix < 15 ? 'calm' : r.vix > 25 ? 'risk' : 'neutral';
    document.getElementById('r-vix').className = 'rv mono ' + vixCls;
    document.getElementById('r-vix-sub').textContent =
      r.vix < 15 ? 'low fear' : r.vix > 25 ? 'high fear' : 'moderate';

    document.getElementById('r-regime').textContent = r.regime.toUpperCase();
    document.getElementById('r-regime').className = 'rv ' + (r.regime === 'risk' ? 'risk' : r.regime === 'calm' ? 'calm' : 'neutral');
    document.getElementById('r-regime-sub').textContent =
      r.regime === 'risk' ? 'risk-off' : r.regime === 'calm' ? 'risk-on' : 'mixed';

    const trendCls = (r.spy_vs_ema50_pct || 0) >= 0 ? 'up' : 'dn';
    document.getElementById('r-spy').textContent = (r.spy_vs_ema50_pct >= 0 ? '+' : '') + (r.spy_vs_ema50_pct || 0).toFixed(2) + '%';
    document.getElementById('r-spy').className = 'rv mono ' + trendCls;
    document.getElementById('r-spy-sub').textContent = r.spy_trend;

    document.getElementById('r-breadth').textContent = `${r.breadth_up}/${r.breadth_total}`;
    document.getElementById('r-breadth-bar').style.width = (r.breadth_pct || 0).toFixed(0) + '%';
    document.getElementById('r-breadth-bar').style.background =
      r.breadth_pct >= 60 ? 'var(--green)' : r.breadth_pct <= 40 ? 'var(--red)' : 'var(--amber)';

    // Sector heat tiles
    const heat = document.getElementById('sector-heat');
    heat.innerHTML = (r.sectors || []).map(s => {
      const pct = s.change_pct || 0;
      const cls = pct >= 0 ? 'up' : 'dn';
      return `<a class="regime-cell" href="/symbol/${s.symbol}" style="background:${heatColor(pct)};text-decoration:none">
        <div class="rl">${s.label}</div>
        <div class="rv ${cls}" style="font-family:'JetBrains Mono',monospace;font-size:13px">${(pct >= 0 ? '+' : '') + pct.toFixed(2)}%</div>
        <div class="rs">${s.symbol}</div>
      </a>`;
    }).join('');

    document.getElementById('regime-meta').textContent = `breadth ${(r.breadth_pct || 0).toFixed(0)}%`;
  } catch (e) {
    document.getElementById('regime-meta').textContent = 'unavailable';
  }
}
loadRegime();
setInterval(loadRegime, 90_000);

/* ---------- Top cards / scanner blocks ---------- */
function renderTopCards(top) {
  const root = document.getElementById('top-cards');
  if (!top || !top.length) {
    root.innerHTML = `<div class="empty">
      <div class="empty-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><path d="M9 9l6 6 M15 9l-6 6"/></svg>
      </div>
      <div>No signals on this universe — try a wider one.</div>
    </div>`;
    return;
  }
  root.innerHTML = top.slice(0, 12).map(s => {
    const cls = scoreClass(s.score);
    const tags = (s.tags || []).slice(0, 4).join(' · ');
    return `<a class="sig-card ${cls}" href="/symbol/${s.symbol}">
      <div class="sc-head">
        <span class="sc-sym">${s.symbol}</span>
        <span class="score-pill score-${cls}">${s.score.toFixed(0)}</span>
      </div>
      <div class="sc-tags">${tags || '&nbsp;'}</div>
      <div class="sc-reason">${s.reason}</div>
    </a>`;
  }).join('');
}

function renderScannerBlock(scanner, signals) {
  const body = document.querySelector(`[data-body="${scanner}"]`);
  const ct = document.querySelector(`[data-ct="${scanner}"]`);
  if (!body) return;
  ct.textContent = signals.length;
  if (!signals.length) {
    body.innerHTML = '<div class="empty-sm">no signals</div>';
    return;
  }
  body.innerHTML = `<table class="dtable">
    <thead><tr><th>SYMBOL</th><th class="r">SCORE</th><th>REASON</th></tr></thead>
    <tbody>
      ${signals.slice(0, 12).map(s => `<tr>
        <td><a class="sym" href="/symbol/${s.symbol}">${s.symbol}</a></td>
        <td class="r mono"><span class="score-${scoreClass(s.score)}">${s.score.toFixed(0)}</span></td>
        <td class="reason">${s.reason}</td>
      </tr>`).join('')}
    </tbody>
  </table>`;
}

function showSkeletonCards() {
  const root = document.getElementById('top-cards');
  root.innerHTML = Array.from({length: 8}, () => '<div class="skel skel-card"></div>').join('');
}

async function runScan(force=false) {
  const universe = document.getElementById('universe-pick').value;
  const scanners = getEnabledScanners().join(',');
  if (!scanners) { toast('Enable at least one scanner', 'err'); return; }

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'SCANNING…';
  showSkeletonCards();

  const t0 = Date.now();
  try {
    const r = await fetch(`/api/scan?universe=${encodeURIComponent(universe)}&scanners=${scanners}` + (force ? '&_=' + Date.now() : ''));
    if (!r.ok) throw new Error('scan failed');
    const data = await r.json();
    document.getElementById('m-universe').textContent = data.universe_size;
    const totalSignals = Object.values(data.totals).reduce((a, b) => a + b, 0);
    document.getElementById('m-signals').textContent = totalSignals;
    document.getElementById('m-scanners').textContent = Object.keys(data.scanners).length;
    document.getElementById('m-top').textContent = data.top_signals[0]
      ? data.top_signals[0].score.toFixed(0)
      : '—';
    document.getElementById('m-last').textContent = new Date().toLocaleTimeString('en-US', { hour12: false });

    renderTopCards(data.top_signals);
    Object.keys(SCANNER_LABELS).forEach(s => {
      renderScannerBlock(s, data.scanners[s] || []);
    });

    document.getElementById('metric-row').classList.add('pulse');
    setTimeout(() => document.getElementById('metric-row').classList.remove('pulse'), 900);

    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    toast(`Scan complete · ${dt}s · ${totalSignals} signals`, 'ok');
  } catch (e) {
    toast('Scan failed', 'err');
    document.getElementById('top-cards').innerHTML = '<div class="empty">Scan failed — try again or check provider.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Scan';
  }
}

document.getElementById('run-btn').addEventListener('click', () => runScan(true));
document.getElementById('universe-pick').addEventListener('change', () => runScan(true));
document.querySelectorAll('#scanner-toggles input').forEach(i => {
  i.addEventListener('change', () => {
    const s = i.dataset.scanner;
    document.querySelector(`.scanner-block[data-scanner="${s}"]`).style.display = i.checked ? '' : 'none';
  });
});
document.getElementById('auto-refresh').addEventListener('change', e => {
  if (autoTimer) clearInterval(autoTimer);
  if (e.target.checked) {
    autoTimer = setInterval(() => runScan(true), 60_000);
    toast('Auto-refresh on (60s)', 'ok');
  } else {
    toast('Auto-refresh off');
  }
});

// First scan on load
runScan(false);
