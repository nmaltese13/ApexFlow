/* =========================================================================
   PROJECTION PANEL · gamma walls + Monte Carlo cone + TradingView widget
   The cone is the simulated one (data.mc.cone) when the API returns a
   simulation, falling back to the closed-form cone otherwise. Touch
   probabilities likewise prefer the simulated values, which come from
   real path maxima with a Brownian-bridge correction.
   ========================================================================= */

let projChart = null;
let projCandleSeries = null;
let projConeSeries = {};
let projWallLines = [];
let tvWidgetCreated = false;
let chartMode = 'smart';

const $ = id => document.getElementById(id);

/* Persist focus symbol */
function getFocusSym() {
  return ($('proj-sym').value || 'SPY').trim().toUpperCase() || 'SPY';
}

/* ---------- Build Lightweight Chart ---------- */
function buildProjChart() {
  const container = $('proj-chart-body');
  if (!container) return;
  if (projChart) { projChart.remove(); projChart = null; projWallLines = []; projConeSeries = {}; }

  projChart = LightweightCharts.createChart(container, {
    layout: {
      background: { type: 'solid', color: 'transparent' },
      textColor: '#8893a8',
      fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: 'rgba(28, 36, 53, 0.55)' },
      horzLines: { color: 'rgba(28, 36, 53, 0.55)' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Magnet,
      vertLine: { color: '#4cc9ff', width: 1, style: 3, labelBackgroundColor: '#0b78c0' },
      horzLine: { color: '#4cc9ff', width: 1, style: 3, labelBackgroundColor: '#0b78c0' },
    },
    rightPriceScale: { borderColor: 'rgba(28, 36, 53, 0.8)' },
    timeScale: { borderColor: 'rgba(28, 36, 53, 0.8)', timeVisible: true, secondsVisible: false, rightOffset: 25 },
  });

  projCandleSeries = projChart.addCandlestickSeries({
    upColor: '#29d896', downColor: '#ff526f',
    borderUpColor: '#29d896', borderDownColor: '#ff526f',
    wickUpColor: '#29d896', wickDownColor: '#ff526f',
  });

  // Cone series — five lines for percentiles
  projConeSeries.p5  = projChart.addLineSeries({ color: 'rgba(76,201,255,0.30)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  projConeSeries.p25 = projChart.addLineSeries({ color: 'rgba(76,201,255,0.45)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  projConeSeries.p50 = projChart.addLineSeries({ color: 'rgba(76,201,255,0.85)', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  projConeSeries.p75 = projChart.addLineSeries({ color: 'rgba(76,201,255,0.45)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
  projConeSeries.p95 = projChart.addLineSeries({ color: 'rgba(76,201,255,0.30)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });

  const ro = new ResizeObserver(entries => {
    for (const e of entries) projChart.applyOptions({ width: e.contentRect.width, height: e.contentRect.height });
  });
  ro.observe(container);
}

/* ---------- Render data into chart ---------- */
function fmtMoney(v) {
  const a = Math.abs(v || 0);
  const sign = v < 0 ? '-' : '';
  if (a >= 1e9) return sign + '$' + (a/1e9).toFixed(2) + 'B';
  if (a >= 1e6) return sign + '$' + (a/1e6).toFixed(2) + 'M';
  if (a >= 1e3) return sign + '$' + (a/1e3).toFixed(0) + 'K';
  return sign + '$' + a.toFixed(0);
}

function renderProjChart(data, keyLevels) {
  if (!projChart || !projCandleSeries) buildProjChart();
  if (!data || !data.candles || !data.candles.length) return;

  // Candles
  const ohlc = data.candles.map(d => ({ time: d.t, open: d.o, high: d.h, low: d.l, close: d.c }));
  projCandleSeries.setData(ohlc);

  // Clear old price lines
  projWallLines.forEach(pl => projCandleSeries.removePriceLine(pl));
  projWallLines = [];

  const addLine = (price, color, width, style, title) => {
    if (!Number.isFinite(price) || price <= 0) return;
    const pl = projCandleSeries.createPriceLine({
      price, color, lineWidth: width, lineStyle: style,
      axisLabelVisible: true, title,
    });
    projWallLines.push(pl);
  };

  // ----- 1. Real reactive S/R levels (priced action, NOT random pivots) -----
  if (keyLevels) {
    if (keyLevels.pdh != null) addLine(keyLevels.pdh, '#b07bff', 1, 1, 'PDH');
    if (keyLevels.pdl != null) addLine(keyLevels.pdl, '#b07bff', 1, 1, 'PDL');
    if (keyLevels.pdc != null) addLine(keyLevels.pdc, '#7a6dbd', 1, 2, 'PDC');
    if (keyLevels.poc != null) addLine(keyLevels.poc, '#7adcff', 2, 0, 'POC');
    if (keyLevels.vah != null) addLine(keyLevels.vah, '#4cc9ff', 1, 2, 'VAH');
    if (keyLevels.val != null) addLine(keyLevels.val, '#4cc9ff', 1, 2, 'VAL');
    (keyLevels.swing_highs || []).slice(0, 3).forEach((p, i) =>
      addLine(p, 'rgba(255, 82, 111, 0.55)', 1, 2, `SH${i + 1}`));
    (keyLevels.swing_lows || []).slice(0, 3).forEach((p, i) =>
      addLine(p, 'rgba(41, 216, 150, 0.55)', 1, 2, `SL${i + 1}`));
  }

  // ----- 2. Gamma walls (green +GEX / red -GEX / yellow King) -----
  const walls = mcMergedWalls(data);
  if (walls.length) {
    const ranked = [...walls].sort((a, b) => Math.abs(b.total_gex) - Math.abs(a.total_gex));
    const kingStrike = ranked[0]?.strike;
    const gateStrikes = new Set(ranked.slice(1, 5).map(r => r.strike));

    walls.forEach(w => {
      const isPos = w.total_gex > 0;
      const isKing = w.strike === kingStrike;
      const isGate = gateStrikes.has(w.strike);
      const color = isKing ? '#ffd540' : (isPos ? '#29d896' : '#ff526f');
      const lineWidth = isKing ? 3 : (isGate ? 2 : 1);
      addLine(w.strike, color, lineWidth, isKing ? 0 : 1,
              `${fmtMoney(w.total_gex)} · ${(w.touch_prob*100).toFixed(0)}%T`);
    });
  }

  // ----- 3. Spot reference -----
  if (data.spot) addLine(data.spot, '#4cc9ff', 2, 0, 'SPOT');

  // Projection cone — extend forward in time
  const lastT = ohlc[ohlc.length - 1].time;
  const horizonSec = (data.horizon_days || 1) * 86400;
  // Simulated cone when available; closed-form cone otherwise. Both expose
  // the same p5/p25/p50/p75/p95 keys and a 0..1 `frac`.
  const mc = (data.mc && !data.mc.error) ? data.mc : null;
  const cone = (mc && mc.cone && mc.cone.length) ? mc.cone : (data.cone || []);
  const conePts = (key) => cone.map(c => ({ time: lastT + Math.round(c.frac * horizonSec), value: c[key] }));
  projConeSeries.p5.setData(conePts('p5'));
  projConeSeries.p25.setData(conePts('p25'));
  projConeSeries.p50.setData(conePts('p50'));
  projConeSeries.p75.setData(conePts('p75'));
  projConeSeries.p95.setData(conePts('p95'));

  projChart.timeScale().fitContent();
  $('pl-spot').textContent = data.symbol + ' $' + (data.spot || 0).toFixed(2);
}


/* ---------- Merge simulated touch probabilities onto the wall rows ----------
   The API returns closed-form `touch_prob` on every wall and, when a
   simulation ran, a parallel `mc.walls` array keyed by strike carrying the
   simulated probability and its standard error. Prefer the simulated value:
   it comes from actual path maxima rather than a first-passage formula, so
   it stays correct under the jump and bootstrap models where no closed form
   exists. `touch_se` is kept so the UI can show how precise the number is. */
function mcMergedWalls(data) {
  const walls = data.gamma_walls || [];
  const mc = (data.mc && !data.mc.error) ? data.mc : null;
  if (!mc || !mc.walls || !mc.walls.length) return walls;
  const byStrike = new Map(mc.walls.map(w => [w.strike, w]));
  return walls.map(w => {
    const m = byStrike.get(w.strike);
    if (!m) return w;
    return { ...w,
             touch_prob: m.touch_prob,
             touch_se: m.touch_se,
             above_prob: (m.beyond_prob != null ? m.beyond_prob : w.above_prob),
             simulated: true };
  });
}

/* ---------- Sidebar ---------- */
function renderSidebar(data, keyLevels) {
  const setF = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  setF('ps-sym', data.symbol);
  setF('ps-spot', '$' + (data.spot || 0).toFixed(2));
  setF('ps-iv',  (data.atm_iv || 0).toFixed(1) + '%');
  // Show how much to trust the IV. A `poor` reading means the near-the-money
  // strikes disagreed badly (common on 0DTE), and every probability derived
  // from it inherits that uncertainty — better to say so than to render a
  // clean-looking 'rich'/'cheap' verdict on top of noise.
  const ivq = data.atm_iv_quality;
  const richness = data.atm_iv >= 35 ? 'rich' : data.atm_iv <= 15 ? 'cheap' : 'fair';
  setF('ps-iv-sub', (ivq === 'poor' || ivq === 'none' || ivq === 'hv_fallback')
       ? (ivq === 'hv_fallback' ? 'from realised vol' : 'low confidence')
       : richness);
  const em$ = data.expected_move || 0;
  const emPct = data.expected_move_pct || 0;
  setF('ps-em',  '$' + em$.toFixed(2));
  setF('ps-em-sub', '±' + emPct.toFixed(2) + '%');
  setF('ps-h', data.horizon_days < 1 ? (data.horizon_days * 24).toFixed(1) + 'h' : data.horizon_days.toFixed(0) + 'd');
  setF('ps-h-sub', `${(data.expiries_used || []).length} expiries`);

  setF('ps-flip', data.gamma_flip ? '$' + data.gamma_flip.toFixed(2) : '—');
  setF('ps-flip-sub', data.gamma_flip
       ? (data.spot >= data.gamma_flip ? 'long γ regime' : 'short γ regime')
       : ((data.gamma_flip_detail && data.gamma_flip_detail.bracketed === false)
          ? 'no flip within ±25%' : ''));

  const upT = (data.spot || 0) + em$;
  const dnT = (data.spot || 0) - em$;
  setF('ps-up', '$' + upT.toFixed(2));
  setF('ps-up-sub', '+' + emPct.toFixed(2) + '%');
  setF('ps-dn', '$' + dnT.toFixed(2));
  setF('ps-dn-sub', '−' + emPct.toFixed(2) + '%');

  // Probability table
  const tbl = $('proj-table');
  tbl.innerHTML = '<div class="pt-h">STRIKE</div><div class="pt-h">$GEX</div><div class="pt-h">TOUCH</div><div class="pt-h">ABOVE</div>';
  const walls = mcMergedWalls(data);
  if (!walls.length) {
    tbl.insertAdjacentHTML('beforeend', `<div style="grid-column: 1 / -1; text-align: center; padding: 12px; color: var(--txt-mute)">no gamma walls</div>`);
    $('proj-foot').textContent = '—';
    return;
  }
  // Sort by strike descending for display (chart-like top-to-bottom)
  const sorted = [...walls].sort((a, b) => b.strike - a.strike);
  const spotIdx = sorted.reduce((best, w, i) =>
    Math.abs(w.strike - data.spot) < Math.abs(sorted[best].strike - data.spot) ? i : best, 0);
  const sortedByMag = [...walls].map((w, i) => ({...w, i})).sort((a,b) => Math.abs(b.total_gex) - Math.abs(a.total_gex));
  const magnetStrikes = new Set(sortedByMag.slice(0, 3).map(w => w.strike));

  sorted.forEach((w, i) => {
    const cls = [];
    if (i === spotIdx) cls.push('spot');
    if (magnetStrikes.has(w.strike)) cls.push('magnet');
    const gexCls = w.total_gex >= 0 ? 'pos' : 'neg';
    tbl.insertAdjacentHTML('beforeend', `
      <div class="pt-strike ${cls.join(' ')}">${w.strike.toFixed(0)}</div>
      <div class="pt-gex ${gexCls}">${fmtMoney(w.total_gex)}</div>
      <div class="pt-prob"><span class="pbar" style="width:${(w.touch_prob*100).toFixed(0)}%"></span><span class="pval">${(w.touch_prob*100).toFixed(0)}%</span></div>
      <div class="pt-prob"><span class="pbar" style="width:${(w.above_prob*100).toFixed(0)}%"></span><span class="pval">${(w.above_prob*100).toFixed(0)}%</span></div>
    `);
  });

  const totalGex = walls.reduce((s, w) => s + w.total_gex, 0);
  const dom = sortedByMag[0];
  let footHtml = `Σ wall GEX <strong>${fmtMoney(totalGex)}</strong> · Dominant <strong>${dom.strike.toFixed(0)}</strong> @ <strong>${fmtMoney(dom.total_gex)}</strong>`;
  const mcInfo = (data.mc && !data.mc.error) ? data.mc : null;
  if (mcInfo) {
    footHtml += ` · <strong>${Number(mcInfo.n_paths).toLocaleString()}</strong> sim paths (${mcInfo.model})`;
  }
  if (keyLevels) {
    const bits = [];
    if (keyLevels.pdh != null) bits.push(`PDH <strong>${keyLevels.pdh.toFixed(2)}</strong>`);
    if (keyLevels.pdl != null) bits.push(`PDL <strong>${keyLevels.pdl.toFixed(2)}</strong>`);
    if (keyLevels.poc != null) bits.push(`POC <strong>${keyLevels.poc.toFixed(2)}</strong>`);
    if (bits.length) footHtml += ` · ${bits.join(' · ')}`;
  }
  $('proj-foot').innerHTML = footHtml;
}

/* ---------- TradingView widget ---------- */
function loadTradingView(sym) {
  const target = $('tv-widget');
  if (!target) return;
  target.innerHTML = '';
  // The tv.js widget creator
  if (typeof TradingView === 'undefined' || !TradingView.widget) {
    target.innerHTML = '<div class="hs-empty">TradingView script not loaded.</div>';
    return;
  }
  new TradingView.widget({
    container_id: 'tv-widget',
    symbol: sym,            // TV resolves exchange automatically for common tickers
    interval: '5',          // 5m default
    theme: 'dark',
    style: '1',             // candlesticks
    locale: 'en',
    toolbar_bg: '#10141e',
    enable_publishing: false,
    allow_symbol_change: true,
    hide_top_toolbar: false,
    hide_legend: false,
    save_image: false,
    autosize: true,
    studies: [
      'BB@tv-basicstudies',
      'MAExp@tv-basicstudies',
      'RSI@tv-basicstudies',
      'Volume@tv-basicstudies',
      'PivotPointsHighLow@tv-basicstudies',
      'VWAP@tv-basicstudies',
    ],
    backgroundColor: '#05070c',
    gridColor: 'rgba(28, 36, 53, 0.55)',
  });
  tvWidgetCreated = true;
}

/* ---------- Mode toggle ---------- */
function setMode(mode) {
  chartMode = mode;
  document.querySelectorAll('#chart-mode button').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  if (mode === 'tv') {
    $('proj-chart-body').classList.add('is-hidden');
    $('proj-tv-body').classList.remove('is-hidden');
    $('proj-chart-wrap').classList.add('tv');
    if (!tvWidgetCreated) loadTradingView(getFocusSym());
  } else {
    $('proj-tv-body').classList.add('is-hidden');
    $('proj-chart-body').classList.remove('is-hidden');
    $('proj-chart-wrap').classList.remove('tv');
    if (!projChart) buildProjChart();
    setTimeout(() => projChart && projChart.timeScale().fitContent(), 80);
  }
}

/* ---------- Load ---------- */
async function loadProjection() {
  const sym = getFocusSym();
  const horizon = parseFloat($('proj-horizon').value);
  const dte = parseInt($('proj-dte').value, 10);
  try {
    // Fetch projection + real key levels in parallel
    const [projR, lvlR] = await Promise.all([
      fetch(`/api/projection/${sym}?horizon_days=${horizon}&dte_max=${dte}`),
      fetch(`/api/keylevels/${sym}`),
    ]);
    if (!projR.ok) throw new Error('proj failed');
    const data = await projR.json();
    const keyLevels = lvlR.ok ? await lvlR.json() : null;
    renderProjChart(data, keyLevels);
    renderSidebar(data, keyLevels);
    if (chartMode === 'tv' && tvWidgetCreated) {
      loadTradingView(sym);
    }
  } catch (e) {
    toast('Projection load failed', 'err');
  }
}

/* ---------- Wire ---------- */
$('proj-sym').addEventListener('change', () => loadProjection());
$('proj-sym').addEventListener('keydown', e => { if (e.key === 'Enter') { e.target.blur(); } });
$('proj-horizon').addEventListener('change', () => loadProjection());
$('proj-dte').addEventListener('change', () => loadProjection());
$('proj-refresh').addEventListener('click', () => loadProjection());
document.querySelectorAll('#chart-mode button').forEach(b => {
  b.addEventListener('click', () => setMode(b.dataset.mode));
});

/* Init */
buildProjChart();
loadProjection();
