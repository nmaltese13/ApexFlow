/* =========================================================================
   Symbol page · price chart (Lightweight Charts) · indicators · options chain
   ========================================================================= */

const SYMBOL = document.querySelector('.sym-page').dataset.symbol;

let chart, candleSeries, volumeSeries, bbUpperSeries, bbLowerSeries, ema20Series, ema50Series;

/* ---------- Quote panel ---------- */
async function loadQuote() {
  try {
    const q = await fetch('/api/quote/' + SYMBOL).then(r => r.json());
    const price = q.price || 0;
    const prev  = q.previous_close || 0;
    const chg   = price - prev;
    const chgPct = prev ? (chg / prev * 100) : 0;
    const cls = chg >= 0 ? 'up' : 'dn';
    const arrow = chg >= 0 ? '▲' : '▼';
    document.getElementById('s-last').innerHTML = '$' + price.toFixed(2);
    document.getElementById('s-last').className = 'm-value mono ' + cls;
    document.getElementById('s-chg').innerHTML = `<span class="${cls === 'up' ? 'tt-up' : 'tt-dn'}">${arrow} ${chg.toFixed(2)} (${chgPct.toFixed(2)}%)</span>`;
    document.getElementById('s-range').textContent = `$${(q.day_low||0).toFixed(2)} – $${(q.day_high||0).toFixed(2)}`;
    document.getElementById('s-vol').textContent = (q.volume || 0).toLocaleString();
    document.getElementById('s-mcap').textContent = q.market_cap ? '$' + fmtBig(q.market_cap) : '—';
  } catch {}
}

/* ---------- Indicators panel ---------- */
async function loadIndicators() {
  try {
    const i = await fetch('/api/indicators/' + SYMBOL).then(r => r.json());
    document.getElementById('i-rsi').textContent = (i.rsi || 0).toFixed(1);
    document.getElementById('i-rsi-sub').textContent =
      i.rsi >= 70 ? 'overbought' : i.rsi <= 30 ? 'oversold' : 'neutral';
    document.getElementById('i-bw').textContent  = (i.bb_bw || 0).toFixed(3);
    document.getElementById('i-bw-sub').textContent = i.bb_bw < 0.10 ? 'squeeze' : 'normal';
    document.getElementById('i-rv').textContent  = (i.rvol || 0).toFixed(2) + 'x';
    document.getElementById('i-atr').textContent = '$' + (i.atr14 || 0).toFixed(2);
    document.getElementById('i-hv').textContent  = (i.hv30 || 0).toFixed(1) + '%';
    document.getElementById('i-e20').textContent = '$' + (i.ema20 || 0).toFixed(2);
    document.getElementById('i-e50').textContent = '$' + (i.ema50 || 0).toFixed(2);
    if (typeof i.iv_rank === 'number') {
      document.getElementById('i-ivr').textContent = i.iv_rank.toFixed(0);
      document.getElementById('i-ivr-sub').textContent =
        i.iv_rank >= 60 ? 'rich' : i.iv_rank <= 30 ? 'cheap' : 'fair';
    }
  } catch {}
}

/* ---------- Indicator computations (client-side overlays) ---------- */
function bollinger(closes, n=20, k=2) {
  const out = [];
  for (let i = 0; i < closes.length; i++) {
    if (i < n - 1) { out.push({ mid: null, upper: null, lower: null }); continue; }
    let sum = 0;
    for (let j = i - n + 1; j <= i; j++) sum += closes[j];
    const mean = sum / n;
    let v = 0;
    for (let j = i - n + 1; j <= i; j++) v += (closes[j] - mean) ** 2;
    const sd = Math.sqrt(v / n);
    out.push({ mid: mean, upper: mean + k * sd, lower: mean - k * sd });
  }
  return out;
}
function ema(closes, n) {
  const k = 2 / (n + 1);
  const out = []; let prev;
  closes.forEach((c, i) => {
    if (i === 0) { prev = c; out.push(c); }
    else { prev = c * k + prev * (1 - k); out.push(prev); }
  });
  return out;
}

/* ---------- Chart ---------- */
function buildChart() {
  const container = document.getElementById('price-chart');
  if (!container) return;
  if (chart) { chart.remove(); chart = null; }

  chart = LightweightCharts.createChart(container, {
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
    rightPriceScale: {
      borderColor: 'rgba(28, 36, 53, 0.8)',
      scaleMargins: { top: 0.06, bottom: 0.22 },
    },
    timeScale: {
      borderColor: 'rgba(28, 36, 53, 0.8)',
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 6,
    },
    handleScroll: true,
    handleScale: true,
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#29d896', downColor: '#ff526f',
    borderUpColor: '#29d896', borderDownColor: '#ff526f',
    wickUpColor: '#29d896', wickDownColor: '#ff526f',
    priceLineColor: '#4cc9ff', priceLineWidth: 1, priceLineStyle: 2,
  });

  bbUpperSeries = chart.addLineSeries({
    color: 'rgba(176, 123, 255, 0.55)', lineWidth: 1, lineStyle: 2,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  });
  bbLowerSeries = chart.addLineSeries({
    color: 'rgba(176, 123, 255, 0.55)', lineWidth: 1, lineStyle: 2,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  });
  ema20Series = chart.addLineSeries({
    color: '#4cc9ff', lineWidth: 1.5,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  });
  ema50Series = chart.addLineSeries({
    color: '#ffb547', lineWidth: 1.5,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
  });

  volumeSeries = chart.addHistogramSeries({
    color: 'rgba(76, 201, 255, 0.5)',
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
  });
  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.85, bottom: 0 },
  });

  // Crosshair → legend
  chart.subscribeCrosshairMove(param => {
    if (!param || !param.time || !param.seriesData) {
      // restore last
      const last = window._lastBar;
      if (last) updateLegend(last);
      return;
    }
    const c = param.seriesData.get(candleSeries);
    if (c) updateLegend(c);
  });

  // Resize
  const ro = new ResizeObserver(entries => {
    for (const e of entries) {
      chart.applyOptions({ width: e.contentRect.width, height: e.contentRect.height });
    }
  });
  ro.observe(container);
}

function updateLegend(bar) {
  const o = bar.open, h = bar.high, l = bar.low, c = bar.close;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = '$' + (v || 0).toFixed(2); };
  set('lg-o', o); set('lg-h', h); set('lg-l', l); set('lg-c', c);
  const chg = c - o;
  const chgPct = o ? (chg / o * 100) : 0;
  const el = document.getElementById('lg-chg');
  if (el) {
    el.textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '  (' + (chgPct >= 0 ? '+' : '') + chgPct.toFixed(2) + '%)';
    el.className = 'lw-val mono ' + (chg >= 0 ? 'up' : 'dn');
  }
}

async function loadChart(period = '6mo', interval = '1d') {
  if (!chart) buildChart();
  const data = await fetch(`/api/history/${SYMBOL}?period=${period}&interval=${interval}`).then(r => r.json());
  if (!data || !data.length) { toast('No history available', 'err'); return; }

  const ohlc = data.map(d => ({ time: d.t, open: d.o, high: d.h, low: d.l, close: d.c }));
  const closes = data.map(d => d.c);
  const bb = bollinger(closes, 20, 2);
  const e20 = ema(closes, 20);
  const e50 = ema(closes, 50);

  candleSeries.setData(ohlc);
  bbUpperSeries.setData(data.map((d, i) => bb[i].upper != null ? { time: d.t, value: bb[i].upper } : null).filter(Boolean));
  bbLowerSeries.setData(data.map((d, i) => bb[i].lower != null ? { time: d.t, value: bb[i].lower } : null).filter(Boolean));
  ema20Series.setData(data.map((d, i) => ({ time: d.t, value: e20[i] })));
  ema50Series.setData(data.map((d, i) => ({ time: d.t, value: e50[i] })));

  volumeSeries.setData(data.map(d => ({
    time: d.t, value: d.v,
    color: d.c >= d.o ? 'rgba(41, 216, 150, 0.45)' : 'rgba(255, 82, 111, 0.45)',
  })));

  chart.timeScale().fitContent();

  // Last bar legend
  const last = ohlc[ohlc.length - 1];
  if (last) {
    window._lastBar = last;
    updateLegend(last);
  }

  // meta
  const meta = document.getElementById('s-legend-meta');
  if (meta) meta.textContent = `${data.length} bars · ${interval} · ${period}`;
}

/* ---------- Options chain ---------- */
let _chainData = null;
let _chainSide = 'both';

async function loadExpiries() {
  try {
    const exps = await fetch('/api/expiries/' + SYMBOL).then(r => r.json());
    const sel = document.getElementById('expiry-pick');
    if (!sel || !exps || !exps.length) return;
    sel.innerHTML = exps.slice(0, 12).map((e, i) => `<option value="${e}" ${i===0?'selected':''}>${e}</option>`).join('');
    sel.addEventListener('change', () => loadChain(sel.value));
  } catch {}
}

async function loadChain(expiry) {
  const wrap = document.getElementById('chain-wrap');
  wrap.innerHTML = '<div class="empty-sm">loading chain…</div>';
  try {
    const url = '/api/chain/' + SYMBOL + (expiry ? `?expiry=${expiry}` : '');
    const data = await fetch(url).then(r => r.json());
    _chainData = data;
    renderChain();
  } catch (e) {
    wrap.innerHTML = '<div class="empty-sm">failed to load chain</div>';
  }
}

function renderChain() {
  const wrap = document.getElementById('chain-wrap');
  if (!_chainData) { wrap.innerHTML = '<div class="empty-sm">no data</div>'; return; }
  const data = _chainData;
  const spot = data.spot || 0;
  const calls = data.calls || [];
  const puts  = data.puts || [];
  if (!calls.length && !puts.length) { wrap.innerHTML = '<div class="empty-sm">no chain available</div>'; return; }

  const allStrikes = [...new Set([...calls.map(c=>c.strike), ...puts.map(p=>p.strike)])].sort((a,b)=>a-b);
  const atmIdx = allStrikes.reduce((best, s, i) =>
    Math.abs(s - spot) < Math.abs(allStrikes[best] - spot) ? i : best, 0);
  const lo = Math.max(0, atmIdx - 12);
  const hi = Math.min(allStrikes.length, atmIdx + 13);
  const strikes = allStrikes.slice(lo, hi);

  const callMap = Object.fromEntries(calls.map(c => [c.strike, c]));
  const putMap  = Object.fromEntries(puts.map(p => [p.strike, p]));

  if (_chainSide === 'greeks') {
    let html = `<table class="dtable"><thead><tr>
      <th class="r">Δ C</th><th class="r">Γ C</th><th class="r">Θ C</th><th class="r">ν C</th>
      <th class="r">STRIKE</th>
      <th class="r">ν P</th><th class="r">Θ P</th><th class="r">Γ P</th><th class="r">Δ P</th>
    </tr></thead><tbody>`;
    strikes.forEach(s => {
      const c = callMap[s] || {}, p = putMap[s] || {};
      const isAtm = s === allStrikes[atmIdx];
      const cls = isAtm ? 'atm' : (s < spot ? 'itm' : '');
      const fmt = (v, d=3) => v == null ? '—' : Number(v).toFixed(d);
      html += `<tr class="${cls}">
        <td class="r">${fmt(c.delta)}</td>
        <td class="r">${fmt(c.gamma, 4)}</td>
        <td class="r">${fmt(c.theta)}</td>
        <td class="r">${fmt(c.vega)}</td>
        <td class="r"><strong>${s}</strong></td>
        <td class="r">${fmt(p.vega)}</td>
        <td class="r">${fmt(p.theta)}</td>
        <td class="r">${fmt(p.gamma, 4)}</td>
        <td class="r">${fmt(p.delta)}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
    return;
  }

  let html = `<table class="dtable"><thead><tr>`;
  if (_chainSide !== 'puts') html += `<th class="r">C VOL</th><th class="r">C OI</th><th class="r">C IV</th><th class="r">C BID</th><th class="r">C ASK</th>`;
  html += `<th class="r">STRIKE</th>`;
  if (_chainSide !== 'calls') html += `<th class="r">P BID</th><th class="r">P ASK</th><th class="r">P IV</th><th class="r">P OI</th><th class="r">P VOL</th>`;
  html += `</tr></thead><tbody>`;

  strikes.forEach(strike => {
    const c = callMap[strike] || {};
    const p = putMap[strike] || {};
    const isAtm = strike === allStrikes[atmIdx];
    const cls = isAtm ? 'atm' : (strike < spot ? 'itm' : '');
    const cVolUnusual = c.vol && c.oi && c.vol / c.oi >= 3;
    const pVolUnusual = p.vol && p.oi && p.vol / p.oi >= 3;
    html += `<tr class="${cls}">`;
    if (_chainSide !== 'puts') html += `
      <td class="r ${cVolUnusual?'unusual':''}">${(c.vol||0).toLocaleString()}</td>
      <td class="r">${(c.oi||0).toLocaleString()}</td>
      <td class="r">${((c.iv||0)*100).toFixed(1)}%</td>
      <td class="r">${(c.bid||0).toFixed(2)}</td>
      <td class="r">${(c.ask||0).toFixed(2)}</td>`;
    html += `<td class="r"><strong>${strike}</strong></td>`;
    if (_chainSide !== 'calls') html += `
      <td class="r">${(p.bid||0).toFixed(2)}</td>
      <td class="r">${(p.ask||0).toFixed(2)}</td>
      <td class="r">${((p.iv||0)*100).toFixed(1)}%</td>
      <td class="r">${(p.oi||0).toLocaleString()}</td>
      <td class="r ${pVolUnusual?'unusual':''}">${(p.vol||0).toLocaleString()}</td>`;
    html += `</tr>`;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

document.querySelectorAll('.chain-tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.chain-tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    _chainSide = t.dataset.side;
    renderChain();
  });
});

/* ---------- Wire up ---------- */
const period = document.getElementById('period-pick');
const interval = document.getElementById('interval-pick');
function reloadChart() { loadChart(period.value, interval.value); }
period.addEventListener('change', reloadChart);
interval.addEventListener('change', reloadChart);

// Initial load
buildChart();
loadQuote();
loadIndicators();
reloadChart();
loadExpiries().then(() => loadChain());
setInterval(loadQuote, 30_000);
