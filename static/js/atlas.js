/* =========================================================================
   ATLAS · intraday GEX node history — "growing band" rendering
   - Each strike that has appeared in the time window gets a horizontal
     "orb band" at y = priceToY(strike).
   - Band LENGTH = how long that strike has been significantly active
     (from first significant snapshot → most recent significant snapshot).
   - Band THICKNESS = peak |GEX| at that strike (stronger strikes get
     fatter pills).
   - Band OPACITY = current strength (decays if the strike has cooled,
     intensifies if it's still building) — so bands "brighten" as the
     session progresses if the strike keeps getting heavier.
   - Color signed: green = positive dealer gamma (pinning/support),
     red = negative (volatility-amplifying / magnet).
   - King strike = bright yellow outline; walls = purple outline.
   - Time scrubber rebuilds the band aggregation only over snapshots
     up to the scrub position so you can replay the session.
   ========================================================================= */

const $ = id => document.getElementById(id);

let chart = null;
let candleSeries = null;
let overlayCanvas = null;
let overlayCtx = null;
let resizeObs = null;
let frameData = null;
let scrubIdx = -1;        // -1 = use full window
let drawMode = 'all';     // 'all' | 'latest' | 'growth'
let autoTimer = null;
let playTimer = null;

const TICK_TO_PIXEL_DEBOUNCE = 16;

/* ----------------------------- helpers ------------------------------------ */
function fmtMoney(v) {
  const a = Math.abs(v || 0);
  const sign = v < 0 ? '-' : v > 0 ? '+' : '';
  if (a >= 1e9) return sign + '$' + (a / 1e9).toFixed(2) + 'B';
  if (a >= 1e6) return sign + '$' + (a / 1e6).toFixed(2) + 'M';
  if (a >= 1e3) return sign + '$' + (a / 1e3).toFixed(0) + 'K';
  return sign + '$' + a.toFixed(0);
}
function fmtTs(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function fmtStrike(s) { return s % 1 === 0 ? s.toFixed(0) : s.toFixed(2); }
function debounce(fn, ms) {
  let t = null; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* ----------------------------- chart -------------------------------------- */
function buildChart() {
  const container = $('atl-chart-body');
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
      vertLine: { color: '#4cc9ff', width: 1, style: 3 },
      horzLine: { color: '#4cc9ff', width: 1, style: 3 },
    },
    rightPriceScale: { borderColor: 'rgba(28, 36, 53, 0.8)', scaleMargins: { top: 0.08, bottom: 0.08 } },
    timeScale: { borderColor: 'rgba(28, 36, 53, 0.8)', timeVisible: true, secondsVisible: false, rightOffset: 6 },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#29d896', downColor: '#ff526f',
    borderUpColor: '#29d896', borderDownColor: '#ff526f',
    wickUpColor: '#29d896', wickDownColor: '#ff526f',
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
  });

  overlayCanvas = $('atl-overlay');
  overlayCtx = overlayCanvas.getContext('2d');
  resizeOverlay();

  const redraw = debounce(drawOverlay, TICK_TO_PIXEL_DEBOUNCE);
  chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);
  chart.priceScale('right').subscribePriceScaleChanged?.(redraw);
  chart.subscribeCrosshairMove(() => {});

  if (resizeObs) resizeObs.disconnect();
  resizeObs = new ResizeObserver(() => {
    if (!chart) return;
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    resizeOverlay();
    drawOverlay();
  });
  resizeObs.observe(container);
}

function resizeOverlay() {
  if (!overlayCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = overlayCanvas.clientWidth || overlayCanvas.parentElement.clientWidth;
  const h = overlayCanvas.clientHeight || overlayCanvas.parentElement.clientHeight;
  overlayCanvas.width = Math.round(w * dpr);
  overlayCanvas.height = Math.round(h * dpr);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/* ----------------------------- coordinate helpers ------------------------- */
function priceToY(p) {
  if (!candleSeries) return null;
  const y = candleSeries.priceToCoordinate(p);
  return Number.isFinite(y) ? y : null;
}
function tsToX(ts) {
  if (!chart) return null;
  const x = chart.timeScale().timeToCoordinate(ts);
  return Number.isFinite(x) ? x : null;
}

/* ----------------------------- band aggregation --------------------------- */
/* From the per-snapshot frames, build one record per strike summarising:
   - first_ts: earliest snapshot where this strike had meaningful strength
   - last_ts:  most recent snapshot where it did
   - peak_strength: max node_strength seen
   - current_strength: most recent node_strength
   - signed_gex: most recent total_gex (drives color)
   - oi_total: most recent OI sum
   - is_king/is_wall_pos/is_wall_neg: from latest frame
   - growth_pct: % change peak_strength → current
   "Significant" = node_strength > 0.05 (filters noise tails).
*/
function buildBandsFromFrames(frames, scrubLimit) {
  if (!frames || !frames.length) return [];
  const SIG_THRESHOLD = 0.05;
  const usable = (scrubLimit > 0 && scrubLimit < frames.length)
    ? frames.slice(0, scrubLimit + 1) : frames;
  const map = new Map();           // key: `${bucket}:${strike}` -> record
  for (const f of usable) {
    for (const s of f.strikes || []) {
      if ((s.node_strength || 0) < SIG_THRESHOLD) continue;
      const key = `${s.expiry_bucket}:${s.strike}`;
      let rec = map.get(key);
      if (!rec) {
        rec = {
          strike: s.strike, bucket: s.expiry_bucket,
          first_ts: f.ts, last_ts: f.ts,
          peak_strength: s.node_strength, current_strength: s.node_strength,
          signed_gex: s.total_gex, current_gex_abs: Math.abs(s.total_gex || 0),
          peak_gex_abs: Math.abs(s.total_gex || 0),
          oi_total: (s.call_oi || 0) + (s.put_oi || 0),
          is_king: !!s.is_king, is_wall_pos: !!s.is_wall_pos, is_wall_neg: !!s.is_wall_neg,
          growth_pct: s.growth_pct || 0,
          n_frames: 1,
        };
        map.set(key, rec);
      } else {
        rec.last_ts = f.ts;
        rec.current_strength = s.node_strength;
        rec.signed_gex = s.total_gex;
        rec.current_gex_abs = Math.abs(s.total_gex || 0);
        rec.oi_total = (s.call_oi || 0) + (s.put_oi || 0);
        rec.is_king = !!s.is_king;
        rec.is_wall_pos = !!s.is_wall_pos;
        rec.is_wall_neg = !!s.is_wall_neg;
        rec.growth_pct = s.growth_pct || 0;
        rec.peak_strength = Math.max(rec.peak_strength, s.node_strength);
        rec.peak_gex_abs = Math.max(rec.peak_gex_abs, Math.abs(s.total_gex || 0));
        rec.n_frames++;
      }
    }
  }
  return Array.from(map.values());
}

/* Map signed gex to color (alpha applied on top) — reads theme RGB triplets
   from CSS custom properties so the Skylit yellow/purple theme works. */
function gexColor(value, alpha) {
  alpha = alpha == null ? 1 : alpha;
  const rs = getComputedStyle(document.documentElement);
  const pos = (rs.getPropertyValue('--gex-pos').trim() || '41, 216, 150');
  const neg = (rs.getPropertyValue('--gex-neg').trim() || '255, 82, 111');
  if (value > 0) return `rgba(${pos}, ${alpha.toFixed(2)})`;
  if (value < 0) return `rgba(${neg}, ${alpha.toFixed(2)})`;
  return `rgba(180, 180, 200, ${alpha.toFixed(2)})`;
}

/* Draw a rounded horizontal pill */
function pill(ctx, x1, x2, y, height, color, strokeColor, strokeWidth) {
  const w = Math.max(2, x2 - x1);
  const h = Math.max(2, height);
  const r = Math.min(h / 2, 8);
  ctx.beginPath();
  ctx.moveTo(x1 + r, y - h / 2);
  ctx.lineTo(x1 + w - r, y - h / 2);
  ctx.quadraticCurveTo(x1 + w, y - h / 2, x1 + w, y - h / 2 + r);
  ctx.lineTo(x1 + w, y + h / 2 - r);
  ctx.quadraticCurveTo(x1 + w, y + h / 2, x1 + w - r, y + h / 2);
  ctx.lineTo(x1 + r, y + h / 2);
  ctx.quadraticCurveTo(x1, y + h / 2, x1, y + h / 2 - r);
  ctx.lineTo(x1, y - h / 2 + r);
  ctx.quadraticCurveTo(x1, y - h / 2, x1 + r, y - h / 2);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  if (strokeColor) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth || 1;
    ctx.stroke();
  }
}

/* ----------------------------- main render -------------------------------- */
function drawOverlay() {
  if (!overlayCtx || !candleSeries || !frameData) return;
  const w = overlayCanvas.clientWidth;
  const h = overlayCanvas.clientHeight;
  overlayCtx.clearRect(0, 0, w, h);

  const frames = frameData.frames || [];
  if (!frames.length) return;

  const bands = buildBandsFromFrames(frames, scrubIdx);
  if (!bands.length) return;

  // Normalise peak_gex_abs across all bands → sets thickness scaling
  const maxPeak = Math.max(...bands.map(b => b.peak_gex_abs || 0)) || 1;

  // Sort: weakest first, strongest last so kings render on top
  bands.sort((a, b) => a.peak_strength - b.peak_strength);

  for (const b of bands) {
    const y = priceToY(b.strike);
    if (y == null) continue;

    let x1 = tsToX(b.first_ts);
    let x2 = tsToX(b.last_ts);
    // Off-screen handling: clamp to canvas if just slightly off
    if (x1 == null && x2 == null) continue;
    if (x1 == null) x1 = 0;
    if (x2 == null) x2 = w;

    // Single-frame bands need a minimum visible length
    if (x2 - x1 < 6) x2 = x1 + 6;

    // Thickness = √(peak_gex / max) × 14 px, min 4
    const thickness = Math.max(4, Math.min(22, 4 + 18 * Math.sqrt((b.peak_gex_abs || 0) / maxPeak)));

    // Opacity: current_strength weighted by recency. drawMode='growth' boosts
    // bands whose current is well above their first appearance.
    let opacity;
    if (drawMode === 'latest') {
      opacity = Math.max(0.15, Math.min(1.0, b.current_strength));
    } else if (drawMode === 'growth') {
      const g = Math.max(0, Math.min(3, (b.growth_pct || 0) / 100));
      opacity = Math.max(0.15, Math.min(1.0, 0.30 + 0.55 * g + 0.25 * b.current_strength));
    } else {
      // 'all' mode: opacity tracks current strength relative to peak —
      // bands that have decayed look ghost-like, ones still building stay vivid.
      const recencyRatio = b.peak_strength > 0 ? b.current_strength / b.peak_strength : 0;
      opacity = Math.max(0.18, Math.min(1.0, 0.30 + 0.65 * recencyRatio));
    }

    // Faint outer halo for stronger bands
    if (b.peak_strength > 0.4) {
      pill(overlayCtx, x1 - 2, x2 + 2, y, thickness + 4,
           gexColor(b.signed_gex, opacity * 0.18), null);
    }

    // Body
    let outline = null, outlineW = 0;
    if (b.is_king) {
      const _rs = getComputedStyle(document.documentElement);
      const king = (_rs.getPropertyValue('--gex-king').trim() || '255, 213, 64');
      outline = `rgba(${king}, 0.95)`; outlineW = 2;
    }
    else if (b.is_wall_pos || b.is_wall_neg) { outline = 'rgba(155, 109, 255, 0.85)'; outlineW = 1.5; }

    pill(overlayCtx, x1, x2, y, thickness, gexColor(b.signed_gex, opacity), outline, outlineW);

    // King label at right edge of band
    if (b.is_king) {
      overlayCtx.fillStyle = 'rgba(255, 213, 64, 0.95)';
      overlayCtx.font = 'bold 10px JetBrains Mono';
      const label = `${fmtStrike(b.strike)} ${fmtMoney(b.signed_gex)}`;
      overlayCtx.fillText(label, x2 + 6, y + 3);
    }
  }

  // Spot price line
  if (frameData.spot) {
    const y = priceToY(frameData.spot);
    if (y != null) {
      overlayCtx.strokeStyle = 'rgba(76, 201, 255, 0.45)';
      overlayCtx.setLineDash([4, 4]);
      overlayCtx.beginPath();
      overlayCtx.moveTo(0, y); overlayCtx.lineTo(w, y);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);
      overlayCtx.fillStyle = 'rgba(76, 201, 255, 0.95)';
      overlayCtx.font = 'bold 11px JetBrains Mono';
      overlayCtx.fillText('spot ' + fmtStrike(frameData.spot), 8, y - 6);
    }
  }
}

/* ----------------------------- sidebar / table ---------------------------- */
function renderSidebar(data) {
  $('atl-sym-label').textContent = data.symbol;
  $('atl-spot').textContent = data.spot ? '$' + fmtStrike(data.spot) : '—';
  $('atl-spot-big').textContent = data.spot ? '$' + fmtStrike(data.spot) : '—';

  const sess = data.session || {};
  $('atl-snap-count').textContent = sess.snapshots || 0;
  if (sess.first_ts && sess.last_ts) {
    $('atl-snap-window').textContent = `${fmtTs(sess.first_ts)} → ${fmtTs(sess.last_ts)}`;
  } else {
    $('atl-snap-window').textContent = 'No snapshots yet';
  }
  $('atl-strike-count').textContent = sess.strikes || 0;
  $('atl-bucket-list').textContent = (sess.buckets || []).join(' · ') || '—';

  const frames = data.frames || [];
  if (!frames.length) {
    $('atl-king').textContent = '—';
    $('atl-king-sub').textContent = 'waiting for first snapshot';
    $('atl-grow').textContent = '—';
    $('atl-grow-sub').textContent = '';
    $('atl-foot').textContent = 'No data yet — snapshot loop runs every 5 min during US RTH (and reduced cadence after-hours/weekends).';
    return;
  }

  const last = frames[frames.length - 1];
  const ranked = (last.strikes || []).slice().sort((a, b) => Math.abs(b.total_gex) - Math.abs(a.total_gex));

  const king = ranked.find(s => s.is_king) || ranked[0];
  if (king) {
    $('atl-king').textContent = fmtStrike(king.strike);
    $('atl-king-sub').textContent = `${fmtMoney(king.total_gex)} · ${(king.expiry_bucket || '').toUpperCase()}`;
  }
  const grower = (last.strikes || []).slice().sort((a, b) => (b.growth_pct || 0) - (a.growth_pct || 0))[0];
  if (grower && grower.growth_pct > 0) {
    $('atl-grow').textContent = fmtStrike(grower.strike);
    $('atl-grow-sub').textContent = `+${grower.growth_pct.toFixed(0)}% since open`;
  } else {
    $('atl-grow').textContent = '—';
    $('atl-grow-sub').textContent = 'no positive growth yet';
  }

  const tableEl = $('atl-table');
  while (tableEl.children.length > 4) tableEl.removeChild(tableEl.lastChild);
  ranked.slice(0, 12).forEach(s => {
    const cls = s.is_king ? 'king' : (s.is_wall_pos || s.is_wall_neg) ? 'wall' : '';
    const dir = s.total_gex >= 0 ? 'up' : 'dn';
    const tag = s.is_king ? '★' : s.is_wall_pos ? '+w' : s.is_wall_neg ? '−w' : '·';
    [
      `<div class="pt ${cls}">${fmtStrike(s.strike)} <span class="muted">${tag}</span></div>`,
      `<div class="pt mono ${dir}">${fmtMoney(s.total_gex)}</div>`,
      `<div class="pt mono">${(s.call_oi + s.put_oi).toLocaleString()}</div>`,
      `<div class="pt mono ${s.growth_pct >= 0 ? 'up' : 'dn'}">${(s.growth_pct >= 0 ? '+' : '')}${(s.growth_pct || 0).toFixed(0)}%</div>`,
    ].forEach(html => {
      const div = document.createElement('div');
      div.innerHTML = html;
      tableEl.appendChild(div.firstElementChild);
    });
  });

  $('atl-foot').textContent = `Updated ${fmtTs(data.generated_at)} · ${frames.length} frames in window`;
}

/* ----------------------------- scrubber ----------------------------------- */
function renderScrubber(frames) {
  const slider = $('atl-scrub'); const label = $('atl-scrub-label');
  if (!frames.length) { slider.max = 0; slider.value = 0; label.textContent = 'no frames'; return; }
  slider.max = frames.length;
  slider.value = frames.length;
  scrubIdx = -1;
  label.textContent = `all · ${frames.length} frames`;
}

$('atl-scrub').addEventListener('input', (e) => {
  if (!frameData) return;
  const i = parseInt(e.target.value, 10);
  const frames = frameData.frames || [];
  if (i >= frames.length) {
    scrubIdx = -1;
    $('atl-scrub-label').textContent = `all · ${frames.length} frames`;
  } else {
    scrubIdx = i;
    $('atl-scrub-label').textContent = `${fmtTs(frames[i].ts)} · frame ${i + 1}/${frames.length}`;
  }
  drawOverlay();
});

$('atl-play').addEventListener('click', () => {
  if (playTimer) {
    clearInterval(playTimer); playTimer = null;
    $('atl-play').textContent = '▶ Play'; return;
  }
  if (!frameData || !frameData.frames.length) return;
  $('atl-play').textContent = '⏸ Pause';
  scrubIdx = 0;
  $('atl-scrub').value = 0;
  drawOverlay();
  playTimer = setInterval(() => {
    if (!frameData) { clearInterval(playTimer); playTimer = null; return; }
    const frames = frameData.frames || [];
    scrubIdx++;
    if (scrubIdx >= frames.length) {
      scrubIdx = -1;
      $('atl-scrub').value = frames.length;
      $('atl-play').textContent = '▶ Play';
      clearInterval(playTimer); playTimer = null;
    } else {
      $('atl-scrub').value = scrubIdx;
      $('atl-scrub-label').textContent = `${fmtTs(frames[scrubIdx].ts)} · frame ${scrubIdx + 1}/${frames.length}`;
    }
    drawOverlay();
  }, 220);
});

/* ----------------------------- loader ------------------------------------- */
async function loadAtlas() {
  const sym = ($('atl-sym').value || 'TSLA').trim().toUpperCase();
  const bucket = $('atl-bucket').value;
  const hours = parseFloat($('atl-hours').value) || 6.5;
  const url = `/api/atlas/${sym}?hours=${hours}` + (bucket ? `&bucket=${bucket}` : '');
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error('atlas api error');
    frameData = await r.json();
  } catch (e) {
    if (typeof toast === 'function') toast('Atlas load failed: ' + e.message, 'err');
    return;
  }
  const candles = (frameData.intraday_candles || []).map(c => ({
    time: c.t, open: c.o, high: c.h, low: c.l, close: c.c,
  }));
  if (candles.length) {
    candleSeries.setData(candles);
    chart.timeScale().fitContent();
  }
  renderSidebar(frameData);
  renderScrubber(frameData.frames || []);
  requestAnimationFrame(() => requestAnimationFrame(drawOverlay));
}

/* ----------------------------- wiring ------------------------------------- */
$('atl-sym').addEventListener('change', loadAtlas);
$('atl-sym').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });
$('atl-bucket').addEventListener('change', loadAtlas);
$('atl-hours').addEventListener('change', loadAtlas);
$('atl-refresh').addEventListener('click', loadAtlas);
document.querySelectorAll('#atl-mode button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#atl-mode button').forEach(x => x.classList.remove('active'));
    b.classList.add('active'); drawMode = b.dataset.mode; drawOverlay();
  });
});
$('atl-auto').addEventListener('change', e => {
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  if (e.target.checked) autoTimer = setInterval(loadAtlas, 30_000);
});

buildChart();
loadAtlas();
if ($('atl-auto').checked) autoTimer = setInterval(loadAtlas, 30_000);
