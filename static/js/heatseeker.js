/* =========================================================================
   Heatseeker · Multi-DTE GEX matrix
   - User picks up to 5 individual DTEs; each becomes its own column.
   - Color scheme:
       +GEX  → shades of GREEN (intensity = magnitude / max)
       -GEX  → shades of RED   (intensity = magnitude / max)
       King  → YELLOW (extreme |GEX| level)
   - Each pane shows self-computed level markers (HVL, walls, zero-gamma,
     vol trigger, charm/vanna peaks) on the strike rail.
   ========================================================================= */

const DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM', 'TSLA'];
const MAX_DTES = 5;
const DEFAULT_DTES = [0, 1, 7];

let cols = 2;
let nStrikes = 40;
let autoTimer = null;
let activeDtes = [];   // numeric DTEs, length 1..5

function loadPaneSymbols() {
  try {
    const saved = JSON.parse(localStorage.getItem('hs_symbols') || 'null');
    if (Array.isArray(saved) && saved.length) return saved;
  } catch {}
  return DEFAULT_TICKERS.slice();
}
function savePaneSymbols(syms) {
  try { localStorage.setItem('hs_symbols', JSON.stringify(syms)); } catch {}
}
function loadActiveDtes() {
  try {
    const saved = JSON.parse(localStorage.getItem('hs_dtes') || 'null');
    if (Array.isArray(saved) && saved.length) {
      return saved.map(Number).filter(n => Number.isFinite(n) && n >= 0).slice(0, MAX_DTES);
    }
  } catch {}
  return DEFAULT_DTES.slice();
}
function saveActiveDtes(arr) {
  try { localStorage.setItem('hs_dtes', JSON.stringify(arr)); } catch {}
}
let paneSymbols = loadPaneSymbols();
activeDtes = loadActiveDtes();

/* ---------- Color spectrum ----------
   value > 0 → green (calm shade → vivid)
   value < 0 → red   (soft shade  → vivid)
   intensity = |value| / maxAbs in [0, 1].
   King node colour is overridden separately to yellow.
*/
function gexColor(value, maxAbs) {
  if (!maxAbs || !value) return 'rgba(20, 26, 38, 0.35)';
  const t = Math.max(0, Math.min(1, Math.abs(value) / maxAbs));
  // Smooth ease so weak values still register but don't dominate.
  const e = Math.pow(t, 0.7);
  if (value > 0) {
    // Light mint → emerald → vivid green
    const r = Math.round(28  + (41  - 28)  * e);
    const g = Math.round(120 + (216 - 120) * e);
    const b = Math.round(95  + (150 - 95)  * e);
    const a = 0.30 + e * 0.55;
    return `rgba(${r}, ${g}, ${b}, ${a.toFixed(2)})`;
  } else {
    // Dim rose → coral → vivid red
    const r = Math.round(200 + (255 - 200) * e);
    const g = Math.round(70  + (82  - 70)  * e);
    const b = Math.round(95  + (111 - 95)  * e);
    const a = 0.30 + e * 0.55;
    return `rgba(${r}, ${g}, ${b}, ${a.toFixed(2)})`;
  }
}

/* Number formatters */
function fmtGex(v) {
  const a = Math.abs(v);
  if (!v) return '0';
  const sign = v < 0 ? '-' : '+';
  if (a >= 1e9) return sign + '$' + (a/1e9).toFixed(2) + 'B';
  if (a >= 1e6) return sign + '$' + (a/1e6).toFixed(1) + 'M';
  if (a >= 1e3) return sign + '$' + (a/1e3).toFixed(0) + 'K';
  return sign + '$' + a.toFixed(0);
}

function fmtStrike(s) {
  return s % 1 === 0 ? s.toFixed(0) : s.toFixed(1);
}


function formatExpiryHeader(s) {
  if (!s) return '';
  // Accept "YYYY-MM-DD" or already-pretty strings
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  if (!m) return s;
  return m[2] + '/' + m[3] + '/' + m[1].slice(2);
}

/* ---------- Pane scaffolding ---------- */
function renderPane(idx) {
  const sym = paneSymbols[idx] || '';
  return `<div class="hs-pane" data-idx="${idx}">
    <div class="hs-pane-head">
      <input class="hs-sym-input" type="text" value="${sym}" maxlength="6" data-pane="${idx}" placeholder="—">
      <div class="hs-pane-meta">
        <span class="hs-spot" data-fld="spot">—</span>
        <span class="hs-chg" data-fld="chg">—</span>
      </div>
      <span class="hs-meta-sm" data-fld="meta">loading…</span>
    </div>
    <div class="hs-row-head" data-fld="header"></div>
    <div class="hs-pane-body" data-fld="body">
      <div class="hs-empty">loading…</div>
    </div>
    <div class="hs-pane-foot" data-fld="foot">—</div>
  </div>`;
}

function buildGrid() {
  const grid = document.getElementById('hs-grid');
  grid.className = 'hs-grid cols-' + cols;
  let html = '';
  for (let i = 0; i < cols; i++) html += renderPane(i);
  grid.innerHTML = html;

  grid.querySelectorAll('.hs-sym-input').forEach(inp => {
    inp.addEventListener('change', e => {
      const i = parseInt(e.target.dataset.pane, 10);
      paneSymbols[i] = e.target.value.trim().toUpperCase();
      savePaneSymbols(paneSymbols);
      e.target.value = paneSymbols[i];
      loadAll();
    });
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });
  });
}

/* ---------- Pane content ---------- */
const LEVEL_DEFS = [
  { key: 'hvl',         glyph: 'H',  cls: 'lv-hvl',    title: 'HVL — High Volume Liquidity (max OI×Vol)' },
  { key: 'call_wall',   glyph: 'CW', cls: 'lv-cwall',  title: 'Call Wall — max call OI (resistance)' },
  { key: 'put_wall',    glyph: 'PW', cls: 'lv-pwall',  title: 'Put Wall — max put OI (support)' },
  { key: 'zero_gamma',  glyph: '0γ', cls: 'lv-zgamma', title: 'Zero-Gamma — dealer flip (interpolated)' },
  { key: 'vol_trigger', glyph: 'VT', cls: 'lv-vtrig',  title: 'Vol Trigger — neg-gamma threshold below spot' },
  { key: 'charm_high',  glyph: 'C↑', cls: 'lv-charm',  title: 'Charm peak above spot' },
  { key: 'charm_low',   glyph: 'C↓', cls: 'lv-charm',  title: 'Charm peak below spot' },
  { key: 'vanna_high',  glyph: 'V↑', cls: 'lv-vanna',  title: 'Vanna peak above spot' },
  { key: 'vanna_low',   glyph: 'V↓', cls: 'lv-vanna',  title: 'Vanna peak below spot' },
];

function nearestStrike(grid, target) {
  if (!grid.length) return null;
  let best = grid[0], bestDist = Math.abs(grid[0] - target);
  for (let i = 1; i < grid.length; i++) {
    const d = Math.abs(grid[i] - target);
    if (d < bestDist) { best = grid[i]; bestDist = d; }
  }
  return best;
}

function fillPane(paneEl, data) {
  const setF = (sel, html) => { const el = paneEl.querySelector(`[data-fld="${sel}"]`); if (el) el.innerHTML = html; };

  if (!data || !data.symbol) {
    setF('spot', '—'); setF('chg', ''); setF('meta', '');
    setF('header', '');
    setF('body', '<div class="hs-empty">no data</div>');
    setF('foot', '—');
    return;
  }

  const { symbol, spot, change_pct, layers, strike_grid } = data;
  paneEl.querySelector('.hs-sym-input').value = symbol;
  setF('spot', '$' + (spot || 0).toFixed(2));
  const chgCls = (change_pct || 0) >= 0 ? 'up' : 'dn';
  const arrow = (change_pct || 0) >= 0 ? '▲' : '▼';
  paneEl.querySelector('[data-fld="chg"]').className = 'hs-chg ' + chgCls;
  setF('chg', `${arrow} ${(change_pct || 0).toFixed(2)}%`);

  const lyrs = layers || [];
  const grid = strike_grid || [];
  setF('meta', `${lyrs.length} DTE${lyrs.length === 1 ? '' : 's'} · ${grid.length} strikes`);

  if (!lyrs.length || !grid.length) {
    setF('header', '');
    setF('body', '<div class="hs-empty">no chain in selected DTEs</div>');
    setF('foot', '—');
    return;
  }

  // ==== NEW: bar-chart style cells (no LVL chips, horizontal magnitude bars) ====
  // Header columns: STRIKE | <one cell per layer showing expiry date>
  let header = `<div class="hs-rh-strike">STRIKE</div>`;
  lyrs.forEach(L => {
    // Show the actual expiry date if available (e.g. "05/01/26") instead of "0DTE"
    const expDate = (L.expiries_used && L.expiries_used[0]) || L.label;
    const niceDate = formatExpiryHeader(expDate);
    header += `<div class="hs-rh-cell">
                 <span>${niceDate}</span>
                 <span class="hs-rh-meta">|max|=${fmtGex(L.max_abs_gex || 0)}</span>
               </div>`;
  });
  setF('header', header);
  paneEl.querySelector('[data-fld="header"]').style.gridTemplateColumns =
    `64px repeat(${lyrs.length}, 1fr)`;

  const spotStrike = nearestStrike(grid, spot);

  // Index strikes by value for each layer
  const layerStrikeMap = lyrs.map(L => {
    const m = new Map();
    (L.strikes || []).forEach(s => m.set(round4(s.strike), s));
    return m;
  });

  let html = '';
  grid.forEach(strike => {
    const isSpot = strike === spotStrike;

    let cells = '';
    lyrs.forEach((L, lyrIdx) => {
      const row = layerStrikeMap[lyrIdx].get(round4(strike));
      const maxAbs = L.max_abs_gex || 1;
      if (!row || !row.total_gex) {
        cells += `<div class="hs-cell hs-cell-empty"><span class="hs-val muted">$0</span></div>`;
      } else {
        const isKing = row.node_type === 'king';
        const pct = Math.max(0, Math.min(100, (Math.abs(row.total_gex) / maxAbs) * 100));
        // Read theme-driven RGB triplets from CSS custom properties so the
        // Skylit yellow/purple toggle takes effect everywhere at once.
        const _rs = getComputedStyle(document.documentElement);
        const POS = (_rs.getPropertyValue('--gex-pos').trim() || '41, 216, 150');
        const NEG = (_rs.getPropertyValue('--gex-neg').trim() || '255, 82, 111');
        const KING = (_rs.getPropertyValue('--gex-king').trim() || '255, 213, 64');
        const barColor = isKing
          ? `rgba(${KING}, 0.85)`
          : `rgba(${row.total_gex > 0 ? POS : NEG}, 0.80)`;
        const cellBg = isKing
          ? `rgba(${KING}, 0.18)`
          : `rgba(${row.total_gex > 0 ? POS : NEG}, 0.14)`;
        const valColor = isKing
          ? `rgb(${KING})`
          : (row.total_gex > 0 ? `rgba(${POS}, 1)` : `rgba(${NEG}, 1)`);
        cells += `<div class="hs-cell hs-cell-bar${isKing ? ' king' : ''}" style="background:${cellBg}">
                    <div class="hs-bar" style="width:${pct.toFixed(1)}%; background:${barColor}"></div>
                    <span class="hs-val mono" style="color:${valColor}">${fmtGex(row.total_gex)}</span>
                  </div>`;
      }
    });

    html += `<div class="hs-strike-row${isSpot ? ' spot' : ''}"
                  style="grid-template-columns: 64px repeat(${lyrs.length}, 1fr)">
               <div class="hs-strike-label">${isSpot ? '<span class="hs-spot-mark"></span>' : ''}${fmtStrike(strike)}</div>
               ${cells}
             </div>`;
  });
  setF('body', html);

  // Foot summary: pick the first selected layer for the headline numbers
  const primary = lyrs[0];
  const totGex  = primary.summary?.total_gex || 0;
  const flip    = primary.summary?.gamma_flip;
  const lev0    = primary.levels || {};
  const stacks  = primary.stacks || [];
  const stacksHtml = stacks.slice(0, 2).map(st => {
    const cls = st.kind === 'rug_pull' ? 'rug' : 'sling';
    const label = st.kind === 'rug_pull' ? 'Reverse stack' : 'Spring stack';
    return `<span class="hs-stack-badge ${cls}">${label} ${st.lower_strike.toFixed(0)}↔${st.upper_strike.toFixed(0)}</span>`;
  }).join('');

  // gex_profile chips (walls, slide direction, pin, violence)
  const profile = primary.profile || {};
  let profileChips = '';
  if (profile.violence && profile.violence.violent) {
    profileChips += `<span class="hs-profile-tag violent" title="Negative gamma dominates near spot — expect amplified moves (the @t38p_flow framework)">⚡ violent</span>`;
  }
  if (profile.pin != null) {
    profileChips += `<span class="hs-profile-tag pin" title="Heavy positive gamma at-money — pinning candidate">📍 pin ${fmtStrike(profile.pin)}</span>`;
  }
  if (profile.slide && profile.slide.direction && profile.slide.direction !== 'flat' && profile.slide.r2 > 0.5) {
    const arrow = profile.slide.direction === 'up' ? '↗' : '↘';
    profileChips += `<span class="hs-profile-tag slide" title="Directional gamma slide — R²=${profile.slide.r2.toFixed(2)}">${arrow} slide</span>`;
  }
  if (profile.walls && profile.walls.length) {
    const wList = profile.walls.slice(0, 3).map(w => fmtStrike(w)).join(', ');
    profileChips += `<span class="hs-profile-tag wall" title="Concentrated single-strike gamma">🧱 walls ${wList}</span>`;
  }

  setF('foot', `
    <span title="${primary.label} — total GEX">Σ <strong>${fmtGex(totGex)}</strong></span>
    ${lev0.king != null ? `<span>King <strong>${fmtStrike(lev0.king)}</strong></span>` : ''}
    ${lev0.zero_gamma != null ? `<span>0γ <strong>${fmtStrike(lev0.zero_gamma)}</strong></span>` : (flip ? `<span>Flip <strong>$${flip.toFixed(2)}</strong></span>` : '')}
    ${lev0.vol_trigger != null ? `<span class="vt">VT <strong>${fmtStrike(lev0.vol_trigger)}</strong></span>` : ''}
    ${lev0.call_wall != null ? `<span class="cw">CW <strong>${fmtStrike(lev0.call_wall)}</strong></span>` : ''}
    ${lev0.put_wall  != null ? `<span class="pw">PW <strong>${fmtStrike(lev0.put_wall)}</strong></span>` : ''}
    ${stacksHtml}
    ${profileChips}
  `);
}

function round4(v) { return Math.round(v * 10000) / 10000; }

/* ---------- Loader ---------- */
async function loadAll() {
  const grid = document.getElementById('hs-grid');
  const symbols = paneSymbols.slice(0, cols).filter(Boolean);
  if (!symbols.length) return;

  grid.querySelectorAll('.hs-pane').forEach(el => {
    el.querySelector('[data-fld="meta"]').textContent = 'loading…';
  });

  try {
    const dtesParam = activeDtes.length ? activeDtes.join(',') : '0';
    const url = `/api/heatseeker?symbols=${symbols.join(',')}&dtes=${dtesParam}&n_strikes=${nStrikes}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error('api error');
    const data = await r.json();
    const panes = grid.querySelectorAll('.hs-pane');
    panes.forEach((paneEl, i) => {
      const sym = paneSymbols[i];
      const match = (data.symbols || []).find(d => d.symbol === sym);
      fillPane(paneEl, match || { symbol: sym });
    });
  } catch (e) {
    if (typeof toast === 'function') toast('Heatseeker load failed', 'err');
  }
}

/* ---------- DTE picker ---------- */
function syncDtePicker() {
  // Quick-pick chips: highlight if currently active
  document.querySelectorAll('#hs-dte-quick .hs-dte-quick').forEach(b => {
    const v = parseInt(b.dataset.dte, 10);
    b.classList.toggle('active', activeDtes.includes(v));
    b.disabled = !activeDtes.includes(v) && activeDtes.length >= MAX_DTES;
  });

  // Selected list (sorted ascending)
  const host = document.getElementById('hs-dte-active');
  if (!host) return;
  const sorted = [...activeDtes].sort((a, b) => a - b);
  if (!sorted.length) {
    host.innerHTML = '<span class="hs-dte-hint">pick at least one DTE above…</span>';
    return;
  }
  host.innerHTML = sorted.map(d => `
    <span class="hs-dte-pill" data-dte="${d}" title="Remove ${d}DTE column">
      <span class="hs-dte-pill-label">${d === 0 ? '0DTE' : d + 'DTE'}</span>
      <span class="hs-dte-pill-x">×</span>
    </span>
  `).join('') + `<span class="hs-dte-count">${sorted.length}/${MAX_DTES}</span>`;

  host.querySelectorAll('.hs-dte-pill').forEach(p => {
    p.addEventListener('click', () => {
      const d = parseInt(p.dataset.dte, 10);
      removeDte(d);
    });
  });
}

function addDte(d) {
  if (!Number.isFinite(d) || d < 0 || d > 365) return;
  if (activeDtes.includes(d)) return;
  if (activeDtes.length >= MAX_DTES) {
    if (typeof toast === 'function') toast(`Max ${MAX_DTES} DTEs at a time`, 'err');
    return;
  }
  activeDtes = [...activeDtes, d];
  saveActiveDtes(activeDtes);
  syncDtePicker();
  loadAll();
}

function removeDte(d) {
  if (!activeDtes.includes(d)) return;
  if (activeDtes.length === 1) {
    if (typeof toast === 'function') toast('Need at least one DTE', 'err');
    return;
  }
  activeDtes = activeDtes.filter(x => x !== d);
  saveActiveDtes(activeDtes);
  syncDtePicker();
  loadAll();
}

/* ---------- Wire toolbar ---------- */
document.getElementById('hs-strikes').addEventListener('change', e => {
  nStrikes = parseInt(e.target.value, 10); loadAll();
});
document.getElementById('hs-refresh').addEventListener('click', () => loadAll());
document.getElementById('hs-auto').addEventListener('change', e => {
  if (autoTimer) clearInterval(autoTimer);
  if (e.target.checked) {
    autoTimer = setInterval(() => loadAll(), 60_000);
    if (typeof toast === 'function') toast('Auto-refresh on (60s)', 'ok');
  } else if (typeof toast === 'function') {
    toast('Auto-refresh off');
  }
});
document.querySelectorAll('#hs-layout .hs-layout-btn').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#hs-layout .hs-layout-btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    cols = parseInt(b.dataset.cols, 10);
    while (paneSymbols.length < cols) paneSymbols.push(DEFAULT_TICKERS[paneSymbols.length] || '');
    savePaneSymbols(paneSymbols);
    buildGrid();
    loadAll();
  });
});

document.querySelectorAll('#hs-dte-quick .hs-dte-quick').forEach(b => {
  b.addEventListener('click', () => {
    const v = parseInt(b.dataset.dte, 10);
    if (activeDtes.includes(v)) removeDte(v);
    else addDte(v);
  });
});

const dteInput = document.getElementById('hs-dte-input');
const dteAdd = document.getElementById('hs-dte-add');
function commitDteInput() {
  const raw = parseInt(dteInput.value, 10);
  if (Number.isFinite(raw)) addDte(raw);
  dteInput.value = '';
}
dteAdd.addEventListener('click', commitDteInput);
dteInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); commitDteInput(); }
});

/* Init */
syncDtePicker();
buildGrid();
loadAll();
