/* =========================================================================
   ApexFlow base JS · clock · ticker tape · search · cmdk · toasts · helpers
   ========================================================================= */

/* ---------- Clock ---------- */
function tickClock() {
  const el = document.getElementById('clock');
  if (!el) return;
  el.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
}
setInterval(tickClock, 1000); tickClock();

/* ---------- Market status (US cash session, NY time) ---------- */
function updateMarketStatus() {
  const el = document.getElementById('market-status');
  const txt = document.getElementById('market-status-text');
  if (!el || !txt) return;

  // Get current time in America/New_York
  const fmt = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    weekday: 'short',
    hour: 'numeric', minute: 'numeric', hour12: false,
  });
  const parts = Object.fromEntries(fmt.formatToParts(new Date()).map(p => [p.type, p.value]));
  const dow = parts.weekday;
  const h = parseInt(parts.hour, 10);
  const m = parseInt(parts.minute, 10);
  const minutes = h * 60 + m;

  const isWeekend = (dow === 'Sat' || dow === 'Sun');
  const open = 9 * 60 + 30;
  const close = 16 * 60;
  const preOpen = 4 * 60;
  const postClose = 20 * 60;

  let state, label;
  if (isWeekend) { state = 'closed'; label = 'CLOSED · ' + dow; }
  else if (minutes >= open && minutes < close) {
    state = 'open';
    const left = close - minutes;
    const lh = Math.floor(left / 60), lm = left % 60;
    label = `OPEN · ${lh}h ${lm}m left`;
  }
  else if (minutes >= preOpen && minutes < open) { state = 'closed'; label = 'PRE-MARKET'; }
  else if (minutes >= close && minutes < postClose) { state = 'closed'; label = 'POST-MARKET'; }
  else { state = 'closed'; label = 'CLOSED'; }

  el.classList.remove('open', 'closed');
  el.classList.add(state);
  txt.textContent = label;
}
setInterval(updateMarketStatus, 30_000); updateMarketStatus();

/* ---------- Ticker tape ---------- */
async function loadTicker() {
  const el = document.getElementById('ticker-tape');
  if (!el) return;
  try {
    const r = await fetch('/api/ribbon');
    const data = await r.json();
    el.innerHTML = data.map(d => {
      const cls = d.change >= 0 ? 'tt-up' : 'tt-dn';
      const arrow = d.change >= 0 ? '▲' : '▼';
      const sign = d.change_pct >= 0 ? '+' : '';
      return `<a class="tt-item" href="/symbol/${d.symbol}">
        <span class="tt-sym">${d.symbol}</span>
        <span class="tt-px">${(d.price || 0).toFixed(2)}</span>
        <span class="${cls}">${arrow} ${sign}${(d.change_pct || 0).toFixed(2)}%</span>
      </a>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="ticker-loading">market data unavailable</div>';
  }
}
loadTicker();
setInterval(loadTicker, 30_000);

/* ---------- Symbol search ---------- */
const search = document.getElementById('symbol-search');
if (search) {
  search.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const v = search.value.trim().toUpperCase();
      if (v) location.href = '/symbol/' + v;
    }
  });
  // Open command palette when user starts typing in search
  search.addEventListener('focus', () => {
    if (window.innerWidth > 700) openCmdK(search.value);
  });
}

/* ---------- Toasts ---------- */
window.toast = (msg, kind = '') => {
  let t = document.querySelector('.toast');
  if (!t) {
    t = document.createElement('div');
    t.className = 'toast';
    document.body.appendChild(t);
  }
  t.className = 'toast ' + (kind || '');
  const icons = { ok: '✓', err: '⚠', '': '' };
  t.innerHTML = `<span class="ti">${icons[kind] || ''}</span><span>${msg}</span>`;
  // Reflow + show
  requestAnimationFrame(() => t.classList.add('show'));
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove('show'), 2400);
};

/* ---------- Helpers ---------- */
window.scoreClass = (s) => s >= 85 ? 'hot' : s >= 70 ? 'warm' : s >= 50 ? 'mild' : '';
window.fmtNum = (n, d=2) => Number(n || 0).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
window.fmtBig = (n) => {
  const a = Math.abs(n || 0);
  if (a >= 1e12) return (n/1e12).toFixed(2) + 'T';
  if (a >= 1e9)  return (n/1e9).toFixed(2)  + 'B';
  if (a >= 1e6)  return (n/1e6).toFixed(2)  + 'M';
  if (a >= 1e3)  return (n/1e3).toFixed(2)  + 'K';
  return Number(n || 0).toFixed(2);
};
window.fmtPct = (n, d=2) => (n >= 0 ? '+' : '') + Number(n || 0).toFixed(d) + '%';
window.fmtUSD = (n, d=2) => '$' + Number(n || 0).toFixed(d);

/* =========================================================================
   COMMAND PALETTE  (Cmd+K / Ctrl+K)
   ========================================================================= */
const COMMANDS = [
  { kind: 'page', label: 'Flow Dashboard',     href: '/' },
  { kind: 'page', label: 'GEX Heatmap',        href: '/heatmap' },
  { kind: 'page', label: 'Watchlist',          href: '/watchlist' },
  { kind: 'page', label: 'Signal Log',         href: '/log' },
  { kind: 'page', label: 'Backtest',           href: '/backtest' },
];
const POPULAR_SYMBOLS = ['SPY','QQQ','IWM','DIA','TSLA','NVDA','AAPL','MSFT','META','AMZN','GOOGL','AMD','NFLX','PLTR','GME','AMC','MSTR','COIN'];

const cmdkOverlay = document.getElementById('cmdk');
const cmdkInput = document.getElementById('cmdk-input');
const cmdkList = document.getElementById('cmdk-list');
let cmdkActive = 0;
let cmdkResults = [];

function openCmdK(prefill = '') {
  if (!cmdkOverlay) return;
  cmdkOverlay.classList.remove('hidden');
  cmdkInput.value = prefill || '';
  cmdkActive = 0;
  setTimeout(() => cmdkInput.focus(), 0);
  renderCmdK();
}
function closeCmdK() {
  if (!cmdkOverlay) return;
  cmdkOverlay.classList.add('hidden');
  if (search) search.blur();
}

function renderCmdK() {
  if (!cmdkList) return;
  const q = (cmdkInput.value || '').trim().toUpperCase();
  const symbolMatch = q && /^[A-Z\^.]{1,6}$/.test(q);
  const pageMatches = COMMANDS.filter(c => !q || c.label.toUpperCase().includes(q));
  const symMatches = q
    ? [q, ...POPULAR_SYMBOLS.filter(s => s.startsWith(q) && s !== q)].slice(0, 8).map(s => ({
        kind: 'ticker', label: s, href: '/symbol/' + s,
      }))
    : POPULAR_SYMBOLS.slice(0, 6).map(s => ({ kind: 'ticker', label: s, href: '/symbol/' + s }));

  cmdkResults = [];
  let html = '';
  if (symbolMatch || q) {
    html += '<div class="cmdk-section">Tickers</div>';
    symMatches.forEach(s => { cmdkResults.push(s); });
    html += symMatches.map((s, i) => itemHtml(s, cmdkResults.length - symMatches.length + i)).join('');
  }
  if (pageMatches.length) {
    html += '<div class="cmdk-section">Navigate</div>';
    const start = cmdkResults.length;
    pageMatches.forEach(c => cmdkResults.push(c));
    html += pageMatches.map((c, i) => itemHtml(c, start + i)).join('');
  }
  if (!q && !symbolMatch) {
    html = '<div class="cmdk-section">Popular Tickers</div>'
         + symMatches.map((s, i) => { cmdkResults.push(s); return itemHtml(s, cmdkResults.length - 1); }).join('')
         + '<div class="cmdk-section">Navigate</div>'
         + pageMatches.map((c, i) => { cmdkResults.push(c); return itemHtml(c, cmdkResults.length - 1); }).join('');
  }
  cmdkList.innerHTML = html;
  highlightCmdK();
}

function itemHtml(c, idx) {
  return `<div class="cmdk-item" data-idx="${idx}" data-href="${c.href}">
    <span>${c.label}</span><span class="cmdk-kind">${c.kind}</span>
  </div>`;
}

function highlightCmdK() {
  cmdkList.querySelectorAll('.cmdk-item').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.idx, 10) === cmdkActive);
  });
  const active = cmdkList.querySelector('.cmdk-item.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

if (cmdkOverlay) {
  cmdkInput.addEventListener('input', () => { cmdkActive = 0; renderCmdK(); });
  cmdkInput.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); cmdkActive = Math.min(cmdkActive + 1, cmdkResults.length - 1); highlightCmdK(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); cmdkActive = Math.max(cmdkActive - 1, 0); highlightCmdK(); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      const r = cmdkResults[cmdkActive];
      if (r) location.href = r.href;
    }
    else if (e.key === 'Escape') { e.preventDefault(); closeCmdK(); }
  });
  cmdkList.addEventListener('click', e => {
    const it = e.target.closest('.cmdk-item');
    if (it) location.href = it.dataset.href;
  });
  cmdkOverlay.addEventListener('click', e => {
    if (e.target === cmdkOverlay) closeCmdK();
  });
}

document.addEventListener('keydown', e => {
  // Cmd+K / Ctrl+K
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault(); openCmdK(); return;
  }
  // "/" to focus
  if (e.key === '/' && !['INPUT','TEXTAREA'].includes(document.activeElement?.tagName)) {
    e.preventDefault(); openCmdK();
  }
});

/* =========================================================================
   DATA FRESHNESS BANNER

   A delayed quote and a dead feed used to render identically. This polls
   /api/freshness and puts an unmissable bar at the top of the page when the
   data behind it is not current enough to act on.

   Deliberately not dismissible while the condition holds: the whole point
   is that stale data is invisible, and a banner you can click away is one
   you will click away.
   ========================================================================= */
(function freshnessWatch() {
  const banner = document.getElementById('stale-banner');
  const foot = document.getElementById('foot-freshness');
  if (!banner && !foot) return;

  const SYM_FROM_PAGE = () =>
    document.querySelector('.sym-page')?.dataset.symbol
    || document.getElementById('dg-sym')?.value
    || document.getElementById('gx-sym')?.value
    || 'SPY';

  function render(f) {
    if (foot) {
      const bits = [`${f.provider || f.source} · ${f.age_label} old`];
      if (f.market_state) bits.push(`market ${f.market_state}`);
      foot.textContent = bits.join(' · ');
      foot.className = 'muted fresh-' + f.level;
    }
    if (!banner) return;

    // Only shout for the two states that can mislead: an outright stale
    // feed during an open session, or data with no timestamp at all.
    const shout = f.level === 'stale' || f.level === 'unknown';
    banner.hidden = !shout;
    if (shout) {
      banner.className = 'stale-banner stale-' + f.level;
      banner.innerHTML =
        `<strong>${f.level === 'stale' ? 'Stale data' : 'Unknown data age'}</strong>`
        + ` — ${escapeHtml(f.detail)}`
        + ` <span class="stale-meta">Do not act on these numbers until this clears.</span>`;
    }
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  async function poll() {
    try {
      const r = await fetch('/api/freshness?sym=' + encodeURIComponent(SYM_FROM_PAGE()));
      if (!r.ok) throw new Error('HTTP ' + r.status);
      render(await r.json());
    } catch (e) {
      // The freshness check itself failing is exactly the situation the
      // banner exists for, so say so rather than staying silent.
      if (banner) {
        banner.hidden = false;
        banner.className = 'stale-banner stale-unknown';
        banner.innerHTML = '<strong>Data age unknown</strong> — the freshness '
          + 'check could not reach the server. Treat everything on screen as '
          + 'potentially out of date.';
      }
      if (foot) foot.textContent = 'data age unknown';
    }
  }

  poll();
  setInterval(poll, 30_000);
})();
