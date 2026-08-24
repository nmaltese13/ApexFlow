"""ApexFlow configuration. API keys live in keys.py — edit that file."""
from __future__ import annotations
import os
from pathlib import Path

# ---- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SIGNAL_LOG_PATH = DATA_DIR / "signals.jsonl"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---- API keys (loaded from keys.py; env vars override) ----------------------
def _load_keys():
    defaults = {
        # Main provider candidates
        "SCHWAB_APP_KEY":    "",
        "SCHWAB_APP_SECRET": "",
        "SCHWAB_REDIRECT_URI": "https://127.0.0.1:8443",
        "POLYGON_KEY":       "",
        "MARKETDATA_TOKEN":  "",
        # Supplementary clients
        "UNUSUAL_WHALES_KEY": "",
        "FINTEL_KEY":         "",
        "FINNHUB_KEY":        "",
    }
    try:
        import keys as _keys
        for k in defaults:
            if hasattr(_keys, k):
                defaults[k] = getattr(_keys, k)
    except ImportError:
        pass
    for k in defaults:
        env_val = os.environ.get(f"APEXFLOW_{k}")
        if env_val:
            defaults[k] = env_val
    return defaults


_keys = _load_keys()
SCHWAB_APP_KEY      = _keys["SCHWAB_APP_KEY"]
SCHWAB_APP_SECRET   = _keys["SCHWAB_APP_SECRET"]
SCHWAB_REDIRECT_URI = _keys["SCHWAB_REDIRECT_URI"]
POLYGON_KEY         = _keys["POLYGON_KEY"]
MARKETDATA_TOKEN    = _keys["MARKETDATA_TOKEN"]
UNUSUAL_WHALES_KEY  = _keys["UNUSUAL_WHALES_KEY"]
FINTEL_KEY          = _keys["FINTEL_KEY"]
FINNHUB_KEY         = _keys["FINNHUB_KEY"]


# ---- Demo mode --------------------------------------------------------------
# Serve the frozen chain snapshot in data/demo/ instead of hitting a provider.
# Turns the whole terminal into something a stranger can run with no key and
# no network. See docs/demo_dataset.md.
DEMO_MODE = os.environ.get("APEXFLOW_DEMO", "").lower() in ("1", "true", "yes", "on")
DEMO_DIR = DATA_DIR / "demo"


def demo_dataset_present() -> bool:
    return (DEMO_DIR / "manifest.json").exists()


# ---- Free options feed ------------------------------------------------------
# Cboe publishes delayed option quotes as JSON with exchange-computed IV and
# Greeks, no key and no meaningful rate limit. It is strictly better than
# yfinance for options, so it is the default free source. Set
# APEXFLOW_DISABLE_CBOE=1 to fall back to yfinance (useful if the endpoint
# changes or you are offline behind a proxy that blocks it).
CBOE_ENABLED = os.environ.get("APEXFLOW_DISABLE_CBOE", "").lower() not in ("1", "true", "yes", "on")


# ---- Risk-free rate ---------------------------------------------------------
# Pull the Treasury par yield curve (free, no key) and match the rate to each
# contract's tenor instead of assuming a flat 4%. Set APEXFLOW_STATIC_RATE=1
# to keep the old constant — useful for reproducible offline runs.
LIVE_RATES = os.environ.get("APEXFLOW_STATIC_RATE", "").lower() not in ("1", "true", "yes", "on")


# ---- Universe defaults ------------------------------------------------------
DEFAULT_UNIVERSE = os.environ.get("APEXFLOW_UNIVERSE", "sp100")

# ---- Scanner thresholds (tune to taste) -------------------------------------
UNUSUAL_VOLUME_RATIO = 3.0          # volume / open_interest threshold
LARGE_BLOCK_PREMIUM = 50_000         # USD premium considered a "block"
BB_SQUEEZE_BANDWIDTH = 0.10          # (upper-lower)/mid < 10% = squeeze
RVOL_IGNITION_THRESHOLD = 3.0        # 3x average volume
MOMENTUM_ACCEL_PCT = 1.5             # % move in first 5 min
SHORT_FLOAT_MIN = 0.15
DAYS_TO_COVER_MIN = 3.0
EARNINGS_IV_CHEAP_RATIO = 0.8

# ---- Loop / alert cadences --------------------------------------------------
HUB_LOOP_SECONDS = 60
ALERT_COOLDOWN_SECONDS = 300

# ---- Cache TTL --------------------------------------------------------------
CACHE_TTL_INTRADAY = 60
CACHE_TTL_DAILY = 3600
CACHE_TTL_OPTIONS = 120


def schwab_tokens_present() -> bool:
    """Returns True if Schwab OAuth tokens are saved AND contain a refresh
    token (post `schwab-auth`). Empty/garbage files don't count."""
    p = DATA_DIR / "schwab_tokens.json"
    if not p.exists():
        return False
    try:
        import json as _json
        data = _json.loads(p.read_text())
        return bool(data.get("refresh_token"))
    except Exception:
        return False


def configured_providers() -> dict[str, bool]:
    """Status badges shown in the UI footer + `python main.py status` output."""
    schwab_ready = bool(SCHWAB_APP_KEY and SCHWAB_APP_SECRET) and schwab_tokens_present()
    return {
        "Demo snapshot (offline, no key)":     DEMO_MODE and demo_dataset_present(),
        "Schwab (price + options + greeks)":  schwab_ready,
        "Polygon (price + options)":          bool(POLYGON_KEY),
        "MarketData.app (chain fallback)":    bool(MARKETDATA_TOKEN),
        "Cboe delayed chains (free, no key)":  CBOE_ENABLED,
        "Unusual Whales (flow + dark pool)":  bool(UNUSUAL_WHALES_KEY),
        "Fintel (short interest / squeeze)":  bool(FINTEL_KEY),
        "Finnhub (news + earnings)":          bool(FINNHUB_KEY),
    }


def primary_provider_name() -> str:
    """Which provider get_provider() will return based on what's configured."""
    if DEMO_MODE and demo_dataset_present():
        return "demo"
    if SCHWAB_APP_KEY and SCHWAB_APP_SECRET and schwab_tokens_present():
        return "schwab"
    if POLYGON_KEY:
        return "polygon"
    if MARKETDATA_TOKEN:
        return "marketdata"
    if CBOE_ENABLED:
        return "cboe"
    return "yfinance"
