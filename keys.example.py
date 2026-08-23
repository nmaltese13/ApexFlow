"""Template for keys.py — copy to keys.py and fill in any keys you have.

Priority for the main price/options provider (auto-selected at import time):
  1. Schwab          — free with Schwab brokerage; OAuth setup required
                        (run `python main.py schwab-auth` once after pasting keys)
  2. Polygon         — paid; Options Starter ($29/mo) gets you live IV+greeks
  3. MarketData.app  — paid; cheap fallback with greeks
  4. yfinance        — free, delayed, throttled (always available as last resort)

Supplementary clients (add what you have, scanners auto-use them when present):
  - Fintel        live short interest / borrow rate / days-to-cover
  - Finnhub       earnings calendar + news catalysts (free tier is plenty)
  - Unusual Whales options flow + dark pool prints (paid, gold standard)
"""

# ---- Main price/options provider (pick one; in priority order) ----
SCHWAB_APP_KEY    = ""
SCHWAB_APP_SECRET = ""
SCHWAB_REDIRECT_URI = "https://127.0.0.1:8443"

POLYGON_KEY       = ""

MARKETDATA_TOKEN  = ""

# ---- Supplementary data clients ----
FINNHUB_KEY        = ""
FINTEL_KEY         = ""
UNUSUAL_WHALES_KEY = ""
