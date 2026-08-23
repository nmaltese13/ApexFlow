"""Charles Schwab API DataProvider (free with a Schwab brokerage account).

Setup
-----
1. Sign up at https://developer.schwab.com (link to your brokerage login).
2. Create an "Individual Developer" app. Set Callback URL to ``https://127.0.0.1``.
   Note your **App Key** and **App Secret**.
3. Wait for app approval (usually 1-7 days).
4. Once approved, paste the keys into ``keys.py``::

       SCHWAB_APP_KEY    = "..."
       SCHWAB_APP_SECRET = "..."
       SCHWAB_REDIRECT_URI = "https://127.0.0.1"   # leave default

5. Run the one-time OAuth dance::

       python main.py schwab-auth

   It opens a browser, you log in, click Allow, and the local handler
   captures the redirect code and exchanges it for a refresh token. Tokens
   are saved to ``data/schwab_tokens.json``.

Token lifecycle
---------------
* Access token: 30 min — auto-refreshed in this provider.
* Refresh token: 7 days — re-run ``schwab-auth`` weekly. The provider will
  raise ``SchwabAuthRequired`` with a clear message when this happens.

Endpoints used
--------------
* GET /marketdata/v1/{symbol}/pricehistory     — candles
* GET /marketdata/v1/quotes                    — real-time quote(s)
* GET /marketdata/v1/chains                    — full options chain
* GET /marketdata/v1/expirationchain           — list of expirations
* GET /marketdata/v1/instruments               — basic fundamentals (symbol_search)

The response shapes get normalized to match the DataProvider contract used
by the rest of ApexFlow (so scanners + Atlas don't care which provider is
behind the wheel).
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config
from .base import DataProvider

log = logging.getLogger(__name__)

API_BASE = "https://api.schwabapi.com"
TOKEN_URL = f"{API_BASE}/v1/oauth/token"
AUTH_URL = f"{API_BASE}/v1/oauth/authorize"
MARKET_BASE = f"{API_BASE}/marketdata/v1"


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------
TOKEN_PATH = config.DATA_DIR / "schwab_tokens.json"


class SchwabAuthRequired(RuntimeError):
    """Raised when user must re-authenticate (refresh token expired)."""


def _load_tokens() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text())
    except Exception:
        return None


def _save_tokens(tok: dict) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(tok, indent=2))


def _basic_auth_header(app_key: str, app_secret: str) -> str:
    raw = f"{app_key}:{app_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# OAuth helpers (used by the schwab-auth CLI command)
# ---------------------------------------------------------------------------
def build_auth_url(app_key: str, redirect_uri: str) -> str:
    """Build the URL the user opens in their browser to authorize the app."""
    qs = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": app_key,
        "redirect_uri": redirect_uri,
        "scope": "readonly",
    })
    return f"{AUTH_URL}?{qs}"


def exchange_code_for_tokens(app_key: str, app_secret: str,
                              code: str, redirect_uri: str) -> dict:
    """Trade the OAuth ``code`` for access + refresh tokens.

    Schwab returns: access_token, refresh_token, expires_in (seconds, ~1800),
    id_token (ignored), scope, token_type. We add ``access_expires_at`` and
    ``refresh_expires_at`` as unix timestamps for easy comparison.
    """
    headers = {
        "Authorization": _basic_auth_header(app_key, app_secret),
        "Content-Type":  "application/x-www-form-urlencoded",
    }
    body = {
        "grant_type":   "authorization_code",
        "code":          code,
        "redirect_uri":  redirect_uri,
    }
    r = requests.post(TOKEN_URL, headers=headers, data=body, timeout=15)
    r.raise_for_status()
    tok = r.json()
    now = int(time.time())
    tok["access_expires_at"]  = now + int(tok.get("expires_in", 1800)) - 30
    tok["refresh_expires_at"] = now + 7 * 24 * 3600 - 60   # docs say 7 days
    return tok


def refresh_access_token(app_key: str, app_secret: str, refresh_token: str) -> dict:
    """Use the long-lived refresh token to mint a new access token."""
    headers = {
        "Authorization": _basic_auth_header(app_key, app_secret),
        "Content-Type":  "application/x-www-form-urlencoded",
    }
    body = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    r = requests.post(TOKEN_URL, headers=headers, data=body, timeout=15)
    if r.status_code in (400, 401):
        raise SchwabAuthRequired(
            "Schwab refresh token rejected — refresh tokens expire after 7 days. "
            "Run `python main.py schwab-auth` again to re-authorize."
        )
    r.raise_for_status()
    tok = r.json()
    now = int(time.time())
    tok["access_expires_at"] = now + int(tok.get("expires_in", 1800)) - 30
    return tok


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class SchwabProvider(DataProvider):
    name = "schwab"

    def __init__(self, app_key: str, app_secret: str,
                 redirect_uri: str = "https://127.0.0.1"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.redirect_uri = redirect_uri
        self._tokens = _load_tokens() or {}
        self._session = requests.Session()
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._ttl = 30  # seconds — Schwab is real-time so we can cache briefly

    # ----- token plumbing ----------------------------------------------------
    def _ensure_access(self) -> str:
        tok = self._tokens
        now = int(time.time())
        if not tok or "refresh_token" not in tok:
            raise SchwabAuthRequired(
                "No Schwab tokens found. Run `python main.py schwab-auth` first."
            )
        # Refresh access token if expired or near-expiry
        if tok.get("access_expires_at", 0) <= now:
            new = refresh_access_token(self.app_key, self.app_secret, tok["refresh_token"])
            self._tokens.update(new)
            _save_tokens(self._tokens)
        # If refresh token has expired entirely, force re-auth
        if tok.get("refresh_expires_at", 0) <= now:
            raise SchwabAuthRequired(
                "Schwab refresh token has expired (7 days). "
                "Run `python main.py schwab-auth` to re-authorize."
            )
        return self._tokens["access_token"]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._ensure_access()}"}

    def _get(self, path: str, params: dict | None = None) -> Any:
        key = (path, tuple(sorted((params or {}).items())))
        now = time.time()
        if key in self._cache and now - self._cache[key][0] < self._ttl:
            return self._cache[key][1]
        url = f"{MARKET_BASE}{path}" if path.startswith("/") else path
        try:
            r = self._session.get(url, headers=self._headers(), params=params or {}, timeout=15)
            if r.status_code == 401:
                # Token might have expired between our check and the call — refresh once
                self._tokens["access_expires_at"] = 0
                r = self._session.get(url, headers=self._headers(), params=params or {}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except SchwabAuthRequired:
            raise
        except Exception as e:
            log.warning("Schwab GET %s failed: %s", path, e)
            return None
        self._cache[key] = (now, data)
        return data

    # ----- DataProvider interface -------------------------------------------
    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV DataFrame indexed by datetime, columns Open/High/Low/Close/Volume."""
        sym = symbol.upper()
        period_type, period_n, freq_type, freq_n = _period_to_schwab(period, interval)
        params = {
            "symbol": sym,
            "periodType":     period_type,
            "period":         period_n,
            "frequencyType":  freq_type,
            "frequency":      freq_n,
            "needExtendedHoursData": "false",
        }
        data = self._get(f"/pricehistory", params=params)
        if not data or not data.get("candles"):
            return pd.DataFrame()
        rows = []
        for c in data["candles"]:
            ts = datetime.fromtimestamp(c["datetime"] / 1000, tz=timezone.utc)
            rows.append({
                "Open": c.get("open"), "High": c.get("high"), "Low": c.get("low"),
                "Close": c.get("close"), "Volume": c.get("volume"),
            })
        df = pd.DataFrame(rows, index=[datetime.fromtimestamp(c["datetime"]/1000, tz=timezone.utc)
                                         for c in data["candles"]])
        df.index.name = "Date"
        return df

    def quote(self, symbol: str) -> dict:
        sym = symbol.upper()
        data = self._get("/quotes", params={"symbols": sym})
        if not data:
            return {}
        node = data.get(sym) or {}
        q = node.get("quote") or {}
        ref = node.get("reference") or {}
        return {
            "symbol": sym,
            "price":         q.get("lastPrice") or q.get("regularMarketLastPrice"),
            "previous_close": q.get("closePrice"),
            "open":          q.get("openPrice"),
            "high":          q.get("highPrice"),
            "low":           q.get("lowPrice"),
            "volume":        q.get("totalVolume"),
            "bid":           q.get("bidPrice"),
            "ask":           q.get("askPrice"),
            "name":          ref.get("description"),
        }

    def options_chain(self, symbol: str, expiry: str | None = None) -> dict:
        """Return the chain in the standard {calls, puts, spot, expiry} shape."""
        sym = symbol.upper()
        params = {
            "symbol":      sym,
            "contractType": "ALL",
            "includeUnderlyingQuote": "true",
            "strategy":    "SINGLE",
            # If expiry is provided, narrow to it
        }
        if expiry:
            params["fromDate"] = expiry
            params["toDate"]   = expiry
        data = self._get("/chains", params=params)
        if not data:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "spot": 0, "expiry": None}

        spot = float((data.get("underlyingPrice") or 0) or 0)
        calls = _flatten_chain_map(data.get("callExpDateMap") or {})
        puts  = _flatten_chain_map(data.get("putExpDateMap")  or {})

        # Pick the requested expiry, or the soonest one we got back.
        if expiry is None:
            all_exp = sorted({d for d in (calls.get("_expiry", []) if not calls.empty else [])})
            expiry = all_exp[0] if all_exp else None

        if expiry and not calls.empty:
            calls = calls[calls["_expiry"] == expiry].drop(columns="_expiry", errors="ignore")
        elif "_expiry" in calls.columns:
            calls = calls.drop(columns="_expiry")
        if expiry and not puts.empty:
            puts = puts[puts["_expiry"] == expiry].drop(columns="_expiry", errors="ignore")
        elif "_expiry" in puts.columns:
            puts = puts.drop(columns="_expiry")

        return {"calls": calls, "puts": puts, "spot": spot, "expiry": expiry}

    def expiries(self, symbol: str) -> list[str]:
        sym = symbol.upper()
        data = self._get("/expirationchain", params={"symbol": sym})
        if not data or not data.get("expirationList"):
            return []
        out = []
        for e in data["expirationList"]:
            d = e.get("expirationDate")
            if d:
                out.append(d)   # already YYYY-MM-DD
        return sorted(set(out))

    def fundamentals(self, symbol: str) -> dict:
        """Schwab's instruments endpoint gives some basics — float, market cap, sector
        require the FUNDAMENTAL projection."""
        sym = symbol.upper()
        data = self._get("/instruments", params={
            "symbol": sym,
            "projection": "fundamental",
        })
        if not data or not data.get("instruments"):
            return {}
        inst = data["instruments"][0] if isinstance(data["instruments"], list) else data["instruments"]
        f = inst.get("fundamental") or {}
        return {
            "symbol":        sym,
            "name":          inst.get("description"),
            "market_cap":    f.get("marketCap"),
            "shares_outstanding": f.get("sharesOutstanding"),
            "shares_float":  f.get("marketCapFloat"),
            "short_pct_float": (f.get("shortIntToFloat") or 0) / 100.0 if f.get("shortIntToFloat") else 0,
            "short_ratio":   f.get("shortIntDayToCover"),
            "beta":          f.get("beta"),
            "div_yield":     f.get("divYield"),
            "pe_ratio":      f.get("peRatio"),
            "sector":        inst.get("sector"),
        }

    def earnings_calendar(self, symbol: str) -> list[date]:
        """Schwab market data doesn't expose an earnings calendar endpoint —
        leave empty so the scanner falls back to Finnhub or yfinance."""
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _period_to_schwab(period: str, interval: str) -> tuple[str, int, str, int]:
    """Convert ApexFlow's period/interval strings to Schwab's pricehistory params.

    Schwab takes (periodType, period, frequencyType, frequency).
    Allowed combinations are documented at
    https://developer.schwab.com/products/trader-api--individual/details/specifications/Market%20Data%20Production
    """
    p = (period or "").lower().strip()
    i = (interval or "").lower().strip()

    # Daily frequency
    if i in ("1d", "1day", "daily", "day"):
        if p in ("1d", "5d"):           return ("day", 5, "minute", 30)  # fallback
        if p == "1mo":                  return ("month", 1, "daily", 1)
        if p in ("3mo", "6mo"):         return ("month", 6, "daily", 1)
        if p == "1y":                   return ("year", 1, "daily", 1)
        if p in ("2y", "5y", "ytd"):    return ("year", 2, "daily", 1)
        return ("year", 1, "daily", 1)

    # Intraday frequencies
    minutes = 1
    if i in ("1m", "1min"):  minutes = 1
    elif i in ("5m", "5min"): minutes = 5
    elif i in ("15m",):       minutes = 15
    elif i in ("30m",):       minutes = 30
    elif i in ("1h", "60m", "1hour"): minutes = 30   # Schwab 1h not supported, use 30m
    if p in ("1d",):  return ("day", 1, "minute", minutes)
    if p in ("5d",):  return ("day", 5, "minute", minutes)
    if p in ("10d",): return ("day", 10, "minute", minutes)
    return ("day", 5, "minute", minutes)


def _flatten_chain_map(exp_map: dict) -> pd.DataFrame:
    """Schwab returns expiration → strike → [contract] nested. Flatten to a flat DataFrame
    matching yfinance options-chain columns: strike, lastPrice, bid, ask, volume,
    openInterest, impliedVolatility (and an internal _expiry column for filtering)."""
    rows = []
    for exp_key, strikes in (exp_map or {}).items():
        # exp_key looks like "2026-05-09:7" — date plus DTE
        exp_iso = exp_key.split(":")[0]
        for strike_key, contracts in (strikes or {}).items():
            for c in contracts:
                rows.append({
                    "_expiry":  exp_iso,
                    "strike":   float(c.get("strikePrice") or strike_key),
                    "lastPrice": c.get("last") or c.get("mark") or 0,
                    "bid":       c.get("bid") or 0,
                    "ask":       c.get("ask") or 0,
                    "volume":    c.get("totalVolume") or 0,
                    "openInterest": c.get("openInterest") or 0,
                    "impliedVolatility": (c.get("volatility") or 0) / 100.0
                                          if c.get("volatility") else 0,
                    "delta":     c.get("delta") or 0,
                    "gamma":     c.get("gamma") or 0,
                    "theta":     c.get("theta") or 0,
                    "vega":      c.get("vega") or 0,
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Local OAuth helper — runs the one-time auth dance
# ---------------------------------------------------------------------------
def run_local_auth_flow(app_key: str, app_secret: str,
                         redirect_uri: str = "https://127.0.0.1",
                         port: int = 8443) -> dict:
    """Open the Schwab auth URL, capture the redirect ``code``, exchange for tokens.

    Schwab requires the redirect URI to be HTTPS, so we listen on a self-signed
    HTTPS server on 127.0.0.1. The user will see a browser warning the first
    time — that's expected because the cert is self-signed.

    Returns the saved token dict.
    """
    import http.server
    import ssl
    import tempfile
    import threading
    import webbrowser

    auth_url = build_auth_url(app_key, redirect_uri)
    print(f"\n  Opening browser to: {auth_url}\n")
    print("  If the browser doesn't open automatically, copy that URL and paste it.\n")
    print("  After you log in and click Allow, you'll be redirected to a")
    print(f"  {redirect_uri}/?code=... URL. Your browser will warn about the cert —")
    print("  click 'Advanced' → 'Proceed' to continue.\n")

    code_holder: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            if "code" in params:
                code_holder["code"] = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Authorized.</h2> You can close this tab and "
                                  b"return to your terminal.")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"missing code")

        def log_message(self, *a, **kw):
            pass  # quiet

    # Generate a throwaway self-signed cert
    cert_path, key_path = _make_self_signed_cert()

    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    def _serve():
        httpd.handle_request()  # one-shot

    th = threading.Thread(target=_serve, daemon=True)
    th.start()

    webbrowser.open(auth_url)
    th.join(timeout=300)  # 5 min to authorize

    if "code" not in code_holder:
        raise RuntimeError("Authorization timed out — re-run schwab-auth.")

    print("\n  Got authorization code. Exchanging for tokens…")
    tokens = exchange_code_for_tokens(app_key, app_secret,
                                       code_holder["code"], redirect_uri)
    _save_tokens(tokens)
    print(f"  Saved tokens to {TOKEN_PATH}")
    return tokens


def _make_self_signed_cert():
    """Generate a self-signed cert for the OAuth callback. Cleanup is left to OS tmp."""
    import tempfile
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from datetime import datetime as _dt, timedelta as _td

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1"),
    ])
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_dt.utcnow() - _td(days=1))
            .not_valid_after(_dt.utcnow() + _td(days=365))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("127.0.0.1")]),
                           critical=False)
            .sign(key, hashes.SHA256()))

    cert_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
    cert_file.close()
    key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    key_file.write(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    key_file.close()
    return cert_file.name, key_file.name
