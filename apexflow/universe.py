"""Ticker universes.

`load_universe(name)` resolves any of:
    sp100        — curated 100 mega-caps (works offline)
    sp500        — full S&P 500 (Wikipedia, with offline fallback)
    nasdaq100    — full NDX 100
    optionable   — broad liquid-options universe (~330 names: SP500 + NDX +
                   sector ETFs + popular squeeze/meme/options names)
    mega         — alias for `optionable`
    squeeze      — high short-interest watch list
    etfs         — sector + index + leveraged ETFs (highly liquid options)
    a path       — text file (one symbol per line) or JSON list

Network calls are cached on disk so the universe survives offline sessions.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Curated lists (work offline, no network needed)
# ---------------------------------------------------------------------------
SP100 = [
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","BRK-B","AVGO",
    "LLY","JPM","V","XOM","UNH","MA","COST","HD","PG","JNJ",
    "ORCL","NFLX","BAC","ABBV","CVX","KO","MRK","CRM","PEP","ADBE",
    "WMT","TMO","AMD","LIN","MCD","CSCO","ACN","ABT","DIS","WFC",
    "INTU","DHR","VZ","TXN","NOW","PM","IBM","CAT","GE","AMGN",
    "GS","QCOM","NEE","RTX","COP","ISRG","PFE","UNP","SPGI","AMAT",
    "LOW","HON","AXP","SYK","BKNG","TJX","T","BLK","BA","MS",
    "ELV","DE","PLD","LMT","VRTX","ADI","SBUX","MDLZ","C","BMY",
    "SCHW","CI","GILD","MMC","ADP","CB","UPS","BX","REGN","SO",
    "AMT","ETN","PANW","MU","FI","INTC","DUK","ZTS","MO","TGT",
]

NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","ADBE","PEP","CSCO","TMUS","AMD","INTC","QCOM","INTU","TXN",
    "AMGN","ISRG","HON","AMAT","BKNG","SBUX","ADP","ADI","MDLZ","GILD",
    "VRTX","REGN","LRCX","PANW","MU","KLAC","SNPS","CDNS","MELI","ABNB",
    "PYPL","CRWD","MRVL","ORLY","FTNT","CHTR","CTAS","NXPI","PDD","WDAY",
    "PCAR","MAR","ROP","KDP","MNST","ASML","CSX","ADSK","AEP","DXCM",
    "FANG","TEAM","DDOG","CEG","BIIB","EXC","FAST","ODFL","KHC","XEL",
    "VRSK","EA","CTSH","GEHC","ROST","CPRT","BKR","IDXX","LULU","ON",
    "AZN","WBD","ANSS","CDW","DLTR","WBA","ZS","TTD","TTWO","SIRI",
    "MDB","MCHP","CSGP","SPLK","ATVI","DASH","ENPH","ILMN","ZM","JD",
]

# Sector / index / leveraged ETFs — extremely liquid options chains
ETFS = [
    "SPY","QQQ","IWM","DIA","VTI","VOO","VEA","VWO","EFA","EEM",
    "XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB","XLC","XLRE",
    "SMH","SOXX","SOXL","SOXS","XBI","IBB","ARKK","ARKG","ARKW","ARKF",
    "TLT","IEF","SHY","LQD","HYG","JNK","TIP",
    "GLD","SLV","USO","UNG","DBA","DBC","XOP","XME","KRE","KBE","ITB","XHB","XLF",
    "TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","UDOW","SDOW","TNA","TZA",
    "FXI","KWEB","ASHR","INDA","EWZ","EWJ","EWU","EWG","EWY","EWT",
    "VXX","UVXY","SVXY","BITO","IBIT","FBTC","ETHE","ETHA",
]

# Popular high-IV / squeeze / meme / momentum names with active options
SQUEEZE_WATCH = [
    "GME","AMC","BBBY","BYND","CVNA","UPST","RIVN","LCID","NKLA","MARA",
    "RIOT","PLTR","SOFI","HOOD","DKNG","FUBO","WKHS","SPCE","SNAP","DASH",
    "BIGC","OPEN","BLNK","CHWY","PTON","W","TUP","IEP","FFIE","MULN",
    "GTLB","NET","SNOW","DDOG","SMCI","ARM","MSTR","COIN","RBLX","U",
    "RKLB","LUNR","ASTS","ACHR","JOBY","ENVX","QS","CHPT","NIO","XPEV",
    "BABA","BIDU","BILI","TME","DIDI","FUTU","TIGR","JD","PDD","WDC",
]

# Popular high-volume options names not always in core indices
LIQUID_OPTIONS_EXTRAS = [
    "BAC","WFC","C","JPM","GS","MS","SCHW","V","MA","AXP","COF","PYPL","SQ","SHOP","MELI",
    "NVDA","AMD","TSM","INTC","MU","QCOM","ARM","SMCI","AVGO","ASML","LRCX","KLAC","AMAT","NXPI","ON",
    "TSLA","RIVN","LCID","NIO","XPEV","LI","F","GM","CCL","UAL","DAL","AAL","LUV","BA","UBER","LYFT","ABNB",
    "META","SNAP","PINS","RBLX","U","SPOT","NFLX","DIS","WBD","PARA","ROKU","FUBO","TTD","DASH",
    "AMZN","BABA","JD","PDD","SHOP","ETSY","W","CHWY","TGT","WMT","COST","HD","LOW","KSS","M","JWN","GPS","LULU","NKE","ULTA",
    "AAPL","GOOG","GOOGL","ORCL","CRM","NOW","ADBE","INTU","SAP","WDAY","TEAM","ZS","CRWD","NET","DDOG","SNOW","MDB","PANW","FTNT","OKTA","TWLO","ZM","DOCU",
    "XOM","CVX","COP","OXY","SLB","HAL","MPC","VLO","PSX","DVN","FANG","MRO","EOG","APA","HES",
    "GLD","SLV","NEM","FCX","GOLD","CLF","X","NUE","STLD","RIO","BHP","VALE",
    "JNJ","PFE","MRNA","BNTX","ABBV","BMY","GILD","REGN","VRTX","ISRG","SYK","MDT","TMO","DHR","CI","UNH","HUM","CVS","ELV","LLY","NVO","AZN",
    "MMM","HON","CAT","DE","BA","LMT","RTX","NOC","GD","GE","ETN","EMR","ROK",
    "T","VZ","TMUS","CMCSA","CHTR","DIS","NFLX","PARA","WBD",
    "PG","KO","PEP","MO","PM","BTI","UL","CL","KMB","CHD","STZ","DEO","BUD","TAP","SAM",
    "MCD","SBUX","CMG","DPZ","YUM","WEN","QSR","DRI","TXRH","PZZA",
    "BLK","BX","KKR","APO","CG","TROW","BEN","SPGI","MCO","ICE","CME","NDAQ","CBOE",
    "PLD","AMT","CCI","SBAC","EQIX","DLR","O","SPG","SLG","BXP","ARE","WELL","VTR","PSA","EXR",
    "DUK","SO","NEE","D","AEP","XEL","SRE","EXC","PEG","ED",
    "MSTR","COIN","HOOD","SOFI","UPST","AFRM","SQ","PYPL",
    "CVNA","CARG","KMX","AN",
]


def _curated_universe() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for src in (SP100, NASDAQ100, ETFS, SQUEEZE_WATCH, LIQUID_OPTIONS_EXTRAS):
        for sym in src:
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out


CURATED_OPTIONABLE = _curated_universe()


# ---------------------------------------------------------------------------
# Disk-cached SP500 fetch
# ---------------------------------------------------------------------------
_SP500_CACHE = config.CACHE_DIR / "sp500.json"


def _fetch_sp500() -> list[str]:
    """Pull the current S&P 500 list from Wikipedia, cache to disk."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if tickers and len(tickers) > 100:
            try:
                _SP500_CACHE.write_text(json.dumps(tickers))
            except OSError:
                pass
            return tickers
    except Exception:
        pass
    if _SP500_CACHE.exists():
        try:
            return json.loads(_SP500_CACHE.read_text())
        except Exception:
            pass
    return list(SP100)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def load_universe(name: str = "sp100") -> list[str]:
    name = (name or "sp100").lower()
    if name == "sp100":
        return list(SP100)
    if name == "nasdaq100" or name == "ndx":
        return list(NASDAQ100)
    if name == "etfs":
        return list(ETFS)
    if name == "squeeze":
        return list(SQUEEZE_WATCH)
    if name in ("optionable", "mega", "all"):
        # Curated + SP500 fetch (deduped, optionable mega-universe)
        sp500 = _fetch_sp500()
        seen = set(); out = []
        for sym in CURATED_OPTIONABLE + sp500:
            if sym and sym not in seen:
                seen.add(sym); out.append(sym)
        return out
    if name == "sp500":
        return _fetch_sp500()

    # Fallback: treat as path
    p = Path(name)
    if p.exists():
        try:
            data = p.read_text().strip()
            if data.startswith("["):
                return json.loads(data)
            return [line.strip().upper() for line in data.splitlines()
                    if line.strip() and not line.startswith("#")]
        except OSError:
            pass
    return list(SP100)


def universe_choices() -> list[tuple[str, str]]:
    """(value, label) pairs for UI dropdowns."""
    return [
        ("data/test_universe.txt", "test (3)"),
        ("sp100",      "S&P 100"),
        ("nasdaq100",  "NASDAQ 100"),
        ("etfs",       "Liquid ETFs"),
        ("squeeze",    "Squeeze watch"),
        ("sp500",      "S&P 500"),
        ("optionable", "Optionable mega-universe"),
    ]
