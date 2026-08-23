"""Pure pandas/numpy technical indicators. No talib dependency."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = stdev(close, n)
    upper = mid + k * sd
    lower = mid - k * sd
    bandwidth = (upper - lower) / mid
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_bw": bandwidth})


def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(close, n)
    a = atr(high, low, close, n)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + mult * a, "kc_lower": mid - mult * a})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = true_range(high, low, close)
    atr_n = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_n
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def rvol(volume: pd.Series, n: int = 30) -> pd.Series:
    """Relative volume vs n-day average volume."""
    avg = volume.rolling(n, min_periods=max(5, n // 2)).mean()
    return volume / avg.replace(0, np.nan)


def relative_strength(price: pd.Series, benchmark: pd.Series, n: int = 63) -> float:
    """IBD-style relative strength: stock return / benchmark return over n bars."""
    if len(price) < n + 1 or len(benchmark) < n + 1:
        return 0.0
    p = price.iloc[-1] / price.iloc[-n - 1] - 1
    b = benchmark.iloc[-1] / benchmark.iloc[-n - 1] - 1
    if b == 0:
        return 0.0
    return float(p / b)


def historical_volatility(close: pd.Series, n: int = 30) -> float:
    """Annualized close-to-close volatility."""
    if len(close) < n + 1:
        return 0.0
    rets = np.log(close / close.shift(1)).dropna()
    return float(rets.tail(n).std(ddof=0) * np.sqrt(252))


def squeeze_on(close: pd.Series, high: pd.Series, low: pd.Series,
               n: int = 20, bb_k: float = 2.0, kc_mult: float = 1.5) -> pd.Series:
    """TTM-style squeeze: BB inside KC."""
    bb = bollinger(close, n, bb_k)
    kc = keltner(high, low, close, n, kc_mult)
    return (bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])


def volume_dryup(volume: pd.Series, lookback: int = 20, ratio: float = 0.5) -> bool:
    """Recent 5-bar avg < `ratio` * lookback avg."""
    if len(volume) < lookback + 5:
        return False
    recent = volume.tail(5).mean()
    longer = volume.tail(lookback).mean()
    return bool(longer > 0 and recent / longer < ratio)


def gap_pct(open_today: float, close_yest: float) -> float:
    if not close_yest:
        return 0.0
    return (open_today - close_yest) / close_yest * 100
