"""Short squeeze probability engine."""
from __future__ import annotations
import logging

from apexflow.analytics import indicators as ind
from apexflow.analytics.squeeze import SqueezeInputs, squeeze_score
from apexflow.models import Signal
from .base import BaseScanner

log = logging.getLogger(__name__)


class ShortSqueezeScanner(BaseScanner):
    name = "squeeze"
    label = "Short Squeeze Probability Engine"

    def scan(self, universe: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        for sym in universe:
            fund = self.provider.fundamentals(sym)
            if not fund:
                continue
            short_pct = fund.get("short_pct_float", 0) or 0
            dtc = fund.get("short_ratio", 0) or 0
            if short_pct < 0.10 and dtc < 2:
                continue

            try:
                df = self.provider.history(sym, period="3mo", interval="1d")
            except Exception:
                continue
            if df is None or len(df) < 30:
                continue
            close = df["Close"]; vol = df["Volume"]

            hv = ind.historical_volatility(close, 30)
            iv = 0.0
            try:
                chain = self.provider.options_chain(sym)
                calls = chain.get("calls")
                spot = chain.get("spot") or close.iloc[-1]
                if calls is not None and not calls.empty and spot:
                    atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:3]]
                    iv = float(atm["impliedVolatility"].mean() or 0)
            except Exception:
                pass
            iv_hv = (iv / hv) if hv else 0.0

            cur_rvol = float(ind.rvol(vol, 30).iloc[-1] or 0)

            # Accumulation proxy: OBV-like up-volume share over past 20 bars
            up_vol = (vol * (close.diff() > 0)).tail(20).sum()
            tot_vol = vol.tail(20).sum() or 1
            accum_score = (up_vol / tot_vol) * 2 - 1  # -1..+1

            # Prefer Fintel for borrow rate / live short data; fall back to provider snapshot.
            sd: dict = {}
            if self.squeeze_client:
                try:
                    sd = self.squeeze_client.short_interest(sym)
                except Exception as e:
                    log.debug("Fintel lookup failed for %s: %s", sym, e)
            if not sd:
                sd = self.provider.short_interest_detail(sym) or {}
            borrow = float(sd.get("borrow_rate") or sd.get("fee") or 0.0)
            # Override with live Fintel values when available
            if sd.get("short_pct_float"):
                short_pct = sd["short_pct_float"]
            if sd.get("days_to_cover"):
                dtc = sd["days_to_cover"]

            inp = SqueezeInputs(
                short_pct_float=short_pct,
                days_to_cover=dtc,
                borrow_rate=borrow,
                iv_hv_ratio=iv_hv,
                rvol=cur_rvol,
                float_shares=fund.get("shares_float", 0),
                price=float(close.iloc[-1]),
                accumulation_score=accum_score,
            )
            score, components = squeeze_score(inp)
            if score < 25:
                continue

            tags = ["squeeze"]
            if short_pct > 0.30: tags.append("very-high-short")
            if dtc > 5: tags.append("hard-to-cover")
            if borrow > 0.20: tags.append("expensive-borrow")
            reason = (f"short {short_pct*100:.1f}% float, DTC {dtc:.1f}d, "
                      f"IV/HV {iv_hv:.2f}, RVOL {cur_rvol:.1f}x, accum {accum_score:+.2f}")
            signals.append(Signal(
                scanner=self.name, symbol=sym, score=float(score),
                reason=reason, tags=tags,
                metrics={**components, **inp.__dict__},
            ))
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
