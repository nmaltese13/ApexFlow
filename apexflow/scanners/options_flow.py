"""Unusual options activity scanner.

What works on free data (yfinance):
  - Volume / OI ratio per contract
  - OTM call/put volume spikes
  - Aggregate bullish vs bearish premium proxy
  - GEX heatmap (computed from chain)

What requires paid feed:
  - Sweep / block / split-strike detection with real-time prints
  - At-ask vs at-bid classification (aggressive buying/selling)
  - Dark pool prints + cross-reference
"""
from __future__ import annotations
import logging

import config
from apexflow.analytics.gex import chain_gex, gex_summary
from apexflow.models import Signal
from .base import BaseScanner

log = logging.getLogger(__name__)


class OptionsFlowScanner(BaseScanner):
    name = "options-flow"
    label = "Options Flow / Unusual Activity"

    def scan(self, universe: list[str]) -> list[Signal]:
        signals: list[Signal] = []

        # 1. Real flow from Unusual Whales (if key set in keys.py)
        real_flow = []
        if self.flow_client:
            try:
                real_flow = self.flow_client.options_flow(since_minutes=30)
            except Exception as e:
                log.debug("flow_client failed: %s", e)
        if real_flow:
            agg: dict[str, dict] = {}
            for p in real_flow:
                a = agg.setdefault(p.symbol, {"call_prem": 0.0, "put_prem": 0.0,
                                              "ask_prem": 0.0, "sweeps": 0, "blocks": 0})
                if p.right == "C":
                    a["call_prem"] += p.premium
                else:
                    a["put_prem"] += p.premium
                if p.side == "ASK":
                    a["ask_prem"] += p.premium
                if "SWEEP" in p.print_type:
                    a["sweeps"] += 1
                if "BLOCK" in p.print_type:
                    a["blocks"] += 1
            for sym, a in agg.items():
                total = a["call_prem"] + a["put_prem"]
                if total < config.LARGE_BLOCK_PREMIUM:
                    continue
                bull = a["call_prem"] - a["put_prem"]
                ask_ratio = a["ask_prem"] / total if total else 0
                score = min(100, 30 + ask_ratio * 40 + min(a["sweeps"] + a["blocks"], 10) * 3)
                signals.append(Signal(
                    scanner=self.name, symbol=sym, score=float(score),
                    reason=(f"${total/1e6:.1f}M flow ({'CALL' if bull > 0 else 'PUT'} bias), "
                            f"{ask_ratio*100:.0f}% at ASK, "
                            f"{a['sweeps']} sweeps / {a['blocks']} blocks"),
                    tags=["real-flow", "bullish" if bull > 0 else "bearish"],
                    metrics=a,
                ))

        # 2. Volume/OI scan from chains (works on yfinance)
        for sym in universe:
            try:
                chain = self.provider.options_chain(sym)
            except Exception as e:
                log.debug("chain failed for %s: %s", sym, e)
                continue
            spot = chain.get("spot") or 0
            calls = chain.get("calls")
            puts = chain.get("puts")
            if spot <= 0 or calls is None or puts is None or calls.empty:
                continue

            unusual = []
            for df, right in ((calls, "C"), (puts, "P")):
                if df.empty:
                    continue
                d = df.copy()
                d = d[(d["openInterest"].fillna(0) > 100) & (d["volume"].fillna(0) > 200)]
                d["vol_oi"] = d["volume"] / d["openInterest"].replace(0, 1)
                hot = d[d["vol_oi"] >= config.UNUSUAL_VOLUME_RATIO]
                for _, row in hot.iterrows():
                    moneyness = row["strike"] / spot - 1
                    otm = (right == "C" and moneyness > 0.02) or (right == "P" and moneyness < -0.02)
                    unusual.append({
                        "right": right, "strike": float(row["strike"]),
                        "vol": int(row["volume"]), "oi": int(row["openInterest"]),
                        "vol_oi": float(row["vol_oi"]), "otm": otm,
                        "iv": float(row.get("impliedVolatility") or 0),
                    })

            if not unusual:
                continue
            top = sorted(unusual, key=lambda x: x["vol_oi"], reverse=True)[:3]
            otm_count = sum(1 for u in unusual if u["otm"])
            call_vol = sum(u["vol"] for u in unusual if u["right"] == "C")
            put_vol = sum(u["vol"] for u in unusual if u["right"] == "P")
            cp_ratio = call_vol / put_vol if put_vol else float("inf")
            score = min(100, 25 + min(len(unusual), 10) * 4 + min(otm_count, 5) * 4)

            # GEX context
            gex_df = chain_gex(calls, puts, spot, chain["expiry"])
            gx = gex_summary(gex_df, spot)

            top_str = ", ".join(f"{u['right']}{u['strike']:g}@{u['vol_oi']:.1f}x" for u in top)
            signals.append(Signal(
                scanner=self.name, symbol=sym, score=float(score),
                reason=(f"{len(unusual)} unusual contracts (top: {top_str}), "
                        f"C/P vol={cp_ratio:.2f}, "
                        f"GEX={gx['total_gex']/1e9:+.2f}B, flip={gx['gamma_flip']}"),
                tags=["volume>>oi"] + (["otm-spike"] if otm_count else []) +
                     (["bullish"] if cp_ratio > 1.5 else ["bearish"] if cp_ratio < 0.67 else []),
                metrics={"unusual": unusual, "gex": gx, "spot": spot},
            ))

        # 3. Dark pool cross-reference (Unusual Whales)
        dp = []
        if self.flow_client:
            try:
                dp = self.flow_client.dark_pool_prints(since_minutes=60)
            except Exception as e:
                log.debug("dark pool fetch failed: %s", e)
        if dp:
            by_sym: dict[str, float] = {}
            for d in dp:
                by_sym[d.symbol] = by_sym.get(d.symbol, 0) + d.notional
            existing = {s.symbol for s in signals}
            for sym, notional in by_sym.items():
                if notional < 5_000_000:
                    continue
                if sym in existing:
                    for s in signals:
                        if s.symbol == sym:
                            s.tags.append("dark-pool")
                            s.metrics["dark_pool_notional"] = notional
                            s.score = min(100.0, s.score + 10)
                else:
                    signals.append(Signal(
                        scanner=self.name, symbol=sym, score=40.0,
                        reason=f"Dark pool ${notional/1e6:.1f}M (no options confirmation)",
                        tags=["dark-pool"], metrics={"dark_pool_notional": notional},
                    ))

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
