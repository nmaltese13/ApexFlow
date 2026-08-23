"""ApexFlow CLI entry point.

Examples:
    python main.py dashboard
    python main.py scan options-flow --universe sp100
    python main.py scan pre-breakout
    python main.py hub --once
    python main.py backtest pre-breakout --hold 5
    python main.py log --last 25
    python main.py gex AAPL
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

import click

import config
from apexflow.providers import get_provider
from apexflow.scanners import ALL_SCANNERS
from apexflow.platform import ScannerHub, Backtester, SignalLogger, Watchlist
from apexflow.ui import (
    print_signals_table, print_gex_heatmap, print_watchlist,
    print_signal_log, run_dashboard, banner,
)
from apexflow.universe import load_universe
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
@click.option("--demo/--no-demo", default=None,
              help="Serve the frozen chain snapshot in data/demo (no API key, "
                   "no network). Same as APEXFLOW_DEMO=1.")
def cli(demo):
    """ApexFlow — options flow & scanner suite."""
    if demo is not None:
        # Set before anything imports a provider, so the selection sees it.
        os.environ["APEXFLOW_DEMO"] = "1" if demo else "0"
        import importlib
        importlib.reload(config)
        from apexflow.providers import reset_provider
        reset_provider()
    console.print(banner())
    cfg = config.configured_providers()
    active = [name for name, ok in cfg.items() if ok]
    if active:
        console.print(f"[green]Active extras: {', '.join(active)}[/green]")
    else:
        console.print("[yellow]Free mode (yfinance). Paste API keys in keys.py to unlock more.[/yellow]\n"
                      "[dim]Run `python main.py status` to see what's configurable.[/dim]\n")


@cli.command()
def status():
    """Show which API keys are configured."""
    from rich.table import Table
    table = Table(title="ApexFlow API Key Status", header_style="bold cyan", expand=True)
    table.add_column("Service", style="bold")
    table.add_column("Configured", justify="center")
    table.add_column("Purpose")
    purposes = {
        "Unusual Whales (flow + dark pool)": "Real sweep/block flow + dark pool prints",
        "Fintel (short interest / squeeze)":  "Live short %, days-to-cover, borrow rate",
        "Finnhub (news + earnings)":          "News catalysts + earnings calendar",
    }
    for name, ok in config.configured_providers().items():
        table.add_row(name,
                      "[green]YES[/green]" if ok else "[dim]no[/dim]",
                      purposes.get(name, ""))
    console.print(table)
    active = config.primary_provider_name()
    console.print(f"\n[bold]Price/options data:[/bold] [cyan]{active}[/cyan]")
    if active == "yfinance":
        console.print("[dim]free, ~15-min delayed, rate-limited[/dim]")
    console.print("[dim]Edit keys.py to add a key — it auto-activates on next run.[/dim]")

    if config.demo_dataset_present():
        import json as _json
        m = _json.loads((config.DEMO_DIR / "manifest.json").read_text())
        syms = ", ".join(s["symbol"] for s in m.get("symbols", []))
        console.print(f"\n[bold]Demo snapshot:[/bold] {m.get('capture_date','?')} "
                      f"from {m.get('source','?')} — {m.get('total_contracts',0):,} contracts")
        console.print(f"[dim]{syms}[/dim]")
        console.print("[dim]Run offline with: python main.py --demo web[/dim]")
    else:
        console.print("\n[dim]No demo snapshot. Build one with "
                      "`python scripts/capture_demo_dataset.py`.[/dim]")


@cli.command()
@click.option("--universe", default=config.DEFAULT_UNIVERSE, help="sp100 | sp500 | squeeze | path/to/file")
@click.option("--refresh", default=60, type=int, help="seconds between hub iterations")
@click.option("--iterations", default=None, type=int, help="stop after N iterations (default: forever)")
def dashboard(universe, refresh, iterations):
    """Live dashboard — runs all scanners on a loop."""
    syms = load_universe(universe)
    hub = ScannerHub(universe=syms)
    try:
        run_dashboard(hub, refresh_seconds=refresh, iterations=iterations)
    except KeyboardInterrupt:
        console.print("\n[dim]stopped[/]")


@cli.command()
@click.argument("scanner", type=click.Choice(list(ALL_SCANNERS.keys())))
@click.option("--universe", default=config.DEFAULT_UNIVERSE)
@click.option("--limit", default=20, type=int)
def scan(scanner, universe, limit):
    """Run a single scanner once and print results."""
    syms = load_universe(universe)
    p = get_provider()
    s = ALL_SCANNERS[scanner](p)
    console.print(f"[cyan]Scanning {len(syms)} symbols with {s.label}...[/]")
    sigs = s.scan_safe(syms)
    print_signals_table(s.label, sigs, limit=limit)
    SignalLogger().log_many(sigs)


@cli.command()
@click.option("--universe", default=config.DEFAULT_UNIVERSE)
@click.option("--once/--loop", default=False)
@click.option("--refresh", default=config.HUB_LOOP_SECONDS, type=int)
def hub(universe, once, refresh):
    """Scanner Hub — runs all five scanners in parallel; populates watchlist; fires alerts."""
    syms = load_universe(universe)
    h = ScannerHub(universe=syms)
    if once:
        results = h.run_once()
        for name, sigs in results.items():
            label = h.scanners[name].label
            print_signals_table(label, sigs, limit=10)
        print_watchlist(h.watchlist.list()[:25])
    else:
        try:
            h.loop(sleep_s=refresh)
        except KeyboardInterrupt:
            console.print("\n[dim]hub stopped[/]")


@cli.command()
@click.argument("scanner", type=click.Choice(list(ALL_SCANNERS.keys())))
@click.option("--universe", default=config.DEFAULT_UNIVERSE)
@click.option("--lookback", default=90, type=int)
@click.option("--hold", default=5, type=int, help="bars to hold after signal")
def backtest(scanner, universe, lookback, hold):
    """Backtest a scanner over historical data."""
    syms = load_universe(universe)
    bt = Backtester()
    console.print(f"[cyan]Backtesting {scanner} over {lookback}d, holding {hold}d, {len(syms)} symbols...[/]")
    try:
        result = bt.run(scanner, syms, lookback_days=lookback, hold_days=hold)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return
    console.print(f"[bold green]{result.summary}[/]")
    console.print(f"  median return: {result.median_return_pct:+.2f}%   "
                  f"wins: {result.wins}   losses: {result.losses}")


@cli.command(name="log")
@click.option("--last", default=50, type=int)
def log_cmd(last):
    """Show the most recent signals from the signal log."""
    sl = SignalLogger()
    records = list(sl.iter_recent(last))
    print_signal_log(records)


@cli.command()
def watchlist_cmd():
    """Show the current auto-populated watchlist."""
    print_watchlist(Watchlist().list())


cli.add_command(watchlist_cmd, name="watchlist")


@cli.command()
@click.option("--port", default=8501, type=int, help="Localhost port (default 8501)")
@click.option("--reload/--no-reload", default=False, help="Auto-reload on code changes (dev)")
def web(port, reload):
    """Launch the browser-based web app at http://localhost:8501"""
    import webbrowser, threading, uvicorn
    url = f"http://localhost:{port}"
    console.print(f"[cyan]Starting ApexFlow web app at {url}[/]")
    console.print("[dim]Press Ctrl+C in this window to stop the server.[/]\n")
    if not reload:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("webapp:app", host="127.0.0.1", port=port,
                reload=reload, log_level="warning")


@cli.command()
@click.argument("symbol")
def gex(symbol):
    """Print the GEX heatmap for a single symbol."""
    p = get_provider()
    print_gex_heatmap(symbol.upper(), p)



@cli.command(name="schwab-auth")
def schwab_auth():
    """One-time OAuth dance to get Schwab refresh tokens.

    Requires SCHWAB_APP_KEY and SCHWAB_APP_SECRET in keys.py. Opens a browser
    to the Schwab login page; after you approve, tokens are saved to
    data/schwab_tokens.json. Re-run weekly when the refresh token expires.
    """
    if not (config.SCHWAB_APP_KEY and config.SCHWAB_APP_SECRET):
        console.print("[red]Missing SCHWAB_APP_KEY or SCHWAB_APP_SECRET in keys.py[/red]")
        return
    from apexflow.providers.schwab_provider import run_local_auth_flow
    from apexflow.providers import reset_provider
    try:
        run_local_auth_flow(
            app_key=config.SCHWAB_APP_KEY,
            app_secret=config.SCHWAB_APP_SECRET,
            redirect_uri=config.SCHWAB_REDIRECT_URI,
        )
        reset_provider()
        console.print("[green]Schwab tokens saved. Provider will switch to Schwab on next call.[/green]")
    except Exception as e:
        console.print(f"[red]Auth failed: {e}[/red]")


@cli.command(name="validate-mc")
@click.option("--paths", default=1_000_000, type=int,
              help="paths for the terminal-distribution checks")
@click.option("--steps", default=128, type=int,
              help="time steps for the barrier check")
@click.option("--converge/--no-converge", default=True,
              help="also print the convergence table up to 5,000,000 paths")
@click.option("--spot", default=100.0, type=float)
@click.option("--iv", default=0.30, type=float)
@click.option("--days", default=91.25, type=float, help="horizon in calendar days")
def validate_mc(paths, steps, converge, spot, iv, days):
    """Check the Monte Carlo engine against closed-form values.

    Prices European options, terminal digitals, the martingale expectation
    and a one-touch barrier by simulation, and compares each against its
    analytic value. `z` is the error measured in Monte Carlo standard
    errors — under about 3 across the board means the simulator agrees with
    the mathematics to within sampling noise.
    """
    from rich.table import Table
    from apexflow.analytics import montecarlo as amc

    t = days / 365.0
    console.print(f"[cyan]Validating Monte Carlo: spot={spot} iv={iv:.0%} "
                  f"horizon={days:.2f}d paths={paths:,} steps={steps}[/]\n")

    rows = amc.validate(spot=spot, t=t, iv=iv, n_paths=paths, n_steps=steps)
    table = Table(title="Simulated vs closed form", header_style="bold cyan", expand=True)
    for col, just in (("Check", "left"), ("Analytic", "right"), ("Simulated", "right"),
                      ("Std err", "right"), ("z", "right"), ("Verdict", "center")):
        table.add_column(col, justify=just)
    worst = 0.0
    for r in rows:
        z = r["z"]
        worst = max(worst, abs(z))
        ok = abs(z) < 4.0
        table.add_row(r["check"], f"{r['analytic']:.6f}", f"{r['simulated']:.6f}",
                      f"{r['stderr']:.6f}", f"{z:+.2f}",
                      "[green]ok[/green]" if ok else "[red]OFF[/red]")
    console.print(table)

    if worst < 4.0:
        console.print(f"[green]All checks within sampling noise "
                      f"(max |z| = {worst:.2f}).[/green]")
    else:
        console.print(f"[red]Max |z| = {worst:.2f} — investigate.[/red]")

    if converge:
        console.print()
        ct = Table(title="Convergence — ATM call, no control variate",
                   header_style="bold cyan", expand=True)
        for col in ("Paths", "Simulated", "Abs error", "Std err", "Seconds"):
            ct.add_column(col, justify="right")
        for r in amc.convergence_table(spot=spot, t=t, iv=iv):
            ct.add_row(f"{r['n_paths']:,}", f"{r['simulated']:.5f}",
                       f"{r['abs_error']:.5f}", f"{r['stderr']:.6f}",
                       f"{r['elapsed_s']:.2f}")
        console.print(ct)
        console.print("[dim]Standard error falls as 1/sqrt(N). At 5,000,000 paths the "
                      "error on a probability is ~2 bp — an order of magnitude below "
                      "the 0.1% the UI renders.[/dim]")


@cli.command(name="backtest-squeeze")
@click.option("--universe", default="sp100", help="sp100 | sp500 | squeeze | path/to/file")
@click.option("--limit", default=60, type=int, help="cap the universe size")
@click.option("--lookback", default=400, type=int, help="calendar days of signal dates")
@click.option("--hold", default=10, type=int, help="forward-return horizon in bars")
@click.option("--permutations", default=1000, type=int, help="shuffles for the null")
@click.option("--vol-proxy/--no-vol-proxy", default=False,
              help="substitute a realised-vol ratio for the IV/HV axis (see docs)")
def backtest_squeeze(universe, limit, lookback, hold, permutations, vol_proxy):
    """Test whether a higher squeeze score predicts a better forward return.

    Evaluates the score cross-sectionally: on each date, does the ranking of
    scores match the ranking of forward returns? Reports a rank information
    coefficient against a permutation null, corrected for the fact that
    overlapping forward windows are not independent observations.

    Most of the score cannot be reconstructed point-in-time from free data —
    short interest, days-to-cover, borrow rate and float are only available
    as *current* values, and using them to score a past date is lookahead
    bias. The run reports its coverage and refuses a verdict when it is too
    low to mean anything. See docs/methodology.md.
    """
    from rich.table import Table
    from apexflow.platform import SqueezeBacktester, PriceDerivedSource

    syms = load_universe(universe)[:limit]
    source = PriceDerivedSource(use_realised_vol_proxy=vol_proxy)

    console.print(f"[cyan]Backtesting squeeze score — {len(syms)} symbols, "
                  f"{lookback}d of dates, {hold}-bar hold[/]")
    console.print(f"[dim]Point-in-time coverage: {source.coverage:.0%} of the score "
                  f"({', '.join(sorted(source.covers))})[/dim]\n")

    bt = SqueezeBacktester(source=source)
    panel = bt.build_panel(syms, lookback_days=lookback, hold_days=hold)
    if panel.empty:
        console.print("[red]No observations — check the universe and provider.[/red]")
        return
    rep = bt.evaluate(panel, hold_days=hold, n_permutations=permutations)

    stats = Table(title="Cross-sectional result", header_style="bold cyan", expand=True)
    stats.add_column("Metric"); stats.add_column("Value", justify="right")
    stats.add_row("Observations", f"{rep.n_observations:,}")
    stats.add_row("Dates", f"{rep.n_dates:,}")
    stats.add_row("Mean breadth (names/date)", f"{rep.mean_breadth:.1f}")
    stats.add_row("Rank IC (all dates)", f"{rep.rank_ic_all_dates:+.4f}")
    stats.add_row("Independent dates per subsample", f"{rep.n_independent_dates}")
    stats.add_row("t-statistic (offset-averaged)", f"{rep.t_stat:+.2f}")
    stats.add_row("t spread across offsets", f"{rep.t_stat_spread:.2f}")
    stats.add_row("Permutation-null percentile", f"{rep.null_percentile:.1f}")
    stats.add_row("Top-minus-bottom bucket", f"{rep.top_minus_bottom:+.3f}%")
    console.print(stats)

    if rep.decile_returns:
        dec = Table(title=f"Mean {hold}-bar forward return by score bucket",
                    header_style="bold cyan", expand=True)
        for col in ("Bucket", "N", "Mean score", "Mean return", "Median return"):
            dec.add_column(col, justify="right")
        for r in rep.decile_returns:
            colour = "green" if r["mean_return"] > 0 else "red"
            dec.add_row(str(r["bucket"]), f"{r['n']:,}", f"{r['mean_score']:.1f}",
                        f"[{colour}]{r['mean_return']:+.3f}%[/{colour}]",
                        f"{r['median_return']:+.3f}%")
        console.print(dec)

    console.print()
    if rep.conclusive:
        console.print(f"[bold]Verdict:[/bold] {rep.verdict}")
    else:
        console.print(f"[bold yellow]Inconclusive.[/bold yellow] {rep.verdict}")
    console.print("\n[bold]Caveats[/bold]")
    for c in rep.caveats:
        console.print(f"  [dim]· {c}[/dim]")


@cli.command(name="demo-info")
def demo_info():
    """Describe the frozen demo dataset, if one is present."""
    from rich.table import Table
    from apexflow.providers.demo_provider import DemoProvider, demo_available

    if not demo_available():
        console.print("[yellow]No demo dataset found in data/demo.[/yellow]")
        console.print("Build one with: [cyan]python scripts/capture_demo_dataset.py[/cyan]")
        return
    p = DemoProvider()
    info = p.info()
    console.print(f"[bold]Captured:[/bold] {info['capture_date']} "
                  f"from [cyan]{info['source']}[/cyan] — "
                  f"{info['total_contracts']:,} contracts")
    if info["shifted"]:
        console.print(f"[dim]Dates shifted forward {info['shift_days']} days "
                      f"(whole weeks) so DTEs stay realistic. "
                      f"Set APEXFLOW_DEMO_SHIFT=0 for raw captured dates.[/dim]")
    table = Table(header_style="bold cyan", expand=True)
    for col in ("Symbol", "Spot", "Expiries", "First", "Last"):
        table.add_column(col)
    for sym in info["symbols"]:
        exps = p.expiries(sym)
        q = p.quote(sym)
        table.add_row(sym, f"{q.get('price', 0):.2f}", str(len(exps)),
                      exps[0] if exps else "-", exps[-1] if exps else "-")
    console.print(table)
    console.print("\nRun offline: [cyan]python main.py --demo web[/cyan]")


@cli.command(name="dealer-greeks")
@click.argument("symbol")
@click.option("--dte", default=30, type=int, help="roll expiries within N days")
@click.option("--convention", default="naive",
              type=click.Choice(["naive", "inverted", "all_short"]),
              help="dealer positioning assumption")
def dealer_greeks_cmd(symbol, dte, convention):
    """Print the dealer exposure surface (DEX/GEX/VEX/charm/vanna) for a symbol."""
    from datetime import datetime, timedelta, timezone
    from rich.table import Table
    import pandas as pd
    from apexflow.analytics.dealer_greeks import (
        chain_exposures, roll_up, exposure_summary, vex_summary, gamma_flip_level)
    from apexflow.analytics.iv_surface import first_usable_atm_iv

    sym = symbol.upper()
    prov = get_provider()
    expiries = prov.expiries(sym) or []
    if not expiries:
        console.print(f"[red]No expiries for {sym}[/red]")
        return
    cutoff = datetime.now(timezone.utc).date() + timedelta(days=dte)
    used = [e for e in expiries
            if datetime.strptime(e, "%Y-%m-%d").date() <= cutoff] or [expiries[0]]

    frames, raw, spot = [], [], 0.0
    for e in used:
        ch = prov.options_chain(sym, e)
        spot = float(ch.get("spot") or spot)
        calls = ch.get("calls")
        puts = ch.get("puts") if ch.get("puts") is not None else pd.DataFrame()
        if calls is None or calls.empty:
            continue
        raw.append((e, calls, puts))
        frames.append(chain_exposures(calls, puts, spot, e, convention=convention))
    if not frames:
        console.print(f"[red]No usable chains for {sym}[/red]")
        return

    rolled = roll_up(frames)
    s = exposure_summary(rolled, spot)
    v = vex_summary(rolled, spot)
    flip = gamma_flip_level(raw, spot, convention=convention)
    iv_val, iv_quality, iv_expiry = first_usable_atm_iv(raw, spot)

    console.print(f"\n[bold]{sym}[/bold]  spot [cyan]{spot:,.2f}[/cyan]   "
                  f"{len(used)} expiries   convention [cyan]{convention}[/cyan]")
    console.print(f"ATM IV [cyan]{iv_val:.2%}[/cyan] "
                  f"([dim]{iv_quality}, from {iv_expiry}[/dim])\n")

    t = Table(header_style="bold cyan", expand=True)
    t.add_column("Exposure"); t.add_column("Total", justify="right")
    t.add_column("Reads as")
    t.add_row("DEX", f"{s['total_dex']:>20,.0f}", "$ of stock dealers hold")
    t.add_row("GEX", f"{s['total_gex']:>20,.0f}", "$ of delta traded per +1% spot")
    t.add_row("VEX", f"{s['total_vex']:>20,.0f}", "$ P&L per +1 vol point")
    t.add_row("Charm", f"{s['total_charm']:>20,.0f}", "$ of delta to re-hedge per day")
    t.add_row("Vanna", f"{s['total_vanna']:>20,.0f}", "$ of delta per +1 vol point")
    console.print(t)

    regime = flip.get("regime", "unknown")
    colour = "green" if regime == "positive_gamma" else "red"
    f = flip.get("flip")
    console.print(f"\nRegime: [{colour}]{regime.replace('_', ' ')}[/{colour}]   "
                  f"zero-gamma level: "
                  f"[cyan]{f:,.2f}[/cyan]" if f else
                  f"\nRegime: [{colour}]{regime}[/{colour}]   "
                  f"zero-gamma level: [dim]none within ±25%[/dim]")
    console.print(f"Dealers net [bold]{'short' if v['net_short_vega'] else 'long'}[/bold] "
                  f"vega; {v['pct_near_spot']:.0%} of it sits within 5% of spot")
    console.print("\n[dim]Conditional on the stated positioning assumption. "
                  "Describes structure, not direction.[/dim]")


if __name__ == "__main__":
    cli()
