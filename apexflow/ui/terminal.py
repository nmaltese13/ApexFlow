"""Rich-powered terminal UI: signal tables, GEX heatmap, live dashboard."""
from __future__ import annotations
import time
from datetime import datetime

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

from apexflow.models import Signal, Alert
from apexflow.analytics.gex import chain_gex, gex_summary, heatmap_rows

console = Console()


def banner() -> Panel:
    title = Text("APEXFLOW", style="bold cyan")
    subtitle = Text("  options flow & market scanner suite", style="dim")
    return Panel(Align.center(Text.assemble(title, subtitle)), border_style="cyan")


def _score_color(score: float) -> str:
    if score >= 85: return "bold red"
    if score >= 70: return "bold yellow"
    if score >= 50: return "green"
    return "white"


def signals_table(scanner_label: str, signals: list[Signal], limit: int = 15) -> Table:
    table = Table(title=f"{scanner_label} — top {min(limit, len(signals))}",
                  expand=True, header_style="bold cyan")
    table.add_column("Symbol", style="bold", width=8)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Tags", width=24)
    table.add_column("Reason", overflow="fold")
    if not signals:
        table.add_row("—", "—", "—", "no signals")
        return table
    for s in signals[:limit]:
        table.add_row(
            s.symbol,
            Text(f"{s.score:.0f}", style=_score_color(s.score)),
            ", ".join(s.tags[:4]),
            s.reason,
        )
    return table


def print_signals_table(scanner_label: str, signals: list[Signal], limit: int = 15) -> None:
    console.print(signals_table(scanner_label, signals, limit))


def gex_heatmap_panel(symbol: str, calls, puts, spot: float, expiry: str) -> Panel:
    df = chain_gex(calls, puts, spot, expiry)
    summary = gex_summary(df, spot)
    rows = heatmap_rows(df, spot, n_strikes=15)

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Strike", justify="right", width=10)
    table.add_column("GEX ($M)", justify="right", width=12)
    table.add_column("Distribution", width=24)

    for strike, gx, bar in rows:
        marker = " <- spot" if abs(strike - spot) < (rows[0][0] - rows[-1][0]) / len(rows) / 2 else ""
        color = "green" if gx >= 0 else "red"
        table.add_row(
            f"{strike:.2f}{marker}",
            Text(f"{gx/1e6:+.1f}", style=color),
            Text(bar, style=color),
        )

    summary_text = (
        f"Spot: ${spot:.2f}  •  Total GEX: ${summary['total_gex']/1e9:+.2f}B  •  "
        f"Gamma flip: ${summary['gamma_flip']:.2f}" if summary["gamma_flip"]
        else f"Spot: ${spot:.2f}  •  Total GEX: ${summary['total_gex']/1e9:+.2f}B"
    )
    return Panel(Group(Text(summary_text, style="bold"), table),
                 title=f"GEX Heatmap — {symbol} ({expiry})", border_style="magenta")


def print_gex_heatmap(symbol: str, provider) -> None:
    chain = provider.options_chain(symbol)
    if not chain.get("calls") is None and not chain["calls"].empty:
        console.print(gex_heatmap_panel(symbol, chain["calls"], chain["puts"],
                                         chain["spot"], chain["expiry"]))
    else:
        console.print(f"[yellow]No options chain available for {symbol}[/]")


def print_watchlist(entries: list[dict]) -> None:
    table = Table(title="Watchlist", expand=True, header_style="bold cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Best Score", justify="right")
    table.add_column("Best Scanner")
    table.add_column("Scanners Hit")
    table.add_column("Tags")
    if not entries:
        table.add_row("—", "—", "—", "—", "empty")
    for e in entries:
        table.add_row(
            e.get("symbol", "?"),
            Text(f"{e.get('best_score', 0):.0f}", style=_score_color(e.get("best_score", 0))),
            e.get("best_scanner", ""),
            ", ".join(e.get("scanners_hit", [])),
            ", ".join(e.get("tags", [])[:3]),
        )
    console.print(table)


def print_signal_log(records: list[dict]) -> None:
    table = Table(title=f"Signal Log (last {len(records)})", expand=True, header_style="bold cyan")
    table.add_column("Time", width=19)
    table.add_column("Scanner", width=14)
    table.add_column("Symbol", style="bold", width=8)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Reason", overflow="fold")
    for r in records:
        ts = (r.get("timestamp") or "")[:19].replace("T", " ")
        score = r.get("score", 0)
        table.add_row(
            ts, r.get("scanner", ""), r.get("symbol", ""),
            Text(f"{score:.0f}", style=_score_color(score)),
            r.get("reason", ""),
        )
    console.print(table)


def alerts_panel(recent_alerts: list[Alert]) -> Panel:
    if not recent_alerts:
        return Panel("[dim]no alerts yet[/]", title="Alerts", border_style="yellow")
    lines = []
    for a in list(recent_alerts)[-12:][::-1]:
        ts = a.timestamp.strftime("%H:%M:%S")
        color = {"critical": "bold red", "warn": "yellow", "info": "white"}[a.severity]
        lines.append(Text.from_markup(f"[dim]{ts}[/] [{color}]{a.symbol}[/] ({a.scanner}) {a.message}"))
    return Panel(Group(*lines), title="Live Alerts", border_style="yellow")


def run_dashboard(hub, refresh_seconds: int = 60, iterations: int | None = None) -> None:
    """Live dashboard: runs hub on a loop and renders results."""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=14),
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    iter_count = 0
    with Live(layout, console=console, refresh_per_second=2, screen=False):
        while iterations is None or iter_count < iterations:
            t0 = time.time()
            results = hub.run_once()
            iter_count += 1

            layout["header"].update(banner())

            left_panels = []
            for name in ["options-flow", "pre-breakout", "momentum"]:
                if name in results:
                    label = hub.scanners[name].label if name in hub.scanners else name
                    left_panels.append(signals_table(label, results[name], limit=8))
            layout["left"].update(Group(*left_panels) if left_panels else Text(""))

            right_panels = []
            for name in ["squeeze", "earnings"]:
                if name in results:
                    label = hub.scanners[name].label if name in hub.scanners else name
                    right_panels.append(signals_table(label, results[name], limit=8))
            layout["right"].update(Group(*right_panels) if right_panels else Text(""))

            wl_table = Table(title=f"Watchlist (top {min(8, len(hub.watchlist.list()))})",
                             header_style="bold cyan", expand=True)
            wl_table.add_column("Symbol", style="bold")
            wl_table.add_column("Score", justify="right")
            wl_table.add_column("Hits")
            for e in hub.watchlist.list()[:8]:
                wl_table.add_row(e["symbol"],
                                 Text(f"{e['best_score']:.0f}", style=_score_color(e['best_score'])),
                                 ", ".join(e.get("scanners_hit", [])))
            footer_group = Group(wl_table, alerts_panel(list(hub.alerts.recent)))
            status = Text(f"  Provider: {hub.provider.name}  •  "
                          f"Universe: {len(hub.universe)} symbols  •  "
                          f"Run #{iter_count}  •  {datetime.now().strftime('%H:%M:%S')}  •  "
                          f"Next refresh: {refresh_seconds}s", style="dim")
            layout["footer"].update(Panel(Group(footer_group, status), border_style="dim"))

            elapsed = time.time() - t0
            sleep_for = max(1, refresh_seconds - int(elapsed))
            if iterations is None or iter_count < iterations:
                time.sleep(sleep_for)
