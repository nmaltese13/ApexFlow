"""Page routes, the GEX merge, and the retired /heatmap redirect.

Runs against the frozen demo dataset, so these exercise real rendering with
no network.
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("APEXFLOW_DEMO", "1")
os.environ.setdefault("APEXFLOW_ATLAS_DISABLE", "1")
os.environ.setdefault("APEXFLOW_BRIEFING_DISABLE", "1")

from fastapi.testclient import TestClient  # noqa: E402
import webapp  # noqa: E402

PAGES = ["/", "/dealer", "/heatseeker", "/atlas", "/radar", "/brief",
         "/earnings", "/guide", "/watchlist", "/log", "/backtest", "/symbol/SPY"]


@pytest.fixture(scope="module")
def client():
    return TestClient(webapp.app)


class TestAllPagesRender:
    @pytest.mark.parametrize("path", PAGES)
    def test_returns_200(self, client, path):
        assert client.get(path).status_code == 200

    @pytest.mark.parametrize("path", PAGES)
    def test_no_unrendered_template_syntax(self, client, path):
        """A stray {{ }} means a Jinja block leaked into the output."""
        body = client.get(path).text
        assert "{{" not in body and "{%" not in body


class TestDealerPage:
    def test_has_the_pieces_the_js_needs(self, client):
        body = client.get("/dealer").text
        for element_id in ("dg-sym", "dg-dte", "dg-convention", "dg-basis",
                           "dg-chart", "dg-totals", "dg-regime", "dg-table"):
            assert f'id="{element_id}"' in body, f"missing #{element_id}"

    def test_offers_every_positioning_convention(self, client):
        """The switchable assumption is the point of the page."""
        body = client.get("/dealer").text
        for convention in ("naive", "inverted", "all_short"):
            assert f'value="{convention}"' in body

    def test_states_that_output_is_conditional(self, client):
        body = client.get("/dealer").text.lower()
        assert "conditional on" in body
        assert "not a recommendation" in body

    def test_loads_its_script(self, client):
        assert "/static/js/dealer.js" in client.get("/dealer").text

    def test_nav_links_to_it(self, client):
        assert 'href="/dealer"' in client.get("/").text


class TestGexMergedIntoSymbol:
    def test_symbol_page_carries_the_gex_panel(self, client):
        body = client.get("/symbol/NVDA").text
        for element_id in ("gx-chart", "gx-band", "gx-nstrikes", "gx-metrics"):
            assert f'id="{element_id}"' in body, f"missing #{element_id}"

    def test_symbol_is_fixed_by_the_page(self, client):
        """The embedded panel must not carry its own symbol picker."""
        body = client.get("/symbol/NVDA").text
        assert 'id="gx-sym" value="NVDA"' in body
        assert 'type="hidden" id="gx-sym"' in body

    def test_loads_the_gex_script(self, client):
        assert "/static/js/heatmap.js" in client.get("/symbol/SPY").text

    def test_links_through_to_the_full_surface(self, client):
        assert 'href="/dealer"' in client.get("/symbol/SPY").text


class TestRetiredHeatmapRoute:
    def test_redirects_rather_than_404s(self, client):
        r = client.get("/heatmap", follow_redirects=False)
        assert r.status_code == 308
        assert r.headers["location"] == "/symbol/SPY#gex"

    def test_carries_the_symbol_through(self, client):
        r = client.get("/heatmap?sym=nvda", follow_redirects=False)
        assert r.headers["location"] == "/symbol/NVDA#gex"

    def test_following_it_lands_on_a_working_page(self, client):
        assert client.get("/heatmap").status_code == 200

    def test_nav_no_longer_offers_it(self, client):
        assert 'href="/heatmap"' not in client.get("/").text


class TestLogPayload:
    """The signal log is runtime state and is gitignored, so a fresh clone
    has none. These assertions therefore must not depend on rows existing —
    an earlier version compared two page sizes and passed only on a machine
    that happened to have a populated log."""

    def test_default_is_capped(self, client):
        """Was 200 rows / ~113KB on every load."""
        assert len(client.get("/log").text) < 60_000

    def test_out_of_range_limits_clamp_instead_of_erroring(self, client):
        for limit in (-5, 0, 1, 99999):
            assert client.get(f"/log?limit={limit}").status_code == 200

    def test_default_limit_is_fifty(self):
        """Pin the cap directly rather than inferring it from page size."""
        import inspect
        sig = inspect.signature(webapp.page_log)
        assert sig.parameters["limit"].default == 50

    def test_a_larger_limit_returns_at_least_as_much(self, client):
        """Weak by necessity: with an empty log both are the same size."""
        small = len(client.get("/log?limit=10").text)
        large = len(client.get("/log?limit=200").text)
        assert large >= small


class TestNoDeadAssets:
    def test_every_referenced_script_exists(self):
        """Catches a template pointing at a file that was deleted or renamed."""
        import re
        from pathlib import Path
        root = Path(webapp.__file__).resolve().parent
        missing = []
        for tpl in (root / "templates").glob("*.html"):
            for src in re.findall(r'src="/static/(js/[^"]+)"', tpl.read_text(encoding="utf-8")):
                if not (root / "static" / src).exists():
                    missing.append(f"{tpl.name} -> {src}")
        assert not missing, f"templates reference missing scripts: {missing}"

    def test_no_orphaned_page_scripts(self):
        """Every page script should be loaded by some template."""
        from pathlib import Path
        root = Path(webapp.__file__).resolve().parent
        templates = " ".join(t.read_text(encoding="utf-8")
                             for t in (root / "templates").glob("*.html"))
        orphans = [js.name for js in (root / "static" / "js").glob("*.js")
                   if js.name not in templates]
        assert not orphans, f"unused page scripts: {orphans}"
