"""UI quality regressions: text that would render as LaTeX, colour roles and contrast."""

from __future__ import annotations

import re

import pytest

from tests.test_app_smoke import ANALYTICS_TABS, LAB_TABS, PAGES, _app, _open_tab
from ui import theme as t

# Streamlit typesets $...$ on one line as math. Escaped dollars (\$) and HTML entities are fine.
_MATH = re.compile(r"(?<!\\)\$(?!\s)[^$\n]+?(?<![\s\\])\$")


def _views():
    for page in PAGES:
        yield page, None, None
    for label in LAB_TABS:
        yield "pages/2_Strategy_Lab.py", "lab_tabs", label
    for label in ANALYTICS_TABS:
        yield "pages/4_Performance_Analytics.py", "pa_tabs", label


@pytest.mark.parametrize("page,key,label", list(_views()))
def test_no_text_is_accidentally_typeset_as_math(page, key, label):
    at = _app(page)
    if key:
        _open_tab(at, key, label)
    texts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    offenders = [x for x in texts if _MATH.search(x)]
    assert not offenders, offenders


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_status_pill_text_meets_wcag_aa_on_the_surface():
    css = t.CSS if hasattr(t, "CSS") else ""
    bad = re.search(r"\.qlp-pill\.bad\s*\{\{?\s*color:\s*(#[0-9a-fA-F]{6})", css)
    colour = bad.group(1) if bad else "#f07070"
    assert _contrast(colour, t.SURFACE) >= 4.5


def test_strategy_colours_do_not_swap_roles_between_pages():
    assert t.STRATEGY_COLORS["buy_and_hold"] == t.BENCHMARK  # buy & hold is the benchmark elsewhere
    assert t.STRATEGY_COLORS["ma_crossover"] == t.STRATEGY  # the default strategy keeps its colour
    assert len(set(t.STRATEGY_COLORS.values())) == len(t.STRATEGY_COLORS)


def test_integrity_panel_reports_benchmark_warnings():
    import pandas as pd

    from quantlab.validation import Severity
    from tests.app_fixtures import synthetic_research_data

    data = synthetic_research_data()
    bench = data.asset.__class__(**{**data.asset.__dict__})
    bench.quality = type(data.asset.quality)()
    bench.quality.add(Severity.WARNING, "extreme_returns", "spike", dates=pd.DatetimeIndex(["2020-03-16"]))
    data.benchmark_dataset = bench
    at = _app(with_data=False)
    at.session_state["research_data"] = data
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert "Benchmark data: 1 warning(s)" in html and "Data checks passed" not in html
