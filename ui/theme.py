"""Colours, Plotly template and CSS for the research terminal look.

Series colours come from a categorical palette validated for colour-vision
deficiency on the dark chart surface (adjacent-pair ΔE ≥ 8). Colours are tied
to *roles* (strategy, benchmark, gross) and to *entities* (each strategy has a
fixed colour), never to rank, so filtering never repaints a series.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Surfaces & ink
BG = "#0e1116"
SURFACE = "#14171c"
SURFACE_2 = "#1a1f27"
BORDER = "#262b33"
GRID = "#232830"
AXIS = "#3a404a"
TEXT = "#e6e4df"
TEXT_2 = "#b4b7bd"
MUTED = "#8a8f98"

# Roles
STRATEGY = "#3987e5"  # blue  — the strategy under study
BENCHMARK = "#d95926"  # orange — the benchmark
GROSS = "#199e70"  # green — zero-cost twin of the strategy

# Fixed colour per strategy (entity, not rank)
# Buy & hold shares the benchmark's orange and the default strategy (MA) the strategy blue, so a
# series keeps its colour when moving between the single-strategy and comparison charts.
STRATEGY_COLORS = {
    "buy_and_hold": "#d95926",
    "ma_crossover": "#3987e5",
    "ts_momentum": "#199e70",
    "rsi_reversion": "#c98500",
    "bollinger_reversion": "#f2c94c",
    "adaptive_ensemble": "#008300",
    "ml_logistic": "#e6e4df",
}

# Status colours (always paired with an icon and a label)
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# Diverging blue <-> red with a neutral grey midpoint (returns heatmaps)
DIVERGING = [
    [0.0, "#c84444"],
    [0.25, "#8a3a3a"],
    [0.5, "#383835"],
    [0.75, "#245a9c"],
    [1.0, "#3987e5"],
]
# Sequential single hue (blue), dark surface -> light
SEQUENTIAL = [[0.0, "#16263b"], [0.5, "#256abf"], [1.0, "#9ec5f4"]]

BUY_MARKER = "#3fb950"
SELL_MARKER = "#f0645c"


def _template() -> go.layout.Template:
    axis = dict(
        automargin=True,
        gridcolor=GRID,
        gridwidth=1,
        zerolinecolor=AXIS,
        zerolinewidth=1,
        linecolor=AXIS,
        linewidth=1,
        tickfont=dict(color=MUTED, size=11),
        title=dict(font=dict(color=TEXT_2, size=12)),
        showspikes=True,
        spikecolor=MUTED,
        spikethickness=1,
        spikedash="solid",
        spikemode="across",
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT_2, size=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=list(STRATEGY_COLORS.values()),
            xaxis=axis,
            yaxis=axis,
            hovermode="x unified",
            hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT, size=12)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_2, size=12),
            ),
            margin=dict(l=64, r=20, t=40, b=44),
            title=dict(font=dict(color=TEXT, size=14), x=0, xanchor="left"),
        )
    )


pio.templates["quantlab"] = _template()
PLOTLY_TEMPLATE = "quantlab"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

CSS = f"""
<style>
/* Layout rhythm */
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1480px; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

/* Numbers align in tables and metrics */
[data-testid="stMetricValue"], [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ color: {TEXT_2}; font-size: 0.8rem; letter-spacing: 0.01em; }}

/* Page header */
.qlp-eyebrow {{ color: {MUTED}; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
               font-weight: 600; margin-bottom: 0.15rem; }}
.qlp-title {{ color: {TEXT}; font-size: 1.85rem; font-weight: 650; line-height: 1.2; margin: 0; }}
.qlp-subtitle {{ color: {TEXT_2}; font-size: 0.95rem; margin-top: 0.35rem; max-width: 70rem; }}

/* Context strip (ticker, period, source) */
.qlp-strip {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; padding: 0.6rem 0.9rem;
             border: 1px solid {BORDER}; border-radius: 0.5rem; background: {SURFACE};
             font-size: 0.82rem; color: {TEXT_2}; margin: 0.8rem 0 1.1rem 0; }}
.qlp-strip b {{ color: {TEXT}; font-weight: 600; }}
.qlp-strip span.k {{ color: {MUTED}; margin-right: 0.35rem; }}

/* Status pills */
.qlp-pill {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.1rem 0.55rem;
            border-radius: 999px; font-size: 0.75rem; font-weight: 600; border: 1px solid; }}
.qlp-pill.good {{ color: {GOOD}; border-color: rgba(12,163,12,0.45); background: rgba(12,163,12,0.08); }}
.qlp-pill.warn {{ color: {WARNING}; border-color: rgba(250,178,25,0.45); background: rgba(250,178,25,0.08); }}
.qlp-pill.bad  {{ color: #f07070; border-color: rgba(208,59,59,0.5); background: rgba(208,59,59,0.08); }}
.qlp-pill.info {{ color: {TEXT_2}; border-color: {BORDER}; background: {SURFACE}; }}

/* Section labels */
.qlp-section {{ color: {TEXT}; font-size: 1.05rem; font-weight: 600; margin: 1.4rem 0 0.2rem 0; }}
.qlp-caption {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 0.5rem; }}

/* Disclaimer footer */
.qlp-disclaimer {{ color: {MUTED}; font-size: 0.75rem; border-top: 1px solid {BORDER};
                  margin-top: 2.5rem; padding-top: 0.8rem; line-height: 1.5; }}

/* Empty / unavailable state */
.qlp-empty {{ border: 1px dashed {BORDER}; border-radius: 0.6rem; padding: 2rem 1.6rem;
             background: {SURFACE}; color: {TEXT_2}; }}
.qlp-empty h3 {{ color: {TEXT}; margin-top: 0; font-size: 1.15rem; }}

/* Key-value list (wraps long values, unlike a grid cell) */
.qlp-kv {{ display: grid; grid-template-columns: minmax(9rem, 38%) 1fr; border: 1px solid {BORDER};
          border-radius: 0.5rem; overflow: hidden; font-size: 0.85rem; }}
.qlp-kv div {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid {BORDER}; line-height: 1.45; }}
.qlp-kv div:nth-last-child(-n+2) {{ border-bottom: none; }}
.qlp-kv .k {{ color: {MUTED}; background: {SURFACE}; }}
.qlp-kv .v {{ color: {TEXT}; }}

/* Metric tables that wrap long definitions */
.qlp-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; border: 1px solid {BORDER};
             border-radius: 0.5rem; overflow: hidden; margin-bottom: 0.4rem; }}
.qlp-table th {{ text-align: left; color: {MUTED}; font-weight: 500; background: {SURFACE};
                padding: 0.5rem 0.7rem; border-bottom: 1px solid {BORDER}; }}
.qlp-table td {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid {BORDER}; vertical-align: top;
                color: {TEXT}; line-height: 1.45; }}
.qlp-table tr:last-child td {{ border-bottom: none; }}
.qlp-table td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; }}
.qlp-table td.def {{ color: {TEXT_2}; }}
.qlp-table .note {{ display: block; color: {WARNING}; font-size: 0.78rem; margin-top: 0.15rem; }}

/* Sidebar brand */
.qlp-brand {{ font-weight: 700; font-size: 1.05rem; color: {TEXT}; letter-spacing: 0.01em; }}
.qlp-brand small {{ display: block; color: {MUTED}; font-weight: 500; font-size: 0.72rem;
                   letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.1rem; }}
</style>
"""
