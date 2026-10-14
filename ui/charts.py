"""Plotly figure builders. Pure functions: results in, figure out.

Mark conventions: 2px lines, hairline grids, one y-axis per panel (never dual
axes), legends for every multi-series chart, hover on every mark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantlab.analytics import MONTH_LABELS
from ui import theme as t

LINE = 2


def _base(fig: go.Figure, height: int = 380, title: str | None = None, y_title: str | None = None,
          y_format: str | None = None) -> go.Figure:
    fig.update_layout(template=t.PLOTLY_TEMPLATE, height=height)
    if title:
        fig.update_layout(title=dict(text=title), margin=dict(t=72))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    return fig


def _line(x, y, name, color, width=LINE, dash=None, hover_fmt=",.2f", fill=None, fillcolor=None,
          showlegend=True, prefix="") -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        fill=fill, fillcolor=fillcolor, showlegend=showlegend,
        hovertemplate=f"{prefix}%{{y:{hover_fmt}}}<extra>{name}</extra>",
    )


# --------------------------------------------------------------------------- #
# Price, signals and executions
# --------------------------------------------------------------------------- #


OVERLAY_STYLE = {
    "ma_short": ("Short MA", t.GROSS, None),
    "ma_long": ("Long MA", t.BENCHMARK, None),
    "bb_middle": ("Middle band", t.BENCHMARK, None),
    "bb_upper": ("Upper band", t.MUTED, None),
    "bb_lower": ("Lower band", t.MUTED, None),
}


def price_with_trades(result, strategy, height: int = 520) -> go.Figure:
    """Close price, strategy indicators and fills at their actual execution dates/prices.

    Strategies can add a lower panel via ``oscillator_spec()`` (RSI, model state, P(up)).
    """
    daily, signals, ledger = result.daily, result.signals, result.ledger
    spec = strategy.oscillator_spec()
    fig = make_subplots(rows=2 if spec else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28] if spec else [1.0], vertical_spacing=0.04)
    fig.add_trace(_line(daily.index, daily["close"], "Close", t.STRATEGY, prefix="$"), row=1, col=1)
    for col in strategy.overlay_columns:
        if col in signals:
            label, color, dash = OVERLAY_STYLE.get(col, (col, t.MUTED, None))
            fig.add_trace(_line(signals.index, signals[col], label, color, width=1.4, dash=dash, prefix="$"),
                          row=1, col=1)
    for side, symbol, color in (("BUY", "triangle-up", t.BUY_MARKER), ("SELL", "triangle-down", t.SELL_MARKER)):
        fills = ledger[ledger["side"] == side]
        if len(fills):
            fig.add_trace(
                go.Scatter(
                    x=fills["execution_date"], y=fills["execution_price"], mode="markers", name=f"{side.title()} fill",
                    marker=dict(symbol=symbol, size=10, color=color, line=dict(color=t.BG, width=1.5)),
                    customdata=np.stack([fills["signal_date"].dt.strftime("%Y-%m-%d"), fills["quantity"],
                                         fills["target_weight"], fills["reason"]], axis=-1),
                    hovertemplate=(f"<b>{side}</b> @ $%{{y:,.2f}}<br>signal %{{customdata[0]}}"
                                   "<br>qty %{customdata[1]:,.4f} → target %{customdata[2]:.0%}"
                                   "<br>%{customdata[3]}<extra></extra>"),
                ),
                row=1, col=1,
            )
    if spec:
        palette = [t.GROSS, t.BENCHMARK, t.STRATEGY]
        for (col, label), color in zip(spec["columns"].items(), palette):
            if col in signals:
                fig.add_trace(_line(signals.index, signals[col], label, color, width=1.4), row=2, col=1)
        for n_level, (level, name) in enumerate(spec.get("levels", [])):
            fig.add_hline(y=level, line=dict(color=t.MUTED, width=1), row=2, col=1,
                          annotation_text=f"{name} {level:g}", annotation_font_color=t.MUTED,
                          annotation_position="top right" if n_level % 2 else "bottom right")
        fig.update_yaxes(title_text=spec["title"], row=2, col=1)
        if spec.get("range"):
            fig.update_yaxes(range=list(spec["range"]), row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    return _base(fig, height=height)


# --------------------------------------------------------------------------- #
# Equity, drawdown, exposure
# --------------------------------------------------------------------------- #


def equity_vs_benchmark(net, bench, gross=None, normalise: bool = False, log: bool = False,
                        height: int = 400, strategy_name: str = "Strategy (net)",
                        bench_name: str = "Benchmark") -> go.Figure:
    fig = go.Figure()
    scale = (lambda s: s / s.iloc[0] * 100) if normalise else (lambda s: s)
    unit = "" if normalise else "$"
    if gross is not None:
        fig.add_trace(_line(gross.equity.index, scale(gross.equity), "Strategy (gross, no costs)", t.GROSS,
                            width=1.5, prefix=unit))
    fig.add_trace(_line(bench.equity.index, scale(bench.equity), bench_name, t.BENCHMARK, prefix=unit))
    fig.add_trace(_line(net.equity.index, scale(net.equity), strategy_name, t.STRATEGY, prefix=unit))
    fig = _base(fig, height=height, y_title="Growth of 100" if normalise else "Portfolio value ($)")
    if log:
        fig.update_yaxes(type="log")
    return fig


def drawdowns(net, bench, height: int = 300) -> go.Figure:
    fig = go.Figure()
    for result, name, color, fill in ((bench, "Benchmark", t.BENCHMARK, None),
                                      (net, "Strategy", t.STRATEGY, "tozeroy")):
        eq = result.equity
        dd = eq / eq.cummax() - 1
        fig.add_trace(_line(dd.index, dd, name, color, width=1.6 if fill is None else LINE, hover_fmt=".2%",
                            fill=fill, fillcolor="rgba(57,135,229,0.15)" if fill else None))
    return _base(fig, height=height, y_title="Drawdown from peak", y_format=".0%")


def exposure(net, height: int = 220) -> go.Figure:
    exp = net.daily["exposure"]
    fig = go.Figure(go.Scatter(x=exp.index, y=exp, mode="lines", line=dict(color=t.STRATEGY, width=1.5, shape="hv"),
                               fill="tozeroy", fillcolor="rgba(57,135,229,0.18)", name="Exposure",
                               hovertemplate="%{y:.0%} invested<extra></extra>"))
    fig = _base(fig, height=height, y_title="Invested share", y_format=".0%")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(showlegend=False)
    return fig


def rolling_metric(series_by_name: dict[str, tuple[pd.Series, str]], y_title: str, y_format: str | None,
                   height: int = 300, zero_line: bool = False) -> go.Figure:
    fig = go.Figure()
    for name, (series, color) in series_by_name.items():
        fig.add_trace(_line(series.index, series, name, color, hover_fmt=".2f" if y_format is None else ".1%"))
    fig = _base(fig, height=height, y_title=y_title, y_format=y_format)
    if zero_line:
        fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    return fig


# --------------------------------------------------------------------------- #
# Period returns and distributions
# --------------------------------------------------------------------------- #


def monthly_heatmap(table: pd.DataFrame, height: int | None = None) -> go.Figure:
    months = table[MONTH_LABELS]
    z = months.to_numpy(dtype=float)
    bound = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.01
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:+.1%}")(np.nan_to_num(z)))
    fig = go.Figure(
        go.Heatmap(
            z=z, x=MONTH_LABELS, y=[str(y) for y in months.index], colorscale=t.DIVERGING,
            zmin=-bound, zmax=bound, zmid=0, xgap=2, ygap=2, text=text, texttemplate="%{text}",
            textfont=dict(size=11), hoverongaps=False,
            hovertemplate="%{y} %{x}: %{z:+.2%}<extra></extra>",
            colorbar=dict(tickformat=".0%", outlinewidth=0, thickness=10, tickfont=dict(color=t.MUTED)),
        )
    )
    fig = _base(fig, height=height or max(220, 34 * len(months) + 80))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(hovermode="closest")
    return fig


def annual_returns(strategy: pd.Series, benchmark: pd.Series, height: int = 320) -> go.Figure:
    years = [str(i.year) for i in strategy.index]
    fig = go.Figure([
        go.Bar(x=years, y=strategy.to_numpy(), name="Strategy", marker=dict(color=t.STRATEGY, cornerradius=4),
               hovertemplate="%{x}: %{y:+.2%}<extra>Strategy</extra>"),
        go.Bar(x=years, y=benchmark.reindex(strategy.index).to_numpy(), name="Benchmark",
               marker=dict(color=t.BENCHMARK, cornerradius=4),
               hovertemplate="%{x}: %{y:+.2%}<extra>Benchmark</extra>"),
    ])
    fig = _base(fig, height=height, y_title="Calendar-year return", y_format=".0%")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="x unified")
    fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    return fig


def trade_distribution(trades: pd.DataFrame, height: int = 300) -> go.Figure:
    closed = trades[trades["status"] == "closed"]
    r = closed["net_return"]
    fig = go.Figure(go.Histogram(x=r, nbinsx=min(40, max(8, len(r) // 2)), name="Closed trades",
                                 marker=dict(color=t.STRATEGY, line=dict(color=t.BG, width=2)),
                                 hovertemplate="%{x}: %{y} trade(s)<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=t.MUTED, width=1))
    if len(r):
        fig.add_vline(x=float(r.mean()), line=dict(color=t.BENCHMARK, width=1.5),
                      annotation_text=f"mean {r.mean():+.2%}", annotation_font_color=t.TEXT_2)
    fig = _base(fig, height=height, y_title="Number of trades")
    fig.update_xaxes(tickformat=".0%", title_text="Net return per closed trade")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def return_distribution(returns: pd.Series, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Histogram(x=returns, nbinsx=80, marker=dict(color=t.STRATEGY, line=dict(color=t.BG, width=1)),
                                 hovertemplate="%{x}: %{y} session(s)<extra></extra>"))
    fig = _base(fig, height=height, y_title="Sessions")
    fig.update_xaxes(tickformat=".1%", title_text="Daily close-to-close return")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


# --------------------------------------------------------------------------- #
# Comparison and sensitivity
# --------------------------------------------------------------------------- #


def comparison_equity(runs, bench, height: int = 420, bench_name: str | None = "Benchmark") -> go.Figure:
    """Equity of each strategy. Pass ``bench_name=None`` when the benchmark is already
    one of the plotted strategies (buy & hold of the same asset), to avoid a duplicate line."""
    fig = go.Figure()
    if bench_name:
        fig.add_trace(_line(bench.equity.index, bench.equity, bench_name, t.MUTED, width=1.5, prefix="$"))
    for run in runs:
        color = t.STRATEGY_COLORS.get(run.strategy.key, t.MUTED)
        fig.add_trace(_line(run.net.equity.index, run.net.equity, run.strategy.name, color, prefix="$"))
    return _base(fig, height=height, y_title="Portfolio value ($, net of costs)")


def risk_return_scatter(table: pd.DataFrame, height: int = 400) -> go.Figure:
    fig = go.Figure()
    for _, row in table.iterrows():
        if row["Volatility"] is None or row["CAGR (net)"] is None or pd.isna(row["Volatility"]):
            continue
        color = t.STRATEGY_COLORS.get(row["key"], t.MUTED)
        exposure = row.get("Average exposure", row["Time invested"])
        exposure = 0.0 if exposure is None or pd.isna(exposure) else float(exposure)
        size = 10 + 26 * exposure
        fig.add_trace(go.Scatter(
            x=[row["Volatility"]], y=[row["CAGR (net)"]], mode="markers", name=row["Strategy"].split(" (")[0],
            marker=dict(size=size, color=color, line=dict(color=t.BG, width=2)),
            hovertemplate=(f"<b>{row['Strategy']}</b><br>Volatility %{{x:.1%}}<br>CAGR %{{y:+.2%}}"
                           f"<br>Average exposure {exposure:.0%}<extra></extra>"),
        ))
    fig = _base(fig, height=height, y_title="Annualised return (net)", y_format=".0%")
    vols = pd.to_numeric(table["Volatility"], errors="coerce").dropna()
    rets = pd.to_numeric(table["CAGR (net)"], errors="coerce").dropna()
    if len(vols):
        span = max(float(vols.max() - vols.min()), 0.02)
        fig.update_xaxes(range=[max(0.0, float(vols.min()) - 0.35 * span), float(vols.max()) + 0.35 * span])
    if len(rets):
        span = max(float(rets.max() - rets.min()), 0.02)
        fig.update_yaxes(range=[float(rets.min()) - 0.2 * span, float(rets.max()) + 0.3 * span])
    fig.update_xaxes(title_text="Annualised volatility", tickformat=".0%")
    fig.update_layout(hovermode="closest", showlegend=True, margin=dict(t=90))
    fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    return fig


def comparison_bars(table: pd.DataFrame, column: str, fmt: str = ".2f", height: int = 260,
                    title: str | None = None) -> go.Figure:
    values = pd.to_numeric(table[column], errors="coerce")
    colors = [t.STRATEGY_COLORS.get(k, t.MUTED) for k in table["key"]]
    names = [s.split(" (")[0] for s in table["Strategy"]]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[("n/a" if pd.isna(v) else format(v, fmt)) for v in values], textposition="outside",
        textfont=dict(color=t.TEXT_2, size=11), cliponaxis=False,
        hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>",
    ))
    fig = _base(fig, height=max(height, 90 + 30 * len(table)), title=title)
    # Pad the value axis on both sides so "outside" labels never collide with the
    # category labels (negative bars) or the plot edge (positive bars).
    finite = values.dropna()
    lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.28 * ((hi - lo) or 1.0)
    fig.update_xaxes(range=[lo - pad if lo < 0 else 0, hi + pad if hi > 0 else 0],
                     tickformat=fmt.replace(",", "") if "%" in fmt else None, zeroline=True, nticks=5,
                     tickangle=0)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.35,
                      uniformtext=dict(minsize=11, mode="show"))
    return fig


def sensitivity_heatmap(frame: pd.DataFrame, x: str, y: str, metric: str, metric_label: str,
                        current: tuple | None = None, height: int = 420, x_label: str | None = None,
                        y_label: str | None = None) -> go.Figure:
    valid = frame[frame["valid"]]
    # Keep every valid combination as a cell, even where the metric is undefined (blank), rather than letting
    # pivot_table drop all-NaN rows and columns.
    pivot = valid.pivot_table(index=y, columns=x, values=metric, aggfunc="first", dropna=False).reindex(
        index=sorted(valid[y].unique()), columns=sorted(valid[x].unique()))
    z = pivot.to_numpy(dtype=float)
    pct = metric in {"total_return", "cagr", "max_drawdown"}
    bound = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    text = [["" if np.isnan(v) else (f"{v:+.1%}" if pct else f"{v:.2f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorscale=t.DIVERGING,
        zmid=0, zmin=-bound, zmax=bound, xgap=2, ygap=2, text=text, texttemplate="%{text}",
        textfont=dict(size=11), hoverongaps=False,
        hovertemplate=f"{x}=%{{x}}, {y}=%{{y}}<br>{metric_label}: %{{z:{'.2%' if pct else '.2f'}}}<extra></extra>",
        colorbar=dict(outlinewidth=0, thickness=10, tickformat=".0%" if pct else ".1f", tickfont=dict(color=t.MUTED)),
    ))
    # Outline the current parameters only when they lie on the grid; otherwise plotly would add a category.
    on_grid = current is not None and str(current[0]) in {str(c) for c in pivot.columns} \
        and str(current[1]) in {str(i) for i in pivot.index}
    if on_grid:
        fig.add_trace(go.Scatter(x=[str(current[0])], y=[str(current[1])], mode="markers", name="Current",
                                 marker=dict(symbol="square-open", size=34, color=t.TEXT, line=dict(width=2)),
                                 hoverinfo="skip"))
    fig = _base(fig, height=height)
    fig.update_xaxes(title_text=x_label or x, type="category", showgrid=False)
    fig.update_yaxes(title_text=y_label or y, type="category", showgrid=False)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def sensitivity_line(frame: pd.DataFrame, x: str, metric: str, metric_label: str, current=None,
                     height: int = 320, x_label: str | None = None) -> go.Figure:
    valid = frame[frame["valid"]].sort_values(x)
    pct = metric in {"total_return", "cagr", "max_drawdown"}
    fig = go.Figure(go.Scatter(x=valid[x], y=valid[metric], mode="lines+markers", name=metric_label,
                               line=dict(color=t.STRATEGY, width=LINE), marker=dict(size=8, line=dict(color=t.BG, width=2)),
                               hovertemplate=f"{x}=%{{x}}<br>{metric_label}: %{{y:{'.2%' if pct else '.2f'}}}<extra></extra>"))
    if current is not None:
        fig.add_vline(x=current, line=dict(color=t.MUTED, width=1), annotation_text="current",
                      annotation_font_color=t.MUTED)
    fig = _base(fig, height=height, y_title=metric_label, y_format=".0%" if pct else None)
    fig.update_xaxes(title_text=x_label or x)
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def cost_sensitivity_chart(frame: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure([
        go.Scatter(x=frame["cost_per_side_bps"], y=frame["benchmark_total_return"], mode="lines+markers",
                   name="Benchmark", line=dict(color=t.BENCHMARK, width=LINE), marker=dict(size=7),
                   hovertemplate="%{x:.1f} bps: %{y:+.2%}<extra>Benchmark</extra>"),
        go.Scatter(x=frame["cost_per_side_bps"], y=frame["total_return"], mode="lines+markers",
                   name="Strategy", line=dict(color=t.STRATEGY, width=LINE), marker=dict(size=7),
                   hovertemplate="%{x:.1f} bps: %{y:+.2%}<extra>Strategy</extra>"),
    ])
    fig = _base(fig, height=height, y_title="Total return (net)", y_format=".0%")
    fig.update_xaxes(title_text="Cost per execution, fee + slippage (basis points)")
    return fig


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #


def walk_forward_equity(wf, height: int = 420) -> go.Figure:
    fig = go.Figure()
    for i, fold in enumerate(wf.folds):
        fig.add_vrect(x0=fold.test_start, x1=fold.test_end, fillcolor=t.SURFACE_2 if i % 2 == 0 else "rgba(0,0,0,0)",
                      line_width=0, layer="below")
        fig.add_annotation(x=fold.test_start, y=1.0, yref="paper", text=f"F{fold.number}", showarrow=False,
                           xanchor="left", yanchor="top", font=dict(color=t.MUTED, size=10))
    fig.add_trace(_line(wf.oos_benchmark.equity.index, wf.oos_benchmark.equity, "Benchmark", t.BENCHMARK, prefix="$"))
    fig.add_trace(_line(wf.oos.equity.index, wf.oos.equity, "Walk-forward strategy (out-of-sample, net)",
                        t.STRATEGY, prefix="$"))
    return _base(fig, height=height, y_title="Portfolio value ($)")


def is_vs_oos(fold_table: pd.DataFrame, height: int = 320) -> go.Figure:
    folds = [f"Fold {f}" for f in fold_table["Fold"]]
    fig = go.Figure([
        go.Bar(x=folds, y=fold_table["IS Sharpe"], name="In-sample (training) Sharpe",
               marker=dict(color=t.GROSS, cornerradius=4), hovertemplate="%{x}: %{y:.2f}<extra>In-sample</extra>"),
        go.Bar(x=folds, y=fold_table["OOS Sharpe"], name="Out-of-sample (test) Sharpe",
               marker=dict(color=t.STRATEGY, cornerradius=4), hovertemplate="%{x}: %{y:.2f}<extra>Out-of-sample</extra>"),
    ])
    fig = _base(fig, height=height, y_title="Sharpe ratio")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    return fig


# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #


def ohlc_volume(prices: pd.DataFrame, eval_start, height: int = 560) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=prices.index, open=prices["open"], high=prices["high"], low=prices["low"], close=prices["close"],
        name="OHLC", increasing=dict(line=dict(color=t.STRATEGY, width=1), fillcolor=t.STRATEGY),
        decreasing=dict(line=dict(color=t.BENCHMARK, width=1), fillcolor=t.BENCHMARK),
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=prices.index, y=prices["volume"], name="Volume", marker=dict(color=t.MUTED),
                         hovertemplate="%{y:,.0f}<extra>Volume</extra>"), row=2, col=1)
    if eval_start is not None and prices.index[0] < eval_start:
        fig.add_vrect(x0=prices.index[0], x1=eval_start, fillcolor="rgba(138,143,152,0.10)", line_width=0,
                      row="all", col=1)
        fig.add_annotation(x=prices.index[0], y=1, yref="paper", text="warm-up history (not evaluated)",
                           showarrow=False, xanchor="left", yanchor="top", font=dict(color=t.MUTED, size=11))
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _base(fig, height=height)


# --------------------------------------------------------------------------- #
# Model diagnostics and robustness
# --------------------------------------------------------------------------- #

POSITIVE = "#3987e5"
NEGATIVE = "#c84444"


def coefficient_bars(weights: dict[str, float], labels: dict[str, str], height: int = 380) -> go.Figure:
    """Standardised model coefficients: blue pushes P(up) higher, red pushes it lower."""
    items = sorted(weights.items(), key=lambda kv: abs(kv[1]))
    names = [labels.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=[POSITIVE if v >= 0 else NEGATIVE for v in values], cornerradius=4),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig = _base(fig, height=height)
    fig.update_xaxes(title_text="Coefficient (log-odds per 1 std of the feature)", zeroline=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.3)
    return fig


def weight_distribution(targets: pd.Series, height: int = 260) -> go.Figure:
    """Share of evaluation sessions spent at each target weight."""
    counts = targets.dropna().round(2).value_counts(normalize=True).sort_index()
    fig = go.Figure(go.Bar(
        x=[f"{v:.0%}" for v in counts.index], y=counts.to_numpy(), marker=dict(color=t.STRATEGY, cornerradius=4),
        hovertemplate="Target %{x}: %{y:.1%} of sessions<extra></extra>",
    ))
    fig = _base(fig, height=height, y_title="Share of sessions", y_format=".0%")
    fig.update_xaxes(title_text="Target weight", type="category")
    fig.update_layout(hovermode="closest", showlegend=False, bargap=0.3)
    return fig


def bootstrap_histogram(samples: pd.DataFrame, column: str, label: str, observed: float | None,
                        bench_column: str | None = None, bench_observed: float | None = None,
                        pct: bool = False, height: int = 340) -> go.Figure:
    fig = go.Figure()
    if bench_column and bench_column in samples:
        fig.add_trace(go.Histogram(x=samples[bench_column], name="Benchmark", nbinsx=60, opacity=0.55,
                                   marker=dict(color=t.BENCHMARK), hovertemplate="%{x}: %{y} paths<extra>Benchmark</extra>"))
    fig.add_trace(go.Histogram(x=samples[column], name="Strategy", nbinsx=60, opacity=0.75,
                               marker=dict(color=t.STRATEGY), hovertemplate="%{x}: %{y} paths<extra>Strategy</extra>"))
    marks = ((observed, "strategy actual", t.TEXT, "top right", -14),
             (bench_observed, "benchmark actual", t.BENCHMARK, "top left", -34))
    for value, name, color, position, shift in marks:
        if value is not None and np.isfinite(value):
            fig.add_vline(x=value, line=dict(color=color, width=1.5), annotation_position=position,
                          annotation=dict(text=name, font=dict(color=t.TEXT_2, size=11), yshift=shift))
    fig = _base(fig, height=height, y_title="Bootstrap paths")
    fig.update_xaxes(title_text=label, tickformat=".0%" if pct else None)
    fig.update_layout(barmode="overlay", hovermode="closest")
    return fig


def null_distribution(null: np.ndarray, observed: float | None, label: str, height: int = 320) -> go.Figure:
    """Histogram of a statistic under a null (e.g. shifted timing) with the observed value marked."""
    values = np.asarray(null, dtype="float64")
    fig = go.Figure(go.Histogram(x=values[np.isfinite(values)], nbinsx=40, name="Shifted timing",
                                 marker=dict(color=t.MUTED), hovertemplate="%{x:.2f}: %{y} shifts<extra></extra>"))
    if observed is not None and np.isfinite(observed):
        fig.add_vline(x=observed, line=dict(color=t.STRATEGY, width=2), annotation_position="top right",
                      annotation=dict(text="actual timing", font=dict(color=t.TEXT_2, size=11)))
    fig = _base(fig, height=height, y_title="Shifted paths")
    fig.update_xaxes(title_text=label)
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def regime_bars(table: pd.DataFrame, height: int = 320) -> go.Figure:
    labels = [f"{d}: {r}" for d, r in zip(table["Dimension"], table["Regime"])]
    fig = go.Figure([
        go.Bar(x=labels, y=table["Strategy return (ann.)"], name="Strategy", marker=dict(color=t.STRATEGY, cornerradius=4),
               hovertemplate="%{x}: %{y:+.1%}<extra>Strategy</extra>"),
        go.Bar(x=labels, y=table["Benchmark return (ann.)"], name="Benchmark",
               marker=dict(color=t.BENCHMARK, cornerradius=4), hovertemplate="%{x}: %{y:+.1%}<extra>Benchmark</extra>"),
    ])
    fig = _base(fig, height=height, y_title="Annualised return (arithmetic)", y_format=".0%")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    return fig
