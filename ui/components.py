"""Reusable presentation components."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st

from quantlab.config import APP_VERSION, METHODOLOGY_VERSION
from quantlab.data import CACHE_STATUS_LABELS, CacheStatus
from quantlab.metrics import Metric, MetricReport, format_value
from quantlab.validation import QualityReport
from ui.theme import PLOTLY_CONFIG

GLOSSARY = {
    "total_return": "Growth of the portfolio over the whole period, after all fees and slippage.",
    "cagr": "Compound annual growth rate: the constant yearly return that would produce the same "
            "total growth. Uses 252 trading sessions per year.",
    "volatility": "How much daily returns fluctuate, scaled to a year (std × √252). Higher = bumpier ride.",
    "sharpe": "Return earned per unit of total volatility, above the risk-free rate. Around 0.5 is typical "
              "for equity indices; very high values on a backtest are a red flag for overfitting.",
    "sortino": "Like Sharpe, but only penalises downside volatility (returns below the risk-free rate).",
    "max_drawdown": "The largest fall from a previous peak, in %. The worst loss an investor who bought "
                    "at the top would have suffered.",
    "calmar": "Annualised return divided by the absolute maximum drawdown.",
    "time_invested": "Share of sessions the strategy held the asset rather than cash.",
    "turnover": "How much is traded per year relative to portfolio size. 100% ≈ the whole portfolio "
                "traded once per year (buys and sells both count).",
    "excess_return": "Strategy total return minus the benchmark's total return, in percentage points.",
    "win_rate": "Share of completed trades that made money after costs. Says nothing about how much.",
    "beta": "Sensitivity to the benchmark: 1.0 moves one-for-one; 0.5 moves half as much.",
    "information_ratio": "Annualised excess return over the benchmark divided by tracking error.",
    "tracking_error": "Annualised volatility of the difference between strategy and benchmark returns.",
    "fills": "Individual buy or sell executions.",
    "cost_drag": "How much fees and slippage reduced the total return.",
}


SHORT_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "volatility": "Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar",
    "time_invested": "Time invested",
    "avg_exposure": "Avg exposure",
    "gross_total_return": "Gross return",
    "cost_drag": "Cost drag",
    "total_fees": "Fees",
    "total_slippage": "Slippage",
    "turnover": "Turnover (ann.)",
    "fills": "Fills",
    "round_trips": "Round trips",
    "win_rate": "Win rate",
    "avg_trade_return": "Avg trade",
    "avg_winner": "Avg winner",
    "avg_loser": "Avg loser",
    "profit_factor": "Profit factor",
    "benchmark_total_return": "Benchmark return",
    "excess_return": "Excess return",
}


def _esc(value: object) -> str:
    """HTML-escape text and encode '$', so amounts are never typeset as LaTeX by Streamlit's Markdown."""
    return html.escape(str(value)).replace("$", "&#36;")


def kv_table(items: dict[str, object]) -> None:
    """Two-column key/value list whose values wrap instead of being truncated."""
    cells = "".join(
        f'<div class="k">{_esc(str(k))}</div><div class="v">{_esc(str(v))}</div>'
        for k, v in items.items()
    )
    st.markdown(f'<div class="qlp-kv">{cells}</div>', unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="qlp-eyebrow">{_esc(eyebrow)}</div>'
        f'<h1 class="qlp-title">{_esc(title)}</h1>'
        f'<div class="qlp-subtitle">{_esc(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="qlp-section">{_esc(title)}</div>', unsafe_allow_html=True)
    if caption:
        # '$' as an HTML entity, so two amounts on one line are never typeset as LaTeX.
        text = _esc(caption)
        st.markdown(f'<div class="qlp-caption">{text}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    icons = {"good": "●", "warn": "▲", "bad": "■", "info": "◆"}
    return f'<span class="qlp-pill {kind}">{icons.get(kind, "")} {_esc(text)}</span>'


def cache_pill(status: CacheStatus) -> str:
    kind = {CacheStatus.DOWNLOADED: "good", CacheStatus.CACHED: "info",
            CacheStatus.STALE_CACHE: "warn", CacheStatus.UPLOADED: "info", CacheStatus.SYNTHETIC: "warn"}[status]
    return pill(CACHE_STATUS_LABELS[status], kind)


def context_strip(data, spec=None, label: str | None = None) -> None:
    """One-line research context: what, when, where from, and how fresh."""
    ds = data.asset
    items = [
        ("Asset", ds.ticker),
        ("Evaluation", f"{data.eval_start:%Y-%m-%d} → {data.eval_end:%Y-%m-%d} ({len(data.eval_prices):,} sessions)"),
        ("Benchmark", data.benchmark_label),
        ("Source", ds.source_name),
        ("Prices", ds.price_convention.split(" (")[0]),
        ("Last observation", f"{ds.last_date:%Y-%m-%d}"),
    ]
    if label:
        items.insert(1, ("Strategy", label))
    body = "".join(f'<div><span class="k">{_esc(k)}</span><b>{_esc(str(v))}</b></div>' for k, v in items)
    body += f"<div>{cache_pill(ds.cache_status)}</div>"  # says SYNTHETIC for simulated data
    st.markdown(f'<div class="qlp-strip">{body}</div>', unsafe_allow_html=True)


def is_synthetic(data) -> bool:
    return data.asset.cache_status is CacheStatus.SYNTHETIC


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="qlp-empty"><h3>{_esc(title)}</h3><p>{body}</p></div>', unsafe_allow_html=True
    )


def demo_button(key: str) -> None:
    from ui.state import start_synthetic_demo

    st.button("Explore with synthetic demo data", type="primary", key=key, on_click=start_synthetic_demo,
              icon=":material/science:",
              help="Loads simulated prices with a known data-generating process (clearly labelled, not market "
                   "data), so every page can be explored without a network connection.")


def require_data():
    """Return usable ResearchData or render the appropriate empty/error state and stop."""
    from ui.state import current_data

    error = st.session_state.get("data_error")
    data = current_data()
    if error:
        st.error(f"**Market data unavailable.** {error['message']}", icon=":material/cloud_off:")
        if error.get("detail"):
            with st.expander("Technical detail"):
                st.code(error["detail"], language=None)
        if data is not None:
            st.caption("Showing the previously loaded dataset below.")
    if data is None:
        empty_state(
            "No market data loaded",
            "Choose a ticker and date range in the sidebar and press <b>Load data</b>. "
            "If Yahoo Finance is unreachable from your network, switch the data source to "
            "<b>CSV file</b> and provide daily OHLCV data (Date, Open, High, Low, Close, Volume), "
            "or explore the app with simulated prices.",
        )
        demo_button("demo_empty")
        st.stop()
    if not data.is_usable:
        problems = [i.message for i in data.asset.quality.errors]
        if data.benchmark_dataset is not None:
            problems += [f"Benchmark: {i.message}" for i in data.benchmark_dataset.quality.errors]
        st.error(
            "**The loaded data failed validation, so no backtest can be run.** "
            + " ".join(problems or ["Too few observations in the evaluation window."]),
            icon=":material/block:",
        )
        st.caption("Open the Data Explorer page for the full quality report.")
        st.stop()
    return data


def require_spec():
    from ui.state import current_spec

    spec = current_spec()
    if spec is None:
        st.warning("Fix the strategy parameters in the sidebar to continue.", icon=":material/tune:")
        st.stop()
    return spec


def metric_card(column, metric: Metric | None, label: str | None = None, delta: str | None = None,
                help_key: str | None = None) -> None:
    if metric is None:
        column.metric(label or "—", "n/a", border=True)
        return
    value = metric.formatted()
    shown_label = label or SHORT_LABELS.get(metric.key, metric.label)
    if not metric.reliable:
        shown_label += " *"  # on the label, which never truncates, rather than the value
    help_text = GLOSSARY.get(help_key or metric.key, metric.definition)
    if metric.note:
        help_text += f"\n\n⚠ {metric.note}"
    column.metric(shown_label, value, delta=delta,
                  delta_color="off", delta_arrow="off", help=f"**{metric.label}.** {help_text}", border=True)


def kpi_row(report: MetricReport, keys: Iterable[str], labels: dict[str, str] | None = None,
            deltas: dict[str, str] | None = None) -> None:
    keys = list(keys)
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        metric_card(col, report.get(key), (labels or {}).get(key), (deltas or {}).get(key))


def unreliable_footnote(report: MetricReport, keys: Iterable[str]) -> None:
    flagged = [report[k] for k in keys if report.get(k) is not None and not report[k].reliable]
    if flagged:
        st.caption("\\* " + " · ".join(f"{m.label}: {m.note}" for m in flagged if m.note))


def metrics_table(report: MetricReport, bench_report: MetricReport | None = None,
                  keys: Iterable[str] | None = None) -> pd.DataFrame:
    rows = []
    for key, metric in report.metrics.items():
        if keys is not None and key not in keys:
            continue
        row = {"Metric": metric.label, "Strategy": metric.formatted() + ("" if metric.reliable else " *")}
        if bench_report is not None:
            other = bench_report.get(key)
            row["Benchmark"] = other.formatted() if other is not None else ""
        row["Note"] = metric.note
        row["Definition"] = metric.definition
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_html_table(report: MetricReport, bench_report: MetricReport | None = None,
                       keys: Iterable[str] | None = None) -> None:
    """Metric | Strategy | Benchmark | Definition, with wrapping text and caveats inline."""
    head = "<th>Metric</th><th style='text-align:right'>Strategy</th>"
    if bench_report is not None:
        head += "<th style='text-align:right'>Benchmark</th>"
    head += "<th>Definition</th>"
    body = []
    for key in keys or report.metrics:
        metric = report.get(key)
        if metric is None:
            continue
        value = metric.formatted() + ("" if metric.reliable else " *")
        row = f"<td>{_esc(metric.label)}</td><td class='num'>{_esc(value)}</td>"
        if bench_report is not None:
            other = bench_report.get(key)
            row += f"<td class='num'>{_esc(other.formatted()) if other is not None else '—'}</td>"
        note = f"<span class='note'>⚠ {_esc(metric.note)}</span>" if metric.note else ""
        row += f"<td class='def'>{_esc(metric.definition)}{note}</td>"
        body.append(f"<tr>{row}</tr>")
    if body:
        st.markdown(f"<table class='qlp-table'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>",
                    unsafe_allow_html=True)


def show_table(frame: pd.DataFrame, height: int | str = "auto", **kwargs) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", height=height, **kwargs)


def html_table(frame: pd.DataFrame, code_columns: Iterable[str] = ()) -> None:
    """A wrapping table for long text, which ``st.dataframe`` would truncate.

    A list or tuple cell renders one item per line; ``code_columns`` render in monospace.
    """
    code = set(code_columns)

    def cell(column, value) -> str:
        parts = [_esc(v) for v in (value if isinstance(value, list | tuple) else [value])]
        return "<br>".join(f"<code>{p}</code>" if column in code else p for p in parts)

    head = "".join(f"<th>{_esc(c)}</th>" for c in frame.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(c, v)}</td>" for c, v in zip(frame.columns, row, strict=True)) + "</tr>"
        for row in frame.itertuples(index=False)
    )
    st.markdown(f"<div class='qlp-scroll'><table class='qlp-table'><thead><tr>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key, width="stretch")


def quality_overview(report: QualityReport) -> None:
    n_err, n_warn, n_info = len(report.errors), len(report.warnings), len(report.infos)
    pills = []
    if n_err:
        pills.append(pill(f"{n_err} blocking error(s)", "bad"))
    if n_warn:
        pills.append(pill(f"{n_warn} warning(s)", "warn"))
    if n_info:
        pills.append(pill(f"{n_info} note(s)", "info"))
    if not pills:
        pills.append(pill("All quality checks passed", "good"))
    st.markdown(" ".join(pills), unsafe_allow_html=True)


def quality_issue_list(report: QualityReport) -> None:
    """Every quality issue with its severity, full explanation and example dates (wrapping text)."""
    kinds = {"error": ("bad", "ERROR"), "warning": ("warn", "WARNING"), "info": ("info", "NOTE")}
    order = {"error": 0, "warning": 1, "info": 2}
    rows = []
    for issue in sorted(report.issues, key=lambda i: order[i.severity.value]):
        kind, label = kinds[issue.severity.value]
        dates = ", ".join(issue.dates[:12]) + (f" … (+{len(issue.dates) - 12} more)" if len(issue.dates) > 12 else "")
        count = f" · {issue.count:,} row(s)" if issue.count else ""
        rows.append(
            f'<div class="k">{pill(label, kind)}<br><code>{_esc(issue.code)}</code></div>'
            f'<div class="v">{_esc(issue.message)}<span style="color:#8a8f98">{_esc(count)}</span>'
            + (f'<br><span style="color:#8a8f98;font-size:0.78rem">{_esc(dates)}</span>' if dates else "")
            + "</div>"
        )
    if rows:
        st.markdown(f'<div class="qlp-kv" style="grid-template-columns: 11rem 1fr">{"".join(rows)}</div>',
                    unsafe_allow_html=True)


def integrity_panel(run, data) -> None:
    """Compact pre-flight checklist shown before results are interpreted."""
    from quantlab.backtest import warmup_shortfall

    net = run.net
    items = []
    if net.is_valid:
        items.append(pill("Accounting reconciled (7 checks)", "good"))
    else:
        items.append(pill("Accounting FAILED to reconcile", "bad"))
    q = data.asset.quality
    bq = data.benchmark_dataset.quality if data.benchmark_dataset is not None else None
    if q.warnings:
        items.append(pill(f"Asset data: {len(q.warnings)} warning(s)", "warn"))
    if bq is not None and bq.warnings:
        items.append(pill(f"Benchmark data: {len(bq.warnings)} warning(s)", "warn"))
    if not q.warnings and not (bq is not None and bq.warnings):
        items.append(pill("Data checks passed", "good"))
    if len(data.dropped_dates):
        items.append(pill(f"{len(data.dropped_dates)} date(s) dropped to align the benchmark", "info"))
    if is_synthetic(data):
        items.append(pill("Synthetic data: results describe a simulated market", "warn"))
    shortfall = warmup_shortfall(net)
    if shortfall > 1:
        items.append(pill(f"Warm-up used {shortfall - 1} evaluation sessions", "warn"))
    needed, available = run.strategy.history_sessions, data.warmup_sessions_available
    if needed is None:
        items.append(pill("Model refits are anchored to the data start", "info"))
    elif available < needed:
        items.append(pill(f"Short history: {available} of {needed} sessions; results may depend on the start date",
                          "warn"))
    if net.unexecuted_order:
        items.append(pill(f"Unexecuted {net.unexecuted_order['side']} signal on final session", "info"))
    if len(net.rejected_orders):
        items.append(pill(f"{len(net.rejected_orders)} rejected order(s)", "warn"))
    st.markdown(" ".join(items), unsafe_allow_html=True)
    if not net.is_valid:
        st.error(
            "The portfolio accounting could not be validated, so these results must not be relied on: "
            + " ".join(net.reconciliation_issues),
            icon=":material/gpp_bad:",
        )


_EVIDENCE_STATUS = {"pass": ("PASS", "good"), "caution": ("CAUTION", "warn"), "fail": ("FAIL", "bad"),
                    "not_run": ("NOT RUN", "info")}


def evidence_table(items) -> None:
    """One row per evidence check: status, question, value, threshold and where to run it."""
    rows = []
    for item in items:
        label, kind = _EVIDENCE_STATUS[item.status]
        rows.append(
            f"<tr><td>{pill(label, kind)}</td><td>{_esc(item.question)}"
            f"<span class='note' style='color:#8a8f98'>{_esc(item.explanation)}</span></td>"
            f"<td class='num'>{_esc(item.value)}</td><td>{_esc(item.threshold)}</td><td>{_esc(item.where)}</td></tr>"
        )
    st.markdown(
        "<div class='qlp-scroll'><table class='qlp-table'><thead><tr><th>Status</th><th>Question</th>"
        "<th style='text-align:right'>Result</th><th>Threshold</th><th>Where</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<div class="qlp-disclaimer"><b>Research demonstration, not investment advice.</b> '
        "Results are hypothetical backtests on historical end-of-day data, which may contain errors or be "
        "revised by the provider. Nothing here is live market data. Backtests omit taxes, borrowing "
        "constraints, market impact beyond a fixed slippage estimate, and the behavioural difficulty of "
        "following a strategy in real time. Historical performance does not guarantee future results. "
        "This application cannot place orders and connects to no broker. "
        f"Quant Lab Pro v{APP_VERSION} · methodology v{METHODOLOGY_VERSION}.</div>",
        unsafe_allow_html=True,
    )


def fmt(value, kind: str) -> str:
    return format_value(value, kind)
