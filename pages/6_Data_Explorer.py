"""Data Explorer — provenance, quality checks and the raw observations."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from quantlab.calendar import session_gaps
from quantlab.data import ADJUSTMENT_LABELS, CACHE_STATUS_LABELS, CSVProvider, YahooFinanceProvider
from quantlab.reporting import prices_csv
from ui import charts
from ui import components as ui
from ui import theme as t
from ui.state import current_data

ui.page_header(
    "Data explorer",
    "Market data & quality",
    "Where the prices came from, how they were adjusted, and every check they passed or failed. Prices are "
    "never filled in or invented: unusable rows are removed and listed here.",
)

data = current_data()
error = st.session_state.get("data_error")
if error:
    st.error(f"**Market data unavailable.** {error['message']}", icon=":material/cloud_off:")
    if error.get("detail"):
        with st.expander("Technical detail"):
            st.code(error["detail"], language=None)
if data is None:
    ui.empty_state("No market data loaded",
                   "Load a ticker from the sidebar, or switch the data source to <b>CSV file</b>.")
    st.stop()

ds = data.asset
ui.section("Provenance")
c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    info = ds.describe()
    info["Evaluation window"] = f"{data.eval_start:%Y-%m-%d} → {data.eval_end:%Y-%m-%d}"
    info["Warm-up history before evaluation"] = f"{data.warmup_sessions_available:,} sessions"
    info["Benchmark"] = data.benchmark_label
    ui.kv_table(info)
    for note in ds.notes + data.notes:
        st.caption(note)
with c2:
    with st.container(border=True):
        st.markdown(ui.cache_pill(ds.cache_status), unsafe_allow_html=True)
        st.markdown(f"**{ds.source_name}**")
        st.caption(YahooFinanceProvider.description if ds.provider_key == "yahoo" else CSVProvider.description)
        st.markdown(f"**Price convention:** {ADJUSTMENT_LABELS.get(ds.adjustment, ds.adjustment)}")
        if ds.adjustment == "total_return":
            st.caption("Open, high, low and close are multiplied by Adj Close ÷ Close for each session, so "
                       "dividends are reinvested and splits are neutralised. Historical adjusted prices change "
                       "whenever a new dividend is paid; returns do not.")
        elif ds.adjustment == "price_only":
            st.caption("Split-adjusted prices without dividends: returns exclude income, which understates "
                       "buy-and-hold for dividend payers.")
        else:
            st.caption("Prices are used exactly as uploaded. Make sure you know whether they are adjusted for "
                       "splits and dividends; an unadjusted split looks like a crash.")
        st.caption("This is delayed historical end-of-day data. It is not live and must not be used as a quote.")
        if ds.cache_status.value == "stale_cache":
            st.warning(f"Provider unavailable: {ds.provider_error}", icon=":material/history:")

# ---- quality -----------------------------------------------------------------------
ui.section("Quality report", "Errors block backtesting; warnings mean rows were removed or look suspicious; "
                             "notes are context worth knowing.")
ui.quality_overview(ds.quality)
ui.quality_issue_list(ds.quality)
if len(ds.quality.removed):
    with st.expander(f"{len(ds.quality.removed)} removed row(s)"):
        ui.show_table(ds.quality.removed, column_config={"date": st.column_config.DateColumn("Date")})
st.caption(f"Rows received: {ds.quality.raw_rows:,} · usable rows: {ds.quality.clean_rows:,}")

if data.benchmark_dataset is not None:
    b = data.benchmark_dataset
    ui.section(f"Benchmark data · {b.ticker}")
    ui.quality_overview(b.quality)
    ui.quality_issue_list(b.quality)
    if len(data.dropped_dates):
        st.caption(f"{len(data.dropped_dates)} date(s) dropped to align calendars: "
                   + ", ".join(d.strftime("%Y-%m-%d") for d in data.dropped_dates[:15])
                   + (" …" if len(data.dropped_dates) > 15 else ""))

if not ds.is_usable:
    st.stop()

# ---- charts --------------------------------------------------------------------------
prices = data.prices
ui.section("Price history", "Shaded region = warm-up history, used only to initialise indicators.")
ui.plot(charts.ohlc_volume(prices, data.eval_start), key="de_ohlc")

c1, c2 = st.columns(2, gap="medium")
returns = data.eval_prices["close"].pct_change().dropna()
with c1:
    ui.section("Daily return distribution (evaluation window)")
    ui.plot(charts.return_distribution(returns), key="de_ret")
with c2:
    ui.section("Return statistics")
    stats = {
        "Sessions": f"{len(returns):,}",
        "Mean daily return": f"{returns.mean():+.4%}",
        "Std of daily returns": f"{returns.std():.4%}",
        "Skewness": f"{returns.skew():.2f}",
        "Excess kurtosis": f"{returns.kurt():.2f}",
        "Largest daily gain": f"{returns.max():+.2%} ({returns.idxmax():%Y-%m-%d})",
        "Largest daily loss": f"{returns.min():+.2%} ({returns.idxmin():%Y-%m-%d})",
        "Mean overnight gap |open/prev close − 1|":
            f"{(data.eval_prices['open'] / data.eval_prices['close'].shift(1) - 1).abs().mean():.3%}",
    }
    ui.kv_table(stats)
    st.caption("Excess kurtosis above 0 means fatter tails than a normal distribution: extreme days happen more "
               "often than a bell curve suggests.")

if "adjustment_factor" in prices:
    ui.section("Total-return adjustment factor",
               "Adj Close ÷ Close. Steps occur on dividend ex-dates; a flat line means no dividends.")
    fig = go.Figure(go.Scatter(x=prices.index, y=prices["adjustment_factor"], mode="lines",
                               line=dict(color=t.GROSS, width=2), hovertemplate="%{y:.4f}<extra></extra>"))
    fig.update_layout(template=t.PLOTLY_TEMPLATE, height=240, showlegend=False)
    ui.plot(fig, key="de_adj")

# ---- calendar ---------------------------------------------------------------------------
ui.section("Trading-calendar check",
           "Observed sessions compared with an approximate NYSE calendar (weekends, exchange holidays and "
           "notable closures). Weekdays without data are not assumed to be missing — they are listed for review.")
missing, unexpected = session_gaps(prices.index)
c1, c2, c3 = st.columns(3)
c1.metric("Observed sessions", f"{len(prices):,}", border=True)
c2.metric("Expected but absent", f"{len(missing):,}", border=True)
c3.metric("Present but unexpected", f"{len(unexpected):,}", border=True)
if len(missing):
    st.caption("Absent weekdays: " + ", ".join(d.strftime("%Y-%m-%d") for d in missing[:30])
               + (" …" if len(missing) > 30 else ""))

# ---- raw table -------------------------------------------------------------------------
ui.section("Observations")
view = st.segmented_control("Show", ["Prices used in backtests", "Raw provider data"],
                            default="Prices used in backtests", key="de_view") or "Prices used in backtests"
frame = prices if view == "Prices used in backtests" else ds.raw
ui.show_table(frame.reset_index(), height=380, column_config={"date": st.column_config.DateColumn("Date")})
st.download_button("Download prices used (CSV)", prices_csv(prices),
                   f"{ds.ticker}_{prices.index[0]:%Y%m%d}_{prices.index[-1]:%Y%m%d}_prices.csv", "text/csv",
                   icon=":material/download:", on_click="ignore")
st.caption("Cache status legend: " + " · ".join(CACHE_STATUS_LABELS.values()))

ui.disclaimer()
