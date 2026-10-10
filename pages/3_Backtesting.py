"""Backtesting Engine — assumptions, executions, ledger and accounting audit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantlab.reporting import (
    build_manifest,
    equity_curve_csv,
    ledger_csv,
    manifest_to_json,
    metrics_csv,
    prices_csv,
    trades_csv,
)
from ui import charts
from ui import components as ui
from ui.state import active_run

data = ui.require_data()
spec = ui.require_spec()
run, bench = active_run(data, spec)
net, cfg = run.net, spec.config

ui.page_header(
    "Backtesting engine",
    "Executions & accounting",
    "Every trade below can be traced from a signal to a fill price, a fee and a change in cash and shares. "
    "If any accounting identity fails, the results are flagged as unreliable.",
)
ui.context_strip(data, label=net.strategy_label)
ui.integrity_panel(run, data)

# ---- assumptions ---------------------------------------------------------------
ui.section("Execution model")
c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    with st.container(border=True):
        st.markdown(
            f"""
1. **Signal** — after the close of session *t*, the strategy sets a target weight between 0% and 100%
   using prices up to and including that close.
2. **Order** — an order is created to enter from cash, to exit fully to cash, or — for partial
   positions — when the held weight drifts more than {cfg.rebalance_threshold:.0%} from the target.
   A target that stays the same creates no new orders.
3. **Fill** — the order executes at the **open of session *t + 1*** (the next row of data). Overnight gaps
   are fully reflected.
4. **Price** — buys fill at open × (1 + {cfg.slippage_rate:.4%}), sells at open × (1 − {cfg.slippage_rate:.4%}).
5. **Fee** — {cfg.fee_rate:.4%} of the actual traded notional (quantity × fill price), on buys and sells.
6. **Sizing** — {"fractional shares" if cfg.fractional_shares else "whole shares only (residual cash stays uninvested)"}:
   buys acquire the value needed to reach the target weight but never spend more than the cash available
   including the fee (a 100% target spends everything); sells reduce holdings to the target, and a 0% target
   sells the whole position.
7. **Cash** — {f"earns {cfg.cash_interest_rate:.2%} a year, accrued each session" if cfg.cash_interest_rate > 0
               else "earns no interest"}. Equity = cash + shares × close, every session.
8. **End of sample** — an open position is marked to market at the final close; a signal on the final
   session cannot execute and is reported separately.
"""
        )
with c2:
    ui.kv_table(cfg.describe())
    st.caption("Limitations: no volume constraints or market impact beyond fixed slippage, "
               "no short selling, no leverage, no taxes, no dividends paid as cash (they are inside "
               "total-return prices when that convention is selected).")

# ---- price and trades ----------------------------------------------------------
ui.section("Price, indicators and fills",
           "Triangles mark actual fills at their execution date and execution price (after slippage). "
           "Hover a marker to see the signal date and the reason.")
ui.plot(charts.price_with_trades(net, run.strategy), key="bt_price")
ui.section("Position exposure", "Share of portfolio value held in the asset at each close.")
ui.plot(charts.exposure(net), key="bt_exposure")

# ---- ledger --------------------------------------------------------------------
ui.section("Trade ledger", f"{len(net.ledger)} fills · fees ${net.ledger['fee'].sum():,.2f} · "
                            f"slippage ${net.ledger['slippage_cost'].sum():,.2f}")
if net.ledger.empty:
    st.info("No trades were executed in this evaluation window.", icon=":material/info:")
else:
    money = st.column_config.NumberColumn(format="dollar")
    ui.show_table(
        net.ledger,
        height=min(420, 38 + 35 * len(net.ledger)),
        column_config={
            "fill_id": st.column_config.NumberColumn("#", format="%d"),
            "signal_date": st.column_config.DateColumn("Signal date"),
            "execution_date": st.column_config.DateColumn("Execution date"),
            "side": "Side",
            "target_weight": st.column_config.NumberColumn("Target weight", format="percent"),
            "quantity": st.column_config.NumberColumn("Quantity", format="%.4f"),
            "reference_price": st.column_config.NumberColumn("Reference (open)", format="$%.4f"),
            "execution_price": st.column_config.NumberColumn("Fill price", format="$%.4f"),
            "gross_notional": st.column_config.NumberColumn("Gross notional", format="dollar"),
            "fee": st.column_config.NumberColumn("Fee", format="$%.2f"),
            "slippage_cost": st.column_config.NumberColumn("Slippage cost", format="$%.2f"),
            "net_cash_flow": st.column_config.NumberColumn("Net cash flow", format="dollar"),
            "cash_before": money,
            "cash_after": money,
            "shares_before": st.column_config.NumberColumn(format="%.4f"),
            "shares_after": st.column_config.NumberColumn(format="%.4f"),
            "reason": st.column_config.TextColumn("Reason", width="large"),
        },
    )

    with st.expander("Trace a single fill step by step", icon=":material/calculate:"):
        fill_id = st.selectbox("Fill", net.ledger["fill_id"], format_func=lambda i: (
            f"#{i} · {net.ledger.loc[net.ledger['fill_id'] == i, 'side'].iloc[0]} on "
            f"{net.ledger.loc[net.ledger['fill_id'] == i, 'execution_date'].iloc[0]:%Y-%m-%d}"), key="bt_trace")
        f = net.ledger.loc[net.ledger["fill_id"] == fill_id].iloc[0]
        sign = "+" if f["side"] == "BUY" else "−"
        prev_close = net.daily["close"].shift(1).loc[f["execution_date"]]
        st.markdown(
            f"""
| Step | Calculation | Value |
|---|---|---|
| Signal | close of {f['signal_date']:%Y-%m-%d} (close ${prev_close:,.4f}) | {f['reason']} |
| Reference price | open of {f['execution_date']:%Y-%m-%d} | ${f['reference_price']:,.4f} |
| Fill price | {f['reference_price']:,.4f} × (1 {sign} {cfg.slippage_rate:.4%}) | ${f['execution_price']:,.4f} |
| Target weight | set at the signal close | {f['target_weight']:.1%} |
| Quantity | {"value needed to reach the target, capped at cash ÷ (fill price × (1 + fee rate))" if f['side'] == 'BUY' else ("all shares held" if f['target_weight'] == 0 else "holdings above the target ÷ reference price")} | {f['quantity']:,.6f} |
| Gross notional | quantity × fill price | ${f['gross_notional']:,.2f} |
| Fee | {cfg.fee_rate:.4%} × notional | ${f['fee']:,.2f} |
| Slippage cost | quantity × \\|fill − reference\\| | ${f['slippage_cost']:,.2f} |
| Cash | {f['cash_before']:,.2f} {'−' if f['side'] == 'BUY' else '+'} {abs(f['net_cash_flow']):,.2f} | ${f['cash_after']:,.2f} |
| Shares | {f['shares_before']:,.6f} → {f['shares_after']:,.6f} | |
"""
        )

ui.section("Round-trip trades",
           "Each position episode from cash back to cash, including any partial rebalances in between. "
           "Returns include all fees and slippage; prices are volume-weighted.")
if net.trades.empty:
    st.caption("No round trips.")
else:
    shown = net.trades.copy()
    for col in ("net_return", "gross_return"):
        shown[col] = shown[col] * 100
    ui.show_table(
        shown,
        height=min(420, 38 + 35 * len(shown)),
        column_config={
            "trade_id": st.column_config.NumberColumn("#", format="%d"),
            "fills": st.column_config.NumberColumn("Fills", format="%d"),
            "entry_signal_date": st.column_config.DateColumn("Entry signal"),
            "entry_date": st.column_config.DateColumn("Entry"),
            "exit_signal_date": st.column_config.DateColumn("Exit signal"),
            "exit_date": st.column_config.DateColumn("Exit"),
            "net_return": st.column_config.NumberColumn("Net return", format="%.2f%%"),
            "gross_return": st.column_config.NumberColumn("Price move", format="%.2f%%"),
            "pnl": st.column_config.NumberColumn("P&L", format="dollar"),
            "entry_cost": st.column_config.NumberColumn(format="dollar"),
            "exit_value": st.column_config.NumberColumn(format="dollar"),
            "fees": st.column_config.NumberColumn(format="$%.2f"),
            "slippage_cost": st.column_config.NumberColumn(format="$%.2f"),
            "entry_price": st.column_config.NumberColumn(format="$%.4f"),
            "exit_price": st.column_config.NumberColumn(format="$%.4f"),
            "entry_reference_price": st.column_config.NumberColumn(format="$%.4f"),
            "exit_reference_price": st.column_config.NumberColumn(format="$%.4f"),
            "quantity": st.column_config.NumberColumn(format="%.4f"),
        },
    )

if net.unexecuted_order:
    u = net.unexecuted_order
    st.info(f"**Unexecuted signal:** {u['side']} on {u['signal_date']:%Y-%m-%d} — {u['reason']}. {u['note']}",
            icon=":material/schedule:")
if len(net.rejected_orders):
    st.warning("Rejected orders (no new order is sent until the signal changes):", icon=":material/block:")
    ui.show_table(net.rejected_orders)

# ---- accounting audit ------------------------------------------------------------
ui.section("Accounting reconciliation")
audit = pd.DataFrame(net.reconciliation)
audit["Status"] = audit["passed"].map({True: "✅ Passed", False: "❌ Failed"})
ui.show_table(audit[["check", "Status", "detail"]].rename(columns={"check": "Identity checked", "detail": "Detail"}))
if not net.is_valid:
    st.error("At least one accounting identity failed. Do not rely on these results.", icon=":material/error:")

with st.expander("Daily accounting record", icon=":material/table_rows:"):
    ui.show_table(net.daily.reset_index(), height=400,
                  column_config={"date": st.column_config.DateColumn("Date"),
                                 "exposure": st.column_config.NumberColumn(format="percent")})

# ---- exports ---------------------------------------------------------------------
ui.section("Export & reproduce",
           "The run manifest records every input (data fingerprint, parameters, costs, methodology version). "
           "Re-run it with `python scripts/reproduce_backtest.py --manifest run_manifest.json --prices prices.csv`.")
manifest = build_manifest(
    ticker=data.asset.ticker, benchmark=data.benchmark_label, eval_start=data.eval_start, eval_end=data.eval_end,
    data_start=data.prices.index[0], provider_key=data.asset.provider_key, source_name=data.asset.source_name,
    adjustment=data.asset.adjustment, fingerprint=data.fingerprint, first_observation=data.prices.index[0],
    last_observation=data.prices.index[-1], strategy_key=spec.strategy_key, params=spec.params, config=cfg,
    report=run.report, benchmark_fingerprint=data.bench_fingerprint,
)
slug = f"{data.asset.ticker}_{spec.strategy_key}_{data.eval_start:%Y%m%d}_{data.eval_end:%Y%m%d}"
cols = st.columns(3)
downloads = [
    ("Prices used (CSV)", prices_csv(data.prices), f"{slug}_prices.csv", "text/csv"),
    ("Equity curve (CSV)", equity_curve_csv(net, bench, run.gross), f"{slug}_equity.csv", "text/csv"),
    ("Trade ledger (CSV)", ledger_csv(net), f"{slug}_ledger.csv", "text/csv"),
    ("Round trips (CSV)", trades_csv(net), f"{slug}_trades.csv", "text/csv"),
    ("Summary metrics (CSV)", metrics_csv(run.report), f"{slug}_metrics.csv", "text/csv"),
    ("Run manifest (JSON)", manifest_to_json(manifest), f"{slug}_manifest.json", "application/json"),
]
for i, (label, payload, name, mime) in enumerate(downloads):
    cols[i % 3].download_button(label, payload, name, mime, icon=":material/download:", width="stretch",
                                on_click="ignore", key=f"dl_{i}")

ui.disclaimer()
