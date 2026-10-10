"""Deterministic, long-only, unleveraged daily backtesting engine.

Life of one session ``t`` in the simulation:

1. **Open** — if an order was generated at the previous close, it fills at this
   session's open (see :mod:`quantlab.execution` for prices and costs).
2. **Close** — the portfolio is valued at the close:
   ``equity_t = cash_t + shares_t * close_t``.
3. **Signal** — the strategy's target for session ``t`` (computed from data up
   to this close) is compared with the current holding. If they differ, an
   order is queued for the next session's open. If they agree, nothing
   happens — a persistent signal never produces repeated orders.

The first session of the evaluation window is the *anchor*: the portfolio is in
cash at its close, so the first possible trade is at the following open. When
an evaluation start date is given, the anchor is the last session *before* it,
so a signal known on the eve of the evaluation period can be acted on at the
first evaluation open. The final session's signal cannot be executed (there is
no next open); it is reported as an unexecuted order, and an open position is
marked to market at the final close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import METHODOLOGY_VERSION, BacktestConfig
from quantlab.execution import ExecutionModel, Side
from quantlab.portfolio import Portfolio
from quantlab.strategies import BuyAndHold, Strategy

LEDGER_COLUMNS = [
    "fill_id",
    "signal_date",
    "execution_date",
    "side",
    "quantity",
    "reference_price",
    "execution_price",
    "gross_notional",
    "fee",
    "slippage_cost",
    "net_cash_flow",
    "cash_before",
    "cash_after",
    "shares_before",
    "shares_after",
    "reason",
]

TRADE_COLUMNS = [
    "trade_id",
    "status",
    "entry_signal_date",
    "entry_date",
    "exit_signal_date",
    "exit_date",
    "quantity",
    "entry_reference_price",
    "entry_price",
    "exit_reference_price",
    "exit_price",
    "entry_cost",
    "exit_value",
    "fees",
    "slippage_cost",
    "pnl",
    "net_return",
    "gross_return",
    "holding_sessions",
]


@dataclass
class BacktestResult:
    strategy_key: str
    strategy_label: str
    params: dict[str, Any]
    config: BacktestConfig
    daily: pd.DataFrame
    ledger: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    rejected_orders: pd.DataFrame
    unexecuted_order: dict[str, Any] | None
    first_signal_date: pd.Timestamp | None
    warmup_sessions: int
    reconciliation_issues: list[str] = field(default_factory=list)
    reconciliation: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- convenient views ------------------------------------------------------
    @property
    def equity(self) -> pd.Series:
        return self.daily["equity"]

    @property
    def returns(self) -> pd.Series:
        """Daily portfolio returns, excluding the anchor session (which has no prior value)."""
        return self.daily["equity"].pct_change().iloc[1:]

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital

    @property
    def final_value(self) -> float:
        return float(self.daily["equity"].iloc[-1])

    @property
    def start_date(self) -> pd.Timestamp:
        return self.daily.index[0]

    @property
    def end_date(self) -> pd.Timestamp:
        return self.daily.index[-1]

    @property
    def is_valid(self) -> bool:
        """True only when every accounting identity reconciles."""
        return not self.reconciliation_issues

    @property
    def liquidation_value(self) -> float:
        """Final value if any open position were sold at the final close, net of costs."""
        shares = float(self.daily["shares"].iloc[-1])
        close = float(self.daily["close"].iloc[-1])
        cash = float(self.daily["cash"].iloc[-1])
        if shares <= 0:
            return cash
        proceeds = shares * close * (1 - self.config.slippage_rate)
        return cash + proceeds * (1 - self.config.fee_rate)


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #


def simulate(
    prices: pd.DataFrame,
    targets: pd.Series,
    config: BacktestConfig,
    reasons: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """Run the session loop on an evaluation window.

    ``prices`` must be clean (no missing open/close); ``targets`` must share its
    index. Returns ``(daily, ledger, rejected_orders, unexecuted_order)``.
    """
    if not prices.index.equals(targets.index):
        raise ValueError("targets must be aligned to the price index")
    if len(prices) < 2:
        raise ValueError("at least two sessions are required to execute any trade")
    opens = prices["open"].to_numpy(dtype="float64")
    closes = prices["close"].to_numpy(dtype="float64")
    if not (np.isfinite(opens).all() and np.isfinite(closes).all()):
        raise ValueError("prices contain missing values; clean the data before backtesting")
    if (opens <= 0).any() or (closes <= 0).any():
        raise ValueError("prices must be strictly positive")

    dates = prices.index
    raw_target = targets.to_numpy(dtype="float64")
    desired = np.where(np.isnan(raw_target), 0.0, raw_target)
    reason_values = (
        reasons.reindex(dates).fillna("").to_numpy(dtype=object)
        if reasons is not None
        else np.full(len(dates), "", dtype=object)
    )

    model = ExecutionModel(config.fee_rate, config.slippage_rate, config.fractional_shares)
    book = Portfolio(cash=config.initial_capital)
    n = len(dates)
    cash_arr = np.empty(n)
    shares_arr = np.empty(n)
    fee_arr = np.zeros(n)
    slip_arr = np.zeros(n)
    notional_arr = np.zeros(n)

    ledger: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    unexecuted: dict[str, Any] | None = None
    buy_blocked = False  # set after a rejected buy until the signal turns off

    for i in range(n):
        if pending is not None:
            side = pending["side"]
            reference = opens[i]
            quantity = (
                model.buy_quantity(book.cash, reference) if side is Side.BUY else book.shares
            )
            if quantity <= 0:
                rejected.append(
                    {
                        "signal_date": dates[pending["signal_idx"]],
                        "attempted_date": dates[i],
                        "side": side.value,
                        "reason": "Insufficient cash to buy one whole share"
                        if side is Side.BUY
                        else "No shares to sell",
                    }
                )
                buy_blocked = side is Side.BUY
            else:
                fill = model.fill(side, quantity, reference)
                cash_before, cash_after, shares_before, shares_after = book.apply(fill)
                fee_arr[i] += fill.fee
                slip_arr[i] += fill.slippage_cost
                notional_arr[i] += fill.gross_notional
                ledger.append(
                    {
                        "fill_id": len(ledger) + 1,
                        "signal_date": dates[pending["signal_idx"]],
                        "execution_date": dates[i],
                        "side": side.value,
                        "quantity": fill.quantity,
                        "reference_price": fill.reference_price,
                        "execution_price": fill.execution_price,
                        "gross_notional": fill.gross_notional,
                        "fee": fill.fee,
                        "slippage_cost": fill.slippage_cost,
                        "net_cash_flow": fill.cash_flow,
                        "cash_before": cash_before,
                        "cash_after": cash_after,
                        "shares_before": shares_before,
                        "shares_after": shares_after,
                        "reason": pending["reason"],
                    }
                )
            pending = None

        cash_arr[i] = book.cash
        shares_arr[i] = book.shares

        want_invested = desired[i] == 1.0
        is_invested = book.shares > 0
        if not want_invested:
            buy_blocked = False
        if want_invested != is_invested and not (want_invested and buy_blocked):
            order = {
                "signal_idx": i,
                "side": Side.BUY if want_invested else Side.SELL,
                "reason": reason_values[i] or _fallback_reason(i, want_invested),
            }
            if i < n - 1:
                pending = order
            else:
                unexecuted = {
                    "signal_date": dates[i],
                    "side": order["side"].value,
                    "reason": order["reason"],
                    "note": "Signal on the final session; there is no next open to execute it.",
                }

    holdings = shares_arr * closes
    equity = cash_arr + holdings
    with np.errstate(invalid="ignore", divide="ignore"):
        exposure = np.where(equity > 0, holdings / equity, 0.0)
    daily = pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "target": raw_target,
            "shares": shares_arr,
            "cash": cash_arr,
            "holdings_value": holdings,
            "equity": equity,
            "exposure": exposure,
            "traded_notional": notional_arr,
            "fees": fee_arr,
            "slippage_cost": slip_arr,
        },
        index=dates,
    )
    daily.index.name = "date"
    ledger_df = pd.DataFrame(ledger, columns=LEDGER_COLUMNS)
    rejected_df = pd.DataFrame(
        rejected, columns=["signal_date", "attempted_date", "side", "reason"]
    )
    return daily, ledger_df, rejected_df, unexecuted


def _fallback_reason(signal_idx: int, want_invested: bool) -> str:
    if signal_idx == 0:
        state = "long" if want_invested else "cash"
        return (
            f"Strategy state was already '{state}' at the start of the evaluation window "
            "(derived from the warm-up history)"
        )
    return "Target exposure changed to long" if want_invested else "Target exposure changed to cash"


# --------------------------------------------------------------------------- #
# Round-trip trades and reconciliation
# --------------------------------------------------------------------------- #


def build_round_trips(ledger: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Pair each buy with the following sell. An open position is marked to market.

    ``net_return`` = (exit value - entry cost) / entry cost, where entry cost
    includes the buy fee and exit value is sale proceeds net of the sell fee.
    ``gross_return`` = exit reference price / entry reference price - 1, i.e. the
    price move before any fees or slippage. For an open trade the exit value is
    the final close times quantity, with no exit costs deducted.
    """
    rows: list[dict[str, Any]] = []
    position = daily.index.get_loc  # session number of a date (binary search)
    entry: dict[str, Any] | None = None
    for fill in ledger.to_dict("records"):
        if fill["side"] == Side.BUY.value:
            entry = fill
            continue
        if entry is None:
            continue
        entry_cost = entry["gross_notional"] + entry["fee"]
        exit_value = fill["gross_notional"] - fill["fee"]
        rows.append(
            {
                "trade_id": len(rows) + 1,
                "status": "closed",
                "entry_signal_date": entry["signal_date"],
                "entry_date": entry["execution_date"],
                "exit_signal_date": fill["signal_date"],
                "exit_date": fill["execution_date"],
                "quantity": entry["quantity"],
                "entry_reference_price": entry["reference_price"],
                "entry_price": entry["execution_price"],
                "exit_reference_price": fill["reference_price"],
                "exit_price": fill["execution_price"],
                "entry_cost": entry_cost,
                "exit_value": exit_value,
                "fees": entry["fee"] + fill["fee"],
                "slippage_cost": entry["slippage_cost"] + fill["slippage_cost"],
                "pnl": exit_value - entry_cost,
                "net_return": exit_value / entry_cost - 1,
                "gross_return": fill["reference_price"] / entry["reference_price"] - 1,
                "holding_sessions": position(fill["execution_date"]) - position(entry["execution_date"]),
            }
        )
        entry = None
    if entry is not None:
        last_close = float(daily["close"].iloc[-1])
        entry_cost = entry["gross_notional"] + entry["fee"]
        exit_value = entry["quantity"] * last_close
        rows.append(
            {
                "trade_id": len(rows) + 1,
                "status": "open (marked to market)",
                "entry_signal_date": entry["signal_date"],
                "entry_date": entry["execution_date"],
                "exit_signal_date": pd.NaT,
                "exit_date": daily.index[-1],
                "quantity": entry["quantity"],
                "entry_reference_price": entry["reference_price"],
                "entry_price": entry["execution_price"],
                "exit_reference_price": last_close,
                "exit_price": last_close,
                "entry_cost": entry_cost,
                "exit_value": exit_value,
                "fees": entry["fee"],
                "slippage_cost": entry["slippage_cost"],
                "pnl": exit_value - entry_cost,
                "net_return": exit_value / entry_cost - 1,
                "gross_return": last_close / entry["reference_price"] - 1,
                "holding_sessions": len(daily) - 1 - position(entry["execution_date"]),
            }
        )
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


RECONCILIATION_CHECKS: tuple[tuple[str, str], ...] = (
    ("equity_identity", "Equity = cash + shares x close on every session"),
    ("no_leverage_or_shorts", "Cash and shares never negative (no leverage, no short selling)"),
    ("cash_flows", "Initial capital + ledger cash flows = final cash"),
    ("share_count", "Ledger quantities = final share count"),
    ("pnl_attribution", "Daily P&L = overnight move on prior shares + intraday move on current shares - fees - slippage"),
    ("fees_match", "Fees in the daily record = fees in the ledger"),
    ("timing", "Every fill executes on the session immediately after its signal"),
)


def reconciliation_report(
    daily: pd.DataFrame, ledger: pd.DataFrame, config: BacktestConfig
) -> list[dict[str, Any]]:
    """Run every accounting identity; one row per check with pass/fail and detail."""
    fail: dict[str, list[str]] = {key: [] for key, _ in RECONCILIATION_CHECKS}
    scale = max(config.initial_capital, 1.0)
    tol = 1e-7 * scale

    # 1. Equity = cash + shares * close on every session.
    recomputed = daily["cash"] + daily["shares"] * daily["close"]
    worst = float((recomputed - daily["equity"]).abs().max())
    if worst > tol:
        fail["equity_identity"].append(f"Equity differs from cash + shares x close by up to {worst:.6f}.")

    # 2. No negative cash or short positions.
    if (daily["cash"] < -tol).any():
        fail["no_leverage_or_shorts"].append("Cash went negative (leverage is not permitted).")
    if (daily["shares"] < -1e-9).any():
        fail["no_leverage_or_shorts"].append("Share count went negative (short selling is not permitted).")

    # 3. Ledger cash flows explain the final cash balance.
    implied_cash = config.initial_capital + (float(ledger["net_cash_flow"].sum()) if len(ledger) else 0.0)
    if abs(implied_cash - float(daily["cash"].iloc[-1])) > tol:
        fail["cash_flows"].append(
            f"Ledger cash flows imply final cash {implied_cash:.6f} but the portfolio holds "
            f"{float(daily['cash'].iloc[-1]):.6f}."
        )

    # 4. Ledger quantities explain the final share count.
    if len(ledger):
        signed = np.where(ledger["side"] == Side.BUY.value, ledger["quantity"], -ledger["quantity"])
        implied_shares = float(signed.sum())
    else:
        implied_shares = 0.0
    if abs(implied_shares - float(daily["shares"].iloc[-1])) > 1e-7 * max(1.0, abs(implied_shares)):
        fail["share_count"].append("Ledger quantities do not explain the final share count.")

    # 5. Session-by-session P&L attribution:
    #    dE_t = shares_{t-1} (open_t - close_{t-1}) + shares_t (close_t - open_t) - fees_t - slippage_t
    expected_change = (
        daily["shares"].shift(1) * (daily["open"] - daily["close"].shift(1))
        + daily["shares"] * (daily["close"] - daily["open"])
        - daily["fees"]
        - daily["slippage_cost"]
    )
    attribution_error = (daily["equity"].diff() - expected_change).iloc[1:].abs()
    if len(attribution_error) and float(attribution_error.max()) > tol:
        fail["pnl_attribution"].append(
            f"Daily P&L attribution fails by up to {float(attribution_error.max()):.6f} "
            f"(first on {attribution_error.idxmax().date()})."
        )

    # 6. Costs in the daily record match the ledger.
    if abs(float(daily["fees"].sum()) - (float(ledger["fee"].sum()) if len(ledger) else 0.0)) > tol:
        fail["fees_match"].append("Fees in the daily record do not match the trade ledger.")

    # 7. Timing: every fill executes on the session immediately after its signal.
    if len(ledger):
        exec_pos = daily.index.get_indexer(pd.DatetimeIndex(ledger["execution_date"]))
        signal_pos = daily.index.get_indexer(pd.DatetimeIndex(ledger["signal_date"]))
        late = np.flatnonzero(exec_pos != signal_pos + 1)
        if len(late):
            fill = ledger.iloc[late[0]]
            fail["timing"].append(
                f"Fill {fill['fill_id']} executed on {fill['execution_date'].date()}, not the "
                f"session after its signal ({fill['signal_date'].date()})."
            )
        if (ledger["execution_date"] <= ledger["signal_date"]).any():
            fail["timing"].append("A fill executed on or before its own signal date (look-ahead).")

    return [
        {"key": key, "check": label, "passed": not fail[key], "detail": " ".join(fail[key])}
        for key, label in RECONCILIATION_CHECKS
    ]


def reconcile(daily: pd.DataFrame, ledger: pd.DataFrame, config: BacktestConfig) -> list[str]:
    """Check accounting identities. Returns human-readable failures (empty list = all passed)."""
    return [row["detail"] for row in reconciliation_report(daily, ledger, config) if not row["passed"]]


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def evaluation_window(
    index: pd.DatetimeIndex,
    eval_start: pd.Timestamp | None,
    eval_end: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    """Sessions in the evaluation window, including the anchor session before it."""
    end = pd.Timestamp(eval_end) if eval_end is not None else index[-1]
    window = index[index <= end]
    if eval_start is None:
        return window
    start = pd.Timestamp(eval_start)
    before = window[window < start]
    on_or_after = window[window >= start]
    if len(on_or_after) == 0:
        return pd.DatetimeIndex([])
    # Anchor on the last session before the start date when history exists;
    # otherwise the first available session has to serve as the anchor.
    anchor = before[-1] if len(before) else on_or_after[0]
    return window[window >= anchor]


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    signals: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> BacktestResult:
    """Backtest ``strategy`` on ``prices`` over ``[eval_start, eval_end]``.

    Data before ``eval_start`` is used only to warm up indicators. Data after
    ``eval_end`` is discarded *before* signals are computed, so it cannot
    influence any decision even if a strategy were accidentally non-causal.
    """
    if eval_end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(eval_end)]
    if signals is None:
        signals = strategy.generate_signals(prices)
    else:
        signals = signals.loc[signals.index <= prices.index[-1]]
    window = evaluation_window(prices.index, eval_start, None)
    if len(window) < 2:
        raise ValueError("The evaluation window contains fewer than two sessions.")

    valid_targets = signals["target"].dropna()
    return backtest_from_signals(
        window_prices=prices.loc[window],
        window_signals=signals.loc[window],
        config=config,
        strategy_key=strategy.key,
        strategy_label=strategy.label,
        params=dict(strategy.params),
        warmup_sessions=strategy.warmup_sessions,
        first_signal_date=valid_targets.index[0] if len(valid_targets) else None,
        meta={
            "evaluation_start": window[window >= pd.Timestamp(eval_start)][0]
            if eval_start is not None
            else window[0],
            **(meta or {}),
        },
    )


def backtest_from_signals(
    window_prices: pd.DataFrame,
    window_signals: pd.DataFrame,
    config: BacktestConfig,
    strategy_key: str,
    strategy_label: str,
    params: dict[str, Any],
    warmup_sessions: int = 0,
    first_signal_date: pd.Timestamp | None = None,
    meta: dict[str, Any] | None = None,
) -> BacktestResult:
    """Simulate pre-computed signals over a window whose first row is the anchor.

    Used directly by walk-forward research, which stitches together targets
    produced by different parameter sets for consecutive test periods.
    """
    daily, ledger, rejected, unexecuted = simulate(
        window_prices, window_signals["target"], config, window_signals.get("reason")
    )
    trades = build_round_trips(ledger, daily)
    checks = reconciliation_report(daily, ledger, config)
    issues = [row["detail"] for row in checks if not row["passed"]]
    return BacktestResult(
        strategy_key=strategy_key,
        strategy_label=strategy_label,
        params=params,
        config=config,
        daily=daily,
        ledger=ledger,
        trades=trades,
        signals=window_signals,
        rejected_orders=rejected,
        unexecuted_order=unexecuted,
        first_signal_date=first_signal_date,
        warmup_sessions=warmup_sessions,
        reconciliation_issues=issues,
        reconciliation=checks,
        meta={
            "methodology_version": METHODOLOGY_VERSION,
            "evaluation_start": window_prices.index[0],
            "anchor_date": window_prices.index[0],
            "evaluation_end": window_prices.index[-1],
            **(meta or {}),
        },
    )


def run_buy_and_hold(
    prices: pd.DataFrame,
    config: BacktestConfig,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    meta: dict[str, Any] | None = None,
) -> BacktestResult:
    """Benchmark run: buy-and-hold through the same engine, costs and timing."""
    return run_backtest(prices, BuyAndHold(), config, eval_start, eval_end, meta=meta)


def warmup_shortfall(result: BacktestResult) -> int:
    """Sessions inside the evaluation window spent waiting for the first valid signal."""
    if result.first_signal_date is None:
        return len(result.daily)
    return int((result.daily.index < result.first_signal_date).sum())
