"""Performance and risk metrics, computed from the portfolio equity series.

Conventions
-----------
* **Daily return** ``r_t = E_t / E_{t-1} - 1`` from the session-close equity.
  The anchor session contributes no return. ``N`` is the number of returns.
* **Annualised return (CAGR)** ``(E_T / E_0) ** (P / N) - 1`` with ``P = 252``
  sessions per year (geometric, session-count based).
* **Annualised volatility** ``std(r, ddof=1) * sqrt(P)``.
* **Sharpe ratio** ``mean(r - rf_d) / std(r - rf_d, ddof=1) * sqrt(P)``, with the
  annual risk-free rate converted to a daily rate ``rf_d = (1 + rf) ** (1/P) - 1``.
* **Sortino ratio** ``mean(r - rf_d) / DD * sqrt(P)`` where the downside deviation
  ``DD = sqrt(mean(min(r - rf_d, 0) ** 2))`` is taken over *all* N observations.
* **Maximum drawdown** ``min_t (E_t / max_{s<=t} E_s - 1)``.

A metric is ``None`` when it is mathematically undefined (e.g. Sharpe with zero
volatility), and is marked ``reliable=False`` with an explanatory note when it
is defined but rests on too few observations to mean much.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from quantlab.config import (
    MIN_OBS_FOR_ANNUALISED,
    MIN_OBS_FOR_RATIO,
    MIN_TRADES_FOR_STATS,
    TRADING_DAYS_PER_YEAR,
)

_EPS = 1e-12


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    value: float | None
    fmt: str  # pct | ratio | currency | int | number | text | date
    definition: str
    note: str = ""
    reliable: bool = True
    text: str | None = None  # for text/date metrics

    def formatted(self) -> str:
        return format_value(self.value if self.text is None else self.text, self.fmt)


def format_value(value: object, fmt: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if fmt == "pct":
        return f"{value:+.2%}" if value != 0 else "0.00%"
    if fmt == "pct_abs":
        return f"{value:.2%}"
    if fmt == "ratio":
        return f"{value:.2f}"
    if fmt == "currency":
        return f"${value:,.2f}"
    if fmt == "int":
        return f"{int(value):,}"
    if fmt == "number":
        return f"{value:,.2f}"
    return str(value)


@dataclass
class MetricReport:
    metrics: dict[str, Metric] = field(default_factory=dict)

    def add(self, metric: Metric) -> None:
        self.metrics[metric.key] = metric

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    def get(self, key: str) -> Metric | None:
        return self.metrics.get(key)

    def value(self, key: str) -> float | None:
        metric = self.metrics.get(key)
        return None if metric is None else metric.value

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "key": m.key,
                    "Metric": m.label,
                    "Value": m.formatted(),
                    "raw_value": m.value,
                    "Reliable": m.reliable,
                    "Note": m.note,
                    "Definition": m.definition,
                }
                for m in self.metrics.values()
            ]
        )

    def to_dict(self) -> dict[str, float | str | None]:
        return {k: (m.text if m.text is not None else m.value) for k, m in self.metrics.items()}


# --------------------------------------------------------------------------- #
# Building blocks (pure functions, unit-tested individually)
# --------------------------------------------------------------------------- #


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().iloc[1:]


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def annualised_return(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    n = len(equity) - 1
    if n < 1:
        return None
    growth = float(equity.iloc[-1] / equity.iloc[0])
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / n) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    if len(returns) < 2:
        return None
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def daily_risk_free(annual_rate: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return (1 + annual_rate) ** (1 / periods_per_year) - 1


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float | None:
    if len(returns) < 2:
        return None
    excess = returns - daily_risk_free(risk_free_rate, periods_per_year)
    std = float(excess.std(ddof=1))
    if std < _EPS:
        return None
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float | None:
    if len(returns) < 2:
        return None
    excess = returns - daily_risk_free(risk_free_rate, periods_per_year)
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    if dd < _EPS:
        return None
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def drawdown_episodes(equity: pd.Series) -> pd.DataFrame:
    """Every peak-to-recovery episode: depth, dates and durations (sessions and calendar days).

    An episode starts at the last peak before equity falls below it and ends at
    the first session equity regains the peak. If it never does, ``recovery`` is
    ``NaT`` and ``ongoing`` is True.
    """
    dd = drawdown_series(equity).to_numpy()
    dates = equity.index
    rows = []
    i, n = 0, len(dd)
    while i < n:
        if dd[i] < -_EPS:
            start = i - 1 if i > 0 else 0
            j = i
            trough = i
            while j < n and dd[j] < -_EPS:
                if dd[j] < dd[trough]:
                    trough = j
                j += 1
            recovered = j < n
            end = j if recovered else n - 1
            rows.append(
                {
                    "peak": dates[start],
                    "trough": dates[trough],
                    "recovery": dates[j] if recovered else pd.NaT,
                    "depth": float(dd[trough]),
                    "sessions": end - start,
                    "calendar_days": int((dates[end] - dates[start]).days),
                    "sessions_to_trough": trough - start,
                    "ongoing": not recovered,
                }
            )
            i = j
        else:
            i += 1
    return pd.DataFrame(
        rows,
        columns=["peak", "trough", "recovery", "depth", "sessions", "calendar_days", "sessions_to_trough", "ongoing"],
    )


def period_returns(equity: pd.Series, freq: str) -> pd.Series:
    """Calendar-period returns (``freq`` = 'ME' month-end or 'YE' year-end).

    Each period's return is measured from the previous period's final equity;
    the first period is measured from the starting (anchor) equity. The first
    and last periods can be partial.
    """
    if len(equity) < 2:
        return pd.Series(dtype="float64")
    # The anchor is only the base of the first period, never a period of its own:
    # an anchor dated 31 Dec would otherwise create a bogus 0.00% year.
    period_end = equity.iloc[1:].resample(freq).last().dropna()
    previous = period_end.shift(1)
    previous.iloc[0] = equity.iloc[0]
    return period_end / previous - 1


def annualised_turnover(
    ledger: pd.DataFrame, equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float | None:
    """Total traded notional (buys + sells) / average equity, per year.

    A strategy that buys once and never sells over one year has turnover 1.0
    (100%); a full round trip each month is roughly 24.0 (2,400%).
    """
    n = len(equity) - 1
    if n < 1:
        return None
    traded = float(ledger["gross_notional"].sum()) if len(ledger) else 0.0
    avg_equity = float(equity.mean())
    if avg_equity <= 0:
        return None
    return traded / avg_equity / (n / periods_per_year)


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #


def _sample_note(n: int, minimum: int, what: str) -> tuple[bool, str]:
    if n < minimum:
        return False, f"Only {n} observations (< {minimum}); {what} is not statistically meaningful."
    return True, ""


def exposed_sessions(result) -> int:
    """Sessions (after the anchor) whose return carried market risk: a position at the previous close or at
    this close. Sharpe inference on a mostly idle strategy rests on these, not on the window length."""
    shares = result.daily["shares"]
    return int(((shares > 0) | (shares.shift(1) > 0)).iloc[1:].sum())


def compute_metrics(result, benchmark=None, gross=None) -> MetricReport:
    """Compute the full metric set for a :class:`~quantlab.backtest.BacktestResult`.

    ``benchmark`` (a buy-and-hold result over the same sessions) and ``gross``
    (the same strategy with zero costs) are optional.
    """
    cfg = result.config
    ppy = cfg.periods_per_year
    equity = result.equity
    rets = result.returns
    n = len(rets)
    report = MetricReport()
    add = report.add

    add(Metric("initial_value", "Initial portfolio value", float(equity.iloc[0]), "currency",
               "Cash at the close of the anchor session."))
    add(Metric("final_value", "Final portfolio value", float(equity.iloc[-1]), "currency",
               "Cash plus shares valued at the final close (mark-to-market)."))
    add(Metric("liquidation_value", "Final value if liquidated", result.liquidation_value, "currency",
               "Final value after hypothetically selling any open position at the final close, "
               "net of slippage and fees."))
    add(Metric("total_return", "Total return (net)", total_return(equity), "pct",
               "Final value / initial value - 1, after all fees and slippage."))

    ann_ok, ann_note = _sample_note(n, MIN_OBS_FOR_ANNUALISED, "the annualised figure (an extrapolation)")
    add(Metric("cagr", "Annualised return (CAGR)", annualised_return(equity, ppy), "pct",
               f"(Final / initial) ^ ({ppy} / N) - 1, N = number of daily returns.",
               ann_note, ann_ok))

    vol = annualised_volatility(rets, ppy)
    ratio_ok, ratio_note = _sample_note(n, MIN_OBS_FOR_RATIO, "this ratio")
    add(Metric("volatility", "Annualised volatility", vol, "pct_abs",
               f"Sample standard deviation of daily returns x sqrt({ppy}).",
               ratio_note if vol is not None else "Needs at least two returns.", ratio_ok and vol is not None))

    sharpe = sharpe_ratio(rets, cfg.risk_free_rate, ppy)
    sharpe_note = ratio_note
    if sharpe is None:
        sharpe_note = "Undefined: daily returns have zero volatility (e.g. the strategy never invested)."
    add(Metric("sharpe", "Sharpe ratio", sharpe, "ratio",
               f"Mean daily excess return / std of daily excess returns x sqrt({ppy}); "
               f"risk-free rate {cfg.risk_free_rate:.2%} p.a.",
               sharpe_note, ratio_ok and sharpe is not None))

    sortino = sortino_ratio(rets, cfg.risk_free_rate, ppy)
    n_down = int((rets < daily_risk_free(cfg.risk_free_rate, ppy)).sum())
    sortino_ok = ratio_ok and sortino is not None and n_down >= 20
    if sortino is None:
        sortino_note = "Undefined: no returns below the risk-free rate, so downside deviation is zero."
    elif n_down < 20:
        sortino_note = f"Only {n_down} below-target returns; downside deviation is poorly estimated."
    else:
        sortino_note = ratio_note
    add(Metric("sortino", "Sortino ratio", sortino, "ratio",
               f"Mean daily excess return / downside deviation x sqrt({ppy}); downside deviation uses "
               "all observations with positive excess returns set to zero.",
               sortino_note, sortino_ok))

    _add_inference_metrics(report, rets, cfg.risk_free_rate, ppy, exposed_sessions(result))

    mdd = max_drawdown(equity)
    episodes = drawdown_episodes(equity)
    add(Metric("max_drawdown", "Maximum drawdown", mdd, "pct",
               "Largest peak-to-trough fall in portfolio value, measured at session closes."))
    if len(episodes):
        longest = episodes.loc[episodes["sessions"].idxmax()]
        deepest = episodes.loc[episodes["depth"].idxmin()]
        add(Metric("max_dd_duration", "Longest drawdown (sessions)", float(longest["sessions"]), "int",
                   "Sessions from a peak until the peak is regained (or until the sample ends).",
                   "Still ongoing at the end of the sample." if longest["ongoing"] else ""))
        add(Metric("max_dd_duration_days", "Longest drawdown (calendar days)", float(longest["calendar_days"]), "int",
                   "Calendar days from a peak until the peak is regained (or the sample ends)."))
        add(Metric("max_dd_peak", "Max drawdown peak date", None, "date",
                   "Session of the peak preceding the maximum drawdown.", text=str(deepest["peak"].date())))
        add(Metric("max_dd_trough", "Max drawdown trough date", None, "date",
                   "Session of the lowest point of the maximum drawdown.", text=str(deepest["trough"].date())))
        rec = deepest["recovery"]
        add(Metric("max_dd_recovery", "Max drawdown recovery date", None, "date",
                   "First session at which the prior peak was regained.",
                   text=str(rec.date()) if pd.notna(rec) else "Not recovered"))
    else:
        add(Metric("max_dd_duration", "Longest drawdown (sessions)", 0.0, "int",
                   "Sessions from a peak until the peak is regained.", "No drawdown occurred."))

    cagr = report.value("cagr")
    calmar = None if (cagr is None or abs(mdd) < _EPS) else cagr / abs(mdd)
    add(Metric("calmar", "Calmar ratio", calmar, "ratio",
               "Annualised return / |maximum drawdown|.",
               "Undefined without a drawdown." if calmar is None else ann_note, ann_ok and calmar is not None))

    # ---- activity --------------------------------------------------------------
    ledger, trades, daily = result.ledger, result.trades, result.daily
    invested = (daily["shares"].iloc[1:] > 0) if n else pd.Series(dtype=bool)
    add(Metric("fills", "Executions (fills)", float(len(ledger)), "int",
               "Number of individual buy and sell executions."))
    held = (daily["shares"] > 0).astype(int)
    transitions = float((held.diff().iloc[1:] != 0).sum()) if n else 0.0
    add(Metric("position_changes", "Entries and exits", transitions, "int",
               "Times the position moved between flat (all cash) and invested, in either direction. "
               "Fills that only resize an existing position are not counted."))
    closed = trades[trades["status"] == "closed"] if len(trades) else trades
    add(Metric("round_trips", "Closed round-trip trades", float(len(closed)), "int",
               "Completed flat-to-flat position episodes. One episode can include several fills "
               "when the strategy rebalances a partial position."))
    add(Metric("open_trades", "Open trades at end", float(len(trades) - len(closed)), "int",
               "Positions still held at the final close (marked to market, not sold)."))
    add(Metric("time_invested", "Sessions invested", float(invested.mean()) if n else None, "pct_abs",
               "Share of sessions (after the anchor) ending with a non-zero position."))
    add(Metric("avg_exposure", "Average exposure", float(daily["exposure"].iloc[1:].mean()) if n else None, "pct_abs",
               "Mean of holdings value / equity at each close (cash drag from whole shares lowers it)."))
    turnover = annualised_turnover(ledger, equity, ppy)
    add(Metric("turnover", "Annualised turnover", turnover, "pct_abs",
               "Total traded notional (buys + sells) / average equity / years.", ann_note, ann_ok))
    add(Metric("traded_notional", "Total traded notional", float(ledger["gross_notional"].sum()) if len(ledger) else 0.0,
               "currency", "Sum of |quantity x execution price| over all fills."))

    # ---- costs -----------------------------------------------------------------
    fees = float(ledger["fee"].sum()) if len(ledger) else 0.0
    slippage = float(ledger["slippage_cost"].sum()) if len(ledger) else 0.0
    add(Metric("total_fees", "Total transaction fees", fees, "currency", "Sum of fees charged on traded notional."))
    add(Metric("total_slippage", "Estimated slippage cost", slippage, "currency",
               "Sum of quantity x |execution price - reference open|. An estimate: real slippage varies."))
    add(Metric("cost_pct_capital", "Total costs / initial capital", (fees + slippage) / cfg.initial_capital,
               "pct_abs", "Fees plus slippage as a share of starting capital."))

    if gross is not None:
        gross_tr = total_return(gross.equity)
        add(Metric("gross_total_return", "Total return (gross, no costs)", gross_tr, "pct",
                   "The same signals simulated with zero fees and zero slippage."))
        add(Metric("cost_drag", "Cost drag", report.value("total_return") - gross_tr, "pct",
                   "Net total return minus gross total return (percentage points)."))
        gross_cagr = annualised_return(gross.equity, ppy)
        add(Metric("gross_cagr", "Annualised return (gross)", gross_cagr, "pct",
                   "CAGR of the zero-cost simulation.", ann_note, ann_ok))

    # ---- trade-level statistics -----------------------------------------------
    n_closed = len(closed)
    trade_ok = n_closed >= MIN_TRADES_FOR_STATS
    trade_note = (
        "" if trade_ok else
        f"Only {n_closed} closed trade(s) (< {MIN_TRADES_FOR_STATS}); treat as anecdotal."
    )
    if n_closed:
        r = closed["net_return"]
        wins, losses = r[r > 0], r[r <= 0]
        gross_win = float(closed.loc[closed["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-closed.loc[closed["pnl"] <= 0, "pnl"].sum())
        add(Metric("avg_trade_return", "Average trade return (net)", float(r.mean()), "pct",
                   "Mean of closed-trade net returns (after both legs' fees and slippage).", trade_note, trade_ok))
        add(Metric("win_rate", "Win rate", float((r > 0).mean()), "pct_abs",
                   "Share of closed trades with a positive net return.", trade_note, trade_ok))
        add(Metric("avg_winner", "Average winning trade", float(wins.mean()) if len(wins) else None, "pct",
                   "Mean net return of closed trades with a positive net return.",
                   "No winning trades." if not len(wins) else trade_note, trade_ok and bool(len(wins))))
        add(Metric("avg_loser", "Average losing trade", float(losses.mean()) if len(losses) else None, "pct",
                   "Mean net return of closed trades with a zero or negative net return.",
                   "No losing trades." if not len(losses) else trade_note, trade_ok and bool(len(losses))))
        add(Metric("profit_factor", "Profit factor", gross_win / gross_loss if gross_loss > _EPS else None, "ratio",
                   "Sum of winning-trade P&L / |sum of losing-trade P&L|.",
                   "Undefined without losing trades." if gross_loss <= _EPS else trade_note,
                   trade_ok and gross_loss > _EPS))
        add(Metric("avg_holding", "Average holding period (sessions)", float(closed["holding_sessions"].mean()),
                   "number", "Mean sessions between entry and exit fills of closed trades."))
        add(Metric("best_trade", "Best trade", float(r.max()), "pct", "Highest closed-trade net return."))
        add(Metric("worst_trade", "Worst trade", float(r.min()), "pct", "Lowest closed-trade net return."))
    else:
        add(Metric("avg_trade_return", "Average trade return (net)", None, "pct",
                   "Mean of closed-trade net returns.", "No closed trades in the sample.", False))
        add(Metric("win_rate", "Win rate", None, "pct_abs",
                   "Share of closed trades with a positive net return.", "No closed trades in the sample.", False))

    # ---- best / worst periods ----------------------------------------------------
    if n:
        add(Metric("best_day", "Best day", float(rets.max()), "pct",
                   f"Largest single-session portfolio return ({rets.idxmax().date()})."))
        add(Metric("worst_day", "Worst day", float(rets.min()), "pct",
                   f"Largest single-session portfolio loss ({rets.idxmin().date()})."))
        monthly = period_returns(equity, "ME")
        if len(monthly):
            add(Metric("best_month", "Best calendar month", float(monthly.max()), "pct",
                       f"Best calendar-month return ({monthly.idxmax():%b %Y}); first/last months may be partial."))
            add(Metric("worst_month", "Worst calendar month", float(monthly.min()), "pct",
                       f"Worst calendar-month return ({monthly.idxmin():%b %Y}); first/last months may be partial."))
        yearly = period_returns(equity, "YE")
        if len(yearly) >= 2:
            add(Metric("best_year", "Best calendar year", float(yearly.max()), "pct",
                       f"Best calendar-year return ({yearly.idxmax():%Y}); first/last years may be partial."))
            add(Metric("worst_year", "Worst calendar year", float(yearly.min()), "pct",
                       f"Worst calendar-year return ({yearly.idxmin():%Y}); first/last years may be partial."))

    # ---- benchmark-relative --------------------------------------------------------
    if benchmark is not None:
        bench_eq = benchmark.equity.reindex(equity.index)
        if bench_eq.isna().any():
            raise ValueError("benchmark must cover exactly the same sessions as the strategy")
        bench_rets = daily_returns(bench_eq)
        b_tr = total_return(bench_eq)
        add(Metric("benchmark_total_return", "Benchmark total return", b_tr, "pct",
                   "Buy-and-hold of the benchmark through the same engine, costs and sessions."))
        add(Metric("excess_return", "Excess total return vs benchmark", report.value("total_return") - b_tr, "pct",
                   "Strategy total return minus benchmark total return (percentage points)."))
        b_cagr = annualised_return(bench_eq, ppy)
        add(Metric("benchmark_cagr", "Benchmark annualised return", b_cagr, "pct",
                   "CAGR of the benchmark over the same sessions.", ann_note, ann_ok))
        if cagr is not None and b_cagr is not None:
            add(Metric("excess_cagr", "Excess annualised return", cagr - b_cagr, "pct",
                       "Strategy CAGR minus benchmark CAGR.", ann_note, ann_ok))
        add(Metric("benchmark_max_drawdown", "Benchmark maximum drawdown", max_drawdown(bench_eq), "pct",
                   "Largest peak-to-trough fall of the benchmark."))
        add(Metric("benchmark_sharpe", "Benchmark Sharpe ratio", sharpe_ratio(bench_rets, cfg.risk_free_rate, ppy),
                   "ratio", "Sharpe ratio of the benchmark, same convention.", ratio_note, ratio_ok))
        b_var = float(bench_rets.var(ddof=1)) if n >= 2 else 0.0
        if n >= 2 and b_var > _EPS:
            beta = float(rets.cov(bench_rets) / b_var)
            corr = float(rets.corr(bench_rets)) if rets.std() > _EPS else None
            active = rets - bench_rets
            te = float(active.std(ddof=1) * math.sqrt(ppy))
            ir = float(active.mean() * ppy / te) if te > _EPS else None
            add(Metric("beta", "Beta to benchmark", beta, "ratio",
                       "Cov(strategy, benchmark daily returns) / Var(benchmark daily returns).", ratio_note, ratio_ok))
            add(Metric("correlation", "Correlation with benchmark", corr, "ratio",
                       "Pearson correlation of daily returns.", ratio_note, ratio_ok and corr is not None))
            add(Metric("tracking_error", "Tracking error", te, "pct_abs",
                       "Annualised std of daily active returns (strategy - benchmark).", ratio_note, ratio_ok))
            add(Metric("information_ratio", "Information ratio", ir, "ratio",
                       "Annualised mean active return / tracking error.",
                       ratio_note if ir is not None else "Undefined: zero tracking error.", ratio_ok and ir is not None))
        from quantlab import inference

        ab = inference.alpha_beta(rets, bench_rets, cfg.risk_free_rate, ppy)
        if ab is not None:
            ab_ok, ab_note = _sample_note(n, MIN_OBS_FOR_ANNUALISED, "the regression")
            add(Metric("alpha_annual", "Alpha (annualised)", ab.alpha_annual, "pct",
                       "Intercept of daily strategy excess returns regressed on benchmark excess returns, "
                       f"x {ppy}. Excess = return minus the daily risk-free rate.", ab_note, ab_ok))
            add(Metric("alpha_tstat", "Alpha t-statistic (Newey-West)", ab.alpha_tstat, "ratio",
                       f"Alpha / its Newey-West (HAC, {ab.lags} lags) standard error. |t| below about 2 means "
                       "the alpha is not distinguishable from zero.", ab_note, ab_ok and ab.alpha_tstat is not None))
            add(Metric("beta_tstat", "Beta t-statistic (Newey-West)", ab.beta_tstat, "ratio",
                       "Regression beta / its Newey-West standard error.", ab_note,
                       ab_ok and ab.beta_tstat is not None))

    add(Metric("observations", "Daily return observations", float(n), "int",
               "Sessions in the evaluation window after the anchor session."))
    return report


def _add_inference_metrics(report: MetricReport, rets: pd.Series, risk_free_rate: float, ppy: int,
                           n_exposed: int | None = None) -> None:
    """Sampling uncertainty of the Sharpe ratio and the shape of the return distribution.

    ``n_exposed`` is the number of sessions that carried market risk. The Sharpe interval, PSR and minimum
    track record of a strategy that is mostly in cash rest on those few sessions, so they are flagged as
    unreliable when there are fewer than a year of them, however long the window.
    """
    from quantlab import inference  # imported here: inference itself builds on this module

    add = report.add
    n = len(rets)
    ok, note = _sample_note(n, MIN_OBS_FOR_ANNUALISED, "this estimate")
    sr_ok, sr_note = ok, note
    if ok and n_exposed is not None and n_exposed < MIN_OBS_FOR_ANNUALISED:
        sr_ok, sr_note = False, (f"Only {n_exposed} sessions with market exposure (< {MIN_OBS_FOR_ANNUALISED}): "
                                 "the other returns are idle cash, so this estimate is not statistically meaningful.")
    si = inference.sharpe_inference(rets, risk_free_rate, ppy)
    if si is not None:
        add(Metric("sharpe_se", "Sharpe ratio standard error", si.se_annual, "ratio",
                   "sqrt((1 - skew x SR + (kurtosis - 1) / 4 x SR^2) / (N - 1)) x sqrt(P), per-period SR of "
                   "excess returns (Lo 2002; Mertens 2002), never below the normal-returns value "
                   "sqrt((1 + SR^2 / 2) / (N - 1)). Assumes serially independent returns.", sr_note, sr_ok))
        add(Metric("sharpe_ci_low", "Sharpe ratio, 95% CI lower bound", si.ci_lower, "ratio",
                   "Sharpe ratio - 1.96 x standard error.", sr_note, sr_ok))
        add(Metric("sharpe_ci_high", "Sharpe ratio, 95% CI upper bound", si.ci_upper, "ratio",
                   "Sharpe ratio + 1.96 x standard error.", sr_note, sr_ok))
        psr = NormalDist().cdf(si.sharpe_per_period / si.se_per_period) if si.se_per_period > 0 else None
        add(Metric("psr", "Probabilistic Sharpe ratio (vs 0)", psr, "pct_abs",
                   "Probability that the true Sharpe ratio is above zero, allowing for sample length, skew and "
                   "fat tails (Bailey & López de Prado 2012). Ignores how many configurations were tried.",
                   sr_note, sr_ok and psr is not None))
    trl = inference.min_track_record_length(rets, risk_free_rate, ppy)
    add(Metric("min_track_record_years", "Minimum track record for 95% confidence (years)",
               None if trl is None else trl.years, "number",
               "Years of data needed before a Sharpe ratio this high would be significantly above zero at 95% "
               "(one-sided), given the observed skew and kurtosis (Bailey & López de Prado 2012).",
               "Undefined: the Sharpe ratio is not above zero." if trl is None else sr_note,
               sr_ok and trl is not None))
    rf_d = daily_risk_free(risk_free_rate, ppy)
    t_hac = inference.hac_mean_tstat(rets - rf_d)
    add(Metric("mean_excess_tstat", "t-statistic of mean excess return (Newey-West)", t_hac, "ratio",
               "Mean daily excess return / its Newey-West (HAC) standard error, which allows for "
               "autocorrelation. |t| below about 2 means the mean is not distinguishable from zero.",
               note, ok and t_hac is not None))
    add(Metric("skewness", "Skewness of daily returns", inference.skewness(rets), "ratio",
               "Third standardised moment. Negative = large losses more likely than large gains."))
    add(Metric("excess_kurtosis", "Excess kurtosis of daily returns", inference.excess_kurtosis(rets), "ratio",
               "Fourth standardised moment minus 3 (0 for a normal distribution). Positive = fat tails."))
    tail_ok, tail_note = _sample_note(n, MIN_OBS_FOR_ANNUALISED, "a 5% tail estimate")
    add(Metric("var_95", "Value at risk, 95% (1 day, historical)", inference.value_at_risk(rets, 0.95), "pct",
               "5th percentile of daily returns: on 1 session in 20 the loss was at least this large.",
               tail_note, tail_ok))
    add(Metric("cvar_95", "Expected shortfall, 95% (1 day, historical)", inference.conditional_value_at_risk(rets),
               "pct", "Average daily return on the sessions at or below the 95% value at risk.", tail_note, tail_ok))
    add(Metric("var_95_cf", "Value at risk, 95% (Cornish-Fisher)",
               inference.value_at_risk(rets, 0.95, "cornish_fisher"), "pct",
               "Normal VaR adjusted for the observed skewness and kurtosis.", tail_note, tail_ok))
    add(Metric("tail_ratio", "Tail ratio (95th / 5th percentile)", inference.tail_ratio(rets), "ratio",
               "Size of the 95th-percentile daily gain relative to the 5th-percentile daily loss.",
               tail_note, tail_ok))


def summary_row(result, report: MetricReport) -> dict[str, object]:
    """Compact dictionary used by comparison tables and experiment records."""
    keys: Iterable[str] = (
        "total_return", "gross_total_return", "cagr", "volatility", "sharpe", "sortino",
        "max_drawdown", "calmar", "time_invested", "fills", "round_trips", "turnover",
        "total_fees", "total_slippage", "win_rate", "excess_return",
    )
    row: dict[str, object] = {"strategy": result.strategy_label}
    for key in keys:
        row[key] = report.value(key)
    return row
