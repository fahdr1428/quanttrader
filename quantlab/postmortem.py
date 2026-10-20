"""POST-TRADE REVIEW: each simulated trade against its recorded thesis, and patterns across trades.

R is the trade's initial risk per unit, ``entry fill − thesis stop``. Realised R is
net of fees; MAE / MFE (maximum adverse / favourable excursion) use the bars'
lows and highs from the entry bar to the exit bar. A trade without a recorded
thesis has no R and is reported as such. With fewer than 20 closed trades every
summary statistic is noise-dominated, and the summary says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantlab.paper import PaperBook, PaperFill, PaperOrder
from quantlab.serialization import clean_json

MIN_TRADES = 20
RECOVERY_SESSIONS = 10


@dataclass(frozen=True)
class TradeReview:
    order_id: int
    ticker: str
    entry_date: date
    exit_date: date | None
    entry: float
    exit: float | None
    realised_r: float | None
    mae_r: float | None
    mfe_r: float | None
    holding_days: int | None            # sessions from the entry bar to the exit bar
    expected_holding_days: int | None
    outcome: str                        # target | stop | manual | invalidated | open
    slippage_vs_assumed: float | None   # planned exit level minus actual exit, in R (positive = worse)
    notes: tuple[str, ...]
    confidence: str | None = None
    pnl: float | None = None            # money, net of both legs' fees (simulated)

    def to_dict(self) -> dict[str, Any]:
        return clean_json({**asdict(self), "simulated": True})


def _exit_fills(book: PaperBook, entry: PaperOrder) -> list[tuple[PaperOrder, PaperFill]]:
    children = [o for o in book.order_list(entry.ticker) if o.parent_id == entry.id and o.side == "sell"]
    fills = {f.order_id: f for f in book.fill_list(entry.ticker)}
    return sorted(((o, fills[o.id]) for o in children if o.id in fills), key=lambda x: (x[1].session, x[1].id))


def review_trade(book: PaperBook, order_id: int, prices: pd.DataFrame) -> TradeReview:
    """Review one filled entry order against its thesis and the bars that followed."""
    entry = book.get_order(order_id)
    if entry is None or entry.side != "buy":
        raise ValueError(f"Order {order_id} is not a simulated entry (buy) order.")
    entry_fill = next((f for f in book.fill_list(entry.ticker) if f.order_id == entry.id), None)
    if entry_fill is None:
        raise ValueError(f"Order {order_id} has not filled ({entry.status}).")
    thesis = entry.thesis
    notes: list[str] = []
    entry_price = entry_fill.price
    risk = entry_price - thesis.stop if thesis else None
    if thesis is None:
        notes.append("No thesis was recorded: R, excursions in R and the plan comparison are not defined.")
    elif risk <= 0:
        notes.append("The entry filled at or below the thesis stop: R is not defined.")
        risk = None

    exits = _exit_fills(book, entry)
    sold = sum(f.quantity for _, f in exits)
    closed = sold >= entry_fill.quantity - 1e-9
    exit_order, exit_fill = exits[-1] if exits else (None, None)
    exit_price = (sum(f.quantity * f.price for _, f in exits) / sold) if sold else None
    if exits and not closed:
        notes.append(f"Partly closed: {sold:g} of {entry_fill.quantity:g} units sold.")
    if closed:
        outcome = {"stop": "stop", "target": "target"}.get(exit_order.role, "manual")
        if outcome == "manual" and exit_order.note.lower().startswith("invalidated"):
            outcome = "invalidated"
    else:
        outcome = "open"

    start = pd.Timestamp(entry_fill.session)
    end = pd.Timestamp(exit_fill.session) if closed else None
    window = prices.loc[(prices.index >= start) & ((prices.index <= end) if end is not None else True)]
    holding = int(len(window) - 1) if len(window) else None
    if not len(window):
        notes.append("No price bars cover the holding period: excursions are not available.")

    realised_r = mae_r = mfe_r = slip_r = pnl = None
    if closed:
        fees = entry_fill.fee * (sold / entry_fill.quantity) + sum(f.fee for _, f in exits)
        pnl = sold * (exit_price - entry_price) - fees
        if risk:
            realised_r = pnl / (sold * risk)
    if risk and len(window):
        mae_r = (float(window["low"].min()) - entry_price) / risk
        mfe_r = (float(window["high"].max()) - entry_price) / risk
    if closed and risk and exit_order.role in ("stop", "target"):
        planned = exit_order.stop_price if exit_order.role == "stop" else exit_order.limit_price
        slip_r = (planned - exit_price) / risk
        if exit_order.role == "stop" and exit_fill.reference_price < planned - 1e-9:
            notes.append(f"Gapped through the stop: filled at {exit_price:.4g} against a stop of {planned:.4g}.")
    if outcome == "stop" and thesis and end is not None:
        after = prices.loc[prices.index > end].iloc[:RECOVERY_SESSIONS]
        if len(after) and float(after["close"].max()) > entry_price:
            notes.append(f"Stopped out, then closed above the entry within {RECOVERY_SESSIONS} sessions "
                         "(the stop may have been too tight for the asset's noise).")
    expected = thesis.expected_holding_days if thesis else None
    if closed and expected and holding is not None and holding > 2 * expected:
        notes.append(f"Held {holding} sessions against {expected} expected.")
    return TradeReview(order_id=entry.id, ticker=entry.ticker, entry_date=start.date(),
                       exit_date=end.date() if end is not None else None, entry=entry_price, exit=exit_price,
                       realised_r=realised_r, mae_r=mae_r, mfe_r=mfe_r, holding_days=holding,
                       expected_holding_days=expected, outcome=outcome, slippage_vs_assumed=slip_r,
                       notes=tuple(notes), confidence=thesis.confidence if thesis else None, pnl=pnl)


def review_all(book: PaperBook, prices_by_ticker: dict[str, pd.DataFrame]) -> tuple[list[TradeReview], dict[str, Any]]:
    """Every filled entry reviewed, plus summary statistics and recurring patterns."""
    filled = {f.order_id for f in book.fill_list()}
    entries = [o for o in book.order_list() if o.side == "buy" and o.id in filled]
    reviews = []
    for entry in entries:
        prices = prices_by_ticker.get(entry.ticker)
        if prices is None:
            prices = pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([]))
        reviews.append(review_trade(book, entry.id, prices))
    closed = [r for r in reviews if r.outcome != "open"]
    with_r = [r for r in closed if r.realised_r is not None]
    rs = np.array([r.realised_r for r in with_r], dtype="float64")
    wins, losses = rs[rs > 0], rs[rs <= 0]
    summary: dict[str, Any] = {
        "trades": len(reviews), "closed": len(closed), "open": len(reviews) - len(closed),
        "closed_with_r": len(with_r),
        "hit_rate": float(len(wins) / len(rs)) if len(rs) else None,
        "average_r": float(rs.mean()) if len(rs) else None,
        "expectancy_r": float(rs.mean()) if len(rs) else None,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
        "total_pnl": float(sum(r.pnl for r in closed if r.pnl is not None)),
        "outcomes": {k: sum(r.outcome == k for r in reviews) for k in ("target", "stop", "manual", "invalidated",
                                                                        "open")},
        "definitions": "R = entry − thesis stop; expectancy = mean realised R per closed trade; profit factor = "
                       "sum of winning R / |sum of losing R|. All trades are SIMULATED.",
    }
    calibration = {}
    for level in ("low", "moderate", "high"):
        group = np.array([r.realised_r for r in with_r if r.confidence == level], dtype="float64")
        calibration[level] = {"trades": int(len(group)),
                              "average_r": float(group.mean()) if len(group) else None,
                              "hit_rate": float((group > 0).mean()) if len(group) else None}
    summary["calibration_by_confidence"] = calibration
    patterns = []
    stops = [r for r in closed if r.outcome == "stop"]
    recovered = [r for r in stops if any("closed above the entry" in n for n in r.notes)]
    if recovered:
        patterns.append(f"{len(recovered)} of {len(stops)} stop-outs recovered above the entry within "
                        f"{RECOVERY_SESSIONS} sessions: stops may sit inside normal noise.")
    gapped = [r for r in stops if r.slippage_vs_assumed is not None and r.slippage_vs_assumed > 0.05]
    if gapped:
        patterns.append(f"{len(gapped)} stop-out(s) filled worse than the stop by more than 0.05R (gaps): size with a "
                        "gap allowance.")
    long_holds = [r for r in closed if r.expected_holding_days and r.holding_days is not None
                  and r.holding_days > 2 * r.expected_holding_days]
    if long_holds:
        patterns.append(f"{len(long_holds)} trade(s) were held more than twice as long as planned.")
    ordered = [calibration[k]["average_r"] for k in ("low", "moderate", "high") if calibration[k]["trades"]]
    if len(ordered) >= 2 and any(b < a for a, b in zip(ordered, ordered[1:])):
        patterns.append("Higher stated confidence did not go with higher realised R: confidence is not calibrated.")
    summary["patterns"] = patterns
    warnings = []
    if len(with_r) < MIN_TRADES:
        warnings.append(f"Only {len(with_r)} closed trade(s) with a defined R: below {MIN_TRADES}, hit rate, "
                        "expectancy and calibration are dominated by chance.")
    if any(r.confidence is None for r in reviews):
        warnings.append("Some trades have no recorded thesis and are excluded from R statistics.")
    summary["warnings"] = warnings
    return reviews, clean_json(summary)
