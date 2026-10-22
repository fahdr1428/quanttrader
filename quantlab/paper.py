"""PAPER TRADE: a simulated order book against daily bars. SIMULATED ONLY.

Nothing here can reach a broker: orders are rows in the local research database
(``paper_orders`` / ``paper_fills``, every row marked simulated by a CHECK
constraint) and fills are computed from end-of-day bars the caller supplies.

Fill rules (daily bars, so intraday order is unknown and the rules lean pessimistic):

* An order can fill only on a bar strictly after its ``decision_date`` — the session
  whose data the decision was based on. No fill uses information from that bar.
* market: at the next open, plus slippage against the trader.
* limit buy: when ``low <= limit``, at ``min(open, limit)``; limit sell (a target):
  when ``high >= limit``, at ``max(open, limit)``. No slippage, and touching the limit
  is assumed to fill (optimistic about queue position; stated in the results).
* stop sell: when ``low <= stop``, at ``min(open, stop)`` minus slippage — a gap
  opens through the stop and fills there, not at the stop.
* When a buy fills, the thesis's protective stop and first target are attached as
  one-cancels-other sell orders. On the entry bar only the stop is checked (the bar's
  high may have come before the fill); on later bars, a bar that reaches both the stop
  and the target counts as stopped out.
* Long only, no margin: a sell beyond the position or a buy beyond the cash is refused.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.experiments import ExperimentStore, check_not_sensitive
from quantlab.risk import LimitCheck, PortfolioState, Position
from quantlab.serialization import clean_json
from quantlab.validation import validate_ticker

NEW_YORK = ZoneInfo("America/New_York")
SIMULATED_LABEL = "SIMULATED paper trade — no real order was sent"
CONFIDENCE_LEVELS = ("low", "moderate", "high")
_ROLE_ORDER = {"stop": 0, "target": 1, "exit": 2, "entry": 3}


@dataclass(frozen=True)
class Thesis:
    """The plan recorded with an entry, so the trade can be reviewed against it later."""

    direction: str
    summary: str
    entry_rationale: str
    stop: float
    targets: tuple[float, ...]
    invalidation: str
    expected_holding_days: int
    confidence: str

    def __post_init__(self) -> None:
        if self.direction != "long":
            raise ValueError("Only long theses can be paper traded (no short selling).")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"confidence must be one of {', '.join(CONFIDENCE_LEVELS)}")
        if not (math.isfinite(self.stop) and self.stop > 0):
            raise ValueError("The thesis needs a positive stop.")
        object.__setattr__(self, "targets", tuple(float(t) for t in self.targets))
        if any(not math.isfinite(t) or t <= self.stop for t in self.targets):
            raise ValueError("Targets must be above the stop.")
        if not isinstance(self.expected_holding_days, int) or self.expected_holding_days < 1:
            raise ValueError("expected_holding_days must be a positive whole number of sessions")
        for name in ("summary", "entry_rationale", "invalidation"):
            text = getattr(self, name)
            if not isinstance(text, str) or len(text) > 2000:
                raise ValueError(f"{name} must be text of at most 2000 characters")
            check_not_sensitive(text, name)
        if not self.invalidation.strip():
            raise ValueError("State what would invalidate the thesis.")

    def to_dict(self) -> dict[str, Any]:
        return clean_json(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Thesis | None:
        if not text:
            return None
        data = json.loads(text)
        data["targets"] = tuple(data.get("targets") or ())
        return cls(**data)


@dataclass(frozen=True)
class PaperOrder:
    id: int
    account: str
    client_order_id: str | None
    ticker: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None
    stop_price: float | None
    status: str
    role: str
    parent_id: int | None
    created_at: str
    decision_date: str
    thesis: Thesis | None
    note: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if k != "thesis"}
        out["thesis"] = self.thesis.to_dict() if self.thesis else None
        out["simulated"] = True
        out["label"] = SIMULATED_LABEL
        return clean_json(out)


@dataclass(frozen=True)
class PaperFill:
    id: int
    order_id: int
    ticker: str
    side: str
    quantity: float
    price: float
    reference_price: float
    fee: float
    session: str

    def to_dict(self) -> dict[str, Any]:
        return clean_json({**asdict(self), "simulated": True})


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


class PaperBook:
    """A simulated account in the research database. SIMULATED ONLY — never connects to a broker."""

    def __init__(self, store: ExperimentStore, account: str = "default", starting_cash: float = 100_000.0,
                 fee_rate: float = 0.001, slippage_rate: float = 0.0005, clock: Callable[[], datetime] | None = None):
        if not account or len(account) > 60 or not account.replace("-", "").replace("_", "").isalnum():
            raise ValueError("account must be 1-60 letters, digits, '-' or '_'")
        if not 0 <= fee_rate < 0.05 or not 0 <= slippage_rate < 0.05:
            raise ValueError("fee_rate and slippage_rate must be in [0, 0.05)")
        self.store, self.conn, self.account = store, store.conn, account
        self.fee_rate, self.slippage_rate = fee_rate, slippage_rate
        self._clock = clock or _now_utc
        row = self.conn.execute("SELECT starting_cash FROM paper_accounts WHERE account = ?", (account,)).fetchone()
        if row is None:
            with self.conn:
                self.conn.execute("INSERT INTO paper_accounts (account, starting_cash, created_at) VALUES (?, ?, ?)",
                                  (account, _positive(starting_cash, "starting_cash"), self._clock().isoformat()))
            self.starting_cash = float(starting_cash)
        else:
            self.starting_cash = float(row["starting_cash"])

    # -- reading the ledger ------------------------------------------------------------------- #

    def _order(self, row: Any) -> PaperOrder:
        return PaperOrder(id=row["id"], account=row["account"], client_order_id=row["client_order_id"],
                          ticker=row["ticker"], side=row["side"], quantity=row["quantity"],
                          order_type=row["order_type"], limit_price=row["limit_price"], stop_price=row["stop_price"],
                          status=row["status"], role=row["role"], parent_id=row["parent_id"],
                          created_at=row["created_at"], decision_date=row["decision_date"],
                          thesis=Thesis.from_json(row["thesis_json"]), note=row["note"], reason=row["reason"])

    def get_order(self, order_id: int) -> PaperOrder | None:
        row = self.conn.execute("SELECT * FROM paper_orders WHERE id = ? AND account = ?",
                                (order_id, self.account)).fetchone()
        return None if row is None else self._order(row)

    def order_list(self, ticker: str | None = None, status: str | None = None) -> list[PaperOrder]:
        sql, args = "SELECT * FROM paper_orders WHERE account = ?", [self.account]
        if ticker:
            sql, args = sql + " AND ticker = ?", [*args, ticker]
        if status:
            sql, args = sql + " AND status = ?", [*args, status]
        return [self._order(r) for r in self.conn.execute(sql + " ORDER BY id", args).fetchall()]

    def fill_list(self, ticker: str | None = None) -> list[PaperFill]:
        sql, args = "SELECT * FROM paper_fills WHERE account = ?", [self.account]
        if ticker:
            sql, args = sql + " AND ticker = ?", [*args, ticker]
        rows = self.conn.execute(sql + " ORDER BY session, id", args).fetchall()
        return [PaperFill(r["id"], r["order_id"], r["ticker"], r["side"], r["quantity"], r["price"],
                          r["reference_price"], r["fee"], r["session"]) for r in rows]

    def orders(self) -> pd.DataFrame:
        return pd.DataFrame([o.to_dict() for o in self.order_list()])

    def fills(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.fill_list()])

    def cash(self) -> float:
        cash = self.starting_cash
        for f in self.fill_list():
            cash += (-1 if f.side == "buy" else 1) * f.quantity * f.price - f.fee
        return cash

    def held(self, ticker: str) -> float:
        return sum((1 if f.side == "buy" else -1) * f.quantity for f in self.fill_list(ticker))

    def positions(self) -> pd.DataFrame:
        """Open positions with average cost (fees included in the cost) and realised P&L by ticker."""
        state: dict[str, dict[str, float]] = {}
        for f in self.fill_list():
            s = state.setdefault(f.ticker, {"units": 0.0, "cost": 0.0, "realised_pnl": 0.0})
            if f.side == "buy":
                s["units"] += f.quantity
                s["cost"] += f.quantity * f.price + f.fee
            else:
                avg = s["cost"] / s["units"] if s["units"] else 0.0
                s["realised_pnl"] += f.quantity * (f.price - avg) - f.fee
                s["cost"] -= avg * f.quantity
                s["units"] -= f.quantity
        rows = [{"ticker": t, "units": s["units"], "avg_cost": s["cost"] / s["units"] if s["units"] > 1e-12 else None,
                 "realised_pnl": s["realised_pnl"], "simulated": True} for t, s in state.items()]
        return pd.DataFrame(rows, columns=["ticker", "units", "avg_cost", "realised_pnl", "simulated"])

    def _reserved_cash(self) -> float:
        """Cash committed to open limit buys (at their limit, with fees)."""
        return sum(o.quantity * o.limit_price * (1 + self.fee_rate) for o in self.order_list(status="open")
                   if o.side == "buy" and o.order_type == "limit")

    def _entry_open_units(self, entry: PaperOrder, include_open_exits: bool = False) -> float:
        """Units of a filled entry not yet sold by its own exits (optionally net of its open manual exits)."""
        fills = self.fill_list(entry.ticker)
        bought = sum(f.quantity for f in fills if f.order_id == entry.id)
        children = {o.id: o for o in self.order_list(entry.ticker) if o.parent_id == entry.id and o.side == "sell"}
        sold = sum(f.quantity for f in fills if f.order_id in children)
        pending = sum(o.quantity for o in children.values() if include_open_exits and o.status == "open"
                      and o.role == "exit")
        return max(0.0, bought - sold - pending)

    # -- placing and cancelling ------------------------------------------------------------------ #

    def submit(self, ticker: str, side: str, quantity: float, order_type: str = "market", limit: float | None = None,
               stop: float | None = None, thesis: Thesis | None = None, client_order_id: str | None = None,
               risk_checks: list[LimitCheck] | None = None, decision_date: date | str | None = None,
               note: str = "", parent_id: int | None = None) -> PaperOrder:
        """Record a SIMULATED order. A failed risk check, a short sale or a buy beyond cash is REJECTED (and kept).

        Orders are decided today (the clock's New York date): a past ``decision_date`` is refused, since
        it would let the order fill on bars that were already known.
        """
        symbol = validate_ticker(ticker)
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell' (sell only reduces a long position)")
        if order_type not in ("market", "limit", "stop"):
            raise ValueError("order_type must be market, limit or stop")
        quantity = _positive(quantity, "quantity")
        limit = _positive(limit, "limit") if order_type == "limit" else None
        stop = _positive(stop, "stop") if order_type == "stop" else None
        if client_order_id is not None:
            if not (0 < len(client_order_id) <= 64):
                raise ValueError("client_order_id must be 1-64 characters")
            row = self.conn.execute("SELECT * FROM paper_orders WHERE account = ? AND client_order_id = ?",
                                    (self.account, client_order_id)).fetchone()
            if row is not None:                  # idempotent: the same request never creates two orders
                same = (row["ticker"], row["side"], row["quantity"], row["order_type"], row["limit_price"],
                        row["stop_price"]) == (symbol, side, quantity, order_type, limit, stop)
                if not same:
                    raise ValueError(f"client_order_id {client_order_id!r} was already used for a different order.")
                return self._order(row)
        check_not_sensitive(note, "note")
        decided = self._decision_date(decision_date)
        reasons: list[str] = []
        if order_type == "stop" and side == "buy":
            reasons.append("Stop orders are protective sells only.")
        failed = [c for c in (risk_checks or []) if not c.ok]
        if failed:
            reasons.append("Risk checks failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed))
        if side == "buy" and thesis is not None and order_type == "limit" and thesis.stop >= limit:
            reasons.append("The thesis stop is not below the limit price.")
        if side == "sell":
            open_sells = sum(o.quantity for o in self.order_list(symbol, "open") if o.side == "sell"
                             and o.role not in ("stop", "target"))
            if quantity + open_sells > self.held(symbol) + 1e-9:
                reasons.append(f"Sell of {quantity:g} exceeds the simulated position of {self.held(symbol):g} "
                               "(no short selling).")
            elif parent_id is None:
                # A manual exit belongs to the oldest entry that still has open units (first in, first out),
                # so the review matches it to that entry's thesis.
                for entry in self.order_list(symbol):
                    if entry.side == "buy" and entry.status == "filled":
                        remaining = self._entry_open_units(entry, include_open_exits=True)
                        if remaining > 1e-9:
                            parent_id = entry.id
                            if quantity > remaining + 1e-9:
                                reasons.append(f"Sell at most {remaining:g}, the open units of entry {entry.id}; "
                                               "sell the rest in a separate order.")
                            break
        elif order_type == "limit":
            needed = quantity * limit * (1 + self.fee_rate)
            free = self.cash() - self._reserved_cash()
            if needed > free + 1e-9:
                reasons.append(f"Cost {needed:,.2f} exceeds free simulated cash {free:,.2f} (no margin).")
        status = "rejected" if reasons else "open"
        role = "entry" if side == "buy" else "exit"
        return self._insert(symbol, side, quantity, order_type, limit, stop, status, role, parent_id, decided,
                            thesis, note, " ".join(reasons), client_order_id)

    def _decision_date(self, value: date | str | None) -> str:
        today = self._clock().astimezone(NEW_YORK).date()
        if value is None:
            return today.isoformat()
        decided = date.fromisoformat(value) if isinstance(value, str) else pd.Timestamp(value).date()
        if decided < today:
            raise ValueError(f"A simulated order is decided today ({today}); a past decision date ({decided}) would "
                             "fill on bars that were already known.")
        return decided.isoformat()

    def _insert(self, ticker: str, side: str, quantity: float, order_type: str, limit: float | None,
                stop: float | None, status: str, role: str, parent_id: int | None, decided: str,
                thesis: Thesis | None, note: str, reason: str, client_order_id: str | None = None) -> PaperOrder:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO paper_orders (account, client_order_id, ticker, side, quantity, order_type, limit_price, "
                "stop_price, status, role, parent_id, created_at, decision_date, thesis_json, note, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self.account, client_order_id, ticker, side, quantity, order_type, limit, stop, status, role,
                 parent_id, self._clock().isoformat(), decided,
                 json.dumps(thesis.to_dict()) if thesis else "", note, reason))
        return self.get_order(int(cursor.lastrowid))

    def cancel(self, order_id: int, reason: str = "cancelled by the user") -> PaperOrder:
        order = self.get_order(order_id)
        if order is None:
            raise KeyError(f"No simulated order {order_id} in account {self.account}")
        if order.status != "open":
            raise ValueError(f"Order {order_id} is {order.status}; only open orders can be cancelled.")
        self._set_status(order_id, "cancelled", reason)
        return self.get_order(order_id)

    def _set_status(self, order_id: int, status: str, reason: str = "") -> None:
        with self.conn:
            self.conn.execute("UPDATE paper_orders SET status = ?, reason = ? WHERE id = ?", (status, reason, order_id))

    # -- simulation ------------------------------------------------------------------------------ #

    def evaluated_through(self, ticker: str) -> str | None:
        row = self.conn.execute("SELECT through FROM paper_progress WHERE account = ? AND ticker = ?",
                                (self.account, ticker)).fetchone()
        return None if row is None else row["through"]

    def process(self, ticker: str, prices: pd.DataFrame) -> list[PaperFill]:
        """Fill this ticker's open orders against ``prices`` (validated daily OHLC bars), in date order."""
        return self.process_all({ticker: prices})

    def process_all(self, prices_by_ticker: dict[str, pd.DataFrame]) -> list[PaperFill]:
        """Fill open orders session by session across every ticker given, forward only.

        Each ticker's bars are evaluated once: a bar on or before the session already evaluated
        for that ticker is skipped, so later data revisions (back-adjusted dividends, corrected
        bars) never fill an order retroactively.
        """
        bars: dict[str, list[tuple[str, str, dict[str, float]]]] = {}
        for raw_ticker, prices in prices_by_ticker.items():
            symbol = validate_ticker(raw_ticker)
            if not prices.index.is_monotonic_increasing:
                raise ValueError("prices must be sorted by date")
            through = self.evaluated_through(symbol)
            for stamp, bar in prices.iterrows():
                session = pd.Timestamp(stamp).date().isoformat()
                if through is not None and session <= through:
                    continue
                values = {k: float(bar[k]) for k in ("open", "high", "low", "close")}
                bars.setdefault(session, []).append((symbol, session, values))
        fills: list[PaperFill] = []
        for session in sorted(bars):
            for symbol, _, values in bars[session]:
                if all(math.isfinite(v) and v > 0 for v in values.values()):   # never fill on an invalid bar
                    fills.extend(self._process_bar(symbol, session, values))
                with self.conn:
                    self.conn.execute("INSERT INTO paper_progress (account, ticker, through) VALUES (?, ?, ?) "
                                      "ON CONFLICT(account, ticker) DO UPDATE SET through = excluded.through",
                                      (self.account, symbol, session))
        return fills

    def _process_bar(self, symbol: str, session: str, bar: dict[str, float]) -> list[PaperFill]:
        def priority(o: PaperOrder) -> tuple[int, int]:
            # A bar that opens at or beyond a target fills it at the open, before anything later in the day;
            # otherwise the stop goes first (pessimistic when one bar reaches both levels).
            if o.role == "target" and o.limit_price is not None and bar["open"] >= o.limit_price:
                return (-1, o.id)
            return (_ROLE_ORDER.get(o.role, 9), o.id)

        fills: list[PaperFill] = []
        pending = sorted((o for o in self.order_list(symbol, "open") if o.decision_date < session), key=priority)
        for order in pending:
            current = self.get_order(order.id)
            if current is None or current.status != "open":      # an OCO sibling filled on this bar
                continue
            fill = self._try_fill(current, bar, session)
            if fill is not None:
                fills.append(fill)
                fills.extend(self._after_fill(current, fill, bar, session))
        return fills

    def _fill_price(self, order: PaperOrder, bar: dict[str, float]) -> float | None:
        o, h, lo = bar["open"], bar["high"], bar["low"]
        if order.order_type == "market":
            return o
        if order.order_type == "limit":
            if order.side == "buy":
                return min(o, order.limit_price) if lo <= order.limit_price else None
            return max(o, order.limit_price) if h >= order.limit_price else None
        return min(o, order.stop_price) if lo <= order.stop_price else None   # stop sell

    def _cash_never_negative(self, cost: float, session: str) -> bool:
        """True when paying ``cost`` on ``session`` keeps cash >= 0 on that session and every later one
        already in the ledger (fills of other tickers may be dated later)."""
        cash = self.starting_cash
        timeline: dict[str, float] = {}
        for f in self.fill_list():
            cash += (-1 if f.side == "buy" else 1) * f.quantity * f.price - f.fee
            timeline[f.session] = cash
        before = [v for d, v in timeline.items() if d <= session]
        at_session = before[-1] if before else self.starting_cash
        later = [v for d, v in timeline.items() if d > session]
        return min([at_session, *later]) - cost >= -1e-9

    def _try_fill(self, order: PaperOrder, bar: dict[str, float], session: str) -> PaperFill | None:
        reference = self._fill_price(order, bar)
        if reference is None:
            return None
        slip = self.slippage_rate if order.order_type in ("market", "stop") else 0.0
        price = reference * (1 + slip) if order.side == "buy" else reference * (1 - slip)
        quantity = order.quantity
        if order.side == "sell":
            quantity = min(quantity, self.held(order.ticker))
            if quantity <= 1e-12:
                self._set_status(order.id, "cancelled", "No simulated position left to sell.")
                return None
        fee = quantity * price * self.fee_rate
        if order.side == "buy" and not self._cash_never_negative(quantity * price + fee, session):
            self._set_status(order.id, "rejected", f"Insufficient simulated cash at the fill on {session} (no margin).")
            return None
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO paper_fills (order_id, account, ticker, side, quantity, price, reference_price, fee, "
                "session, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (order.id, self.account, order.ticker, order.side, quantity, price, reference, fee, session,
                 self._clock().isoformat()))
            self.conn.execute("UPDATE paper_orders SET status = 'filled' WHERE id = ?", (order.id,))
        return PaperFill(int(cursor.lastrowid), order.id, order.ticker, order.side, quantity, price, reference, fee,
                         session)

    def _after_fill(self, order: PaperOrder, fill: PaperFill, bar: dict[str, float], session: str) -> list[PaperFill]:
        """Attach the thesis's stop and target to a filled entry; after an exit, shrink or cancel the entry's
        remaining protective orders."""
        out: list[PaperFill] = []
        if order.side == "sell":
            entry = self.get_order(order.parent_id) if order.parent_id is not None else None
            remaining = self._entry_open_units(entry) if entry is not None else 0.0
            flat = self.held(order.ticker) <= 1e-12
            for sibling in self.order_list(order.ticker, "open"):
                if sibling.side != "sell":
                    continue
                if entry is not None and sibling.parent_id == entry.id and sibling.role in ("stop", "target"):
                    if remaining <= 1e-9:
                        self._set_status(sibling.id, "cancelled", f"One-cancels-other: order {order.id} filled.")
                    elif sibling.quantity > remaining + 1e-9:
                        with self.conn:   # a partial exit: keep protecting the units that remain
                            self.conn.execute("UPDATE paper_orders SET quantity = ? WHERE id = ?",
                                              (remaining, sibling.id))
                elif flat and sibling.role in ("stop", "target"):
                    # Protective orders must never outlive their position and act on a later one.
                    self._set_status(sibling.id, "cancelled", f"Position closed by order {order.id}.")
        if order.side != "buy" or order.thesis is None:
            return out
        # Children carry the entry session as their decision date, so they fill on later bars; the stop is
        # also checked on the entry bar itself, after the fill (pessimistic: the low may have come later).
        stop = self._insert(order.ticker, "sell", fill.quantity, "stop", None, order.thesis.stop, "open", "stop",
                            order.id, session, None, "protective stop from the thesis", "")
        if order.thesis.targets:
            self._insert(order.ticker, "sell", fill.quantity, "limit", order.thesis.targets[0], None, "open", "target",
                         order.id, session, None, "first target from the thesis", "")
        same_bar = self._try_fill(stop, {**bar, "open": min(bar["open"], fill.reference_price)}, session)
        if same_bar is not None:
            out.append(same_bar)
            out.extend(self._after_fill(self.get_order(stop.id), same_bar, bar, session))
        return out

    # -- valuation -------------------------------------------------------------------------------- #

    def equity_curve(self, prices_by_ticker: dict[str, pd.DataFrame]) -> pd.Series:
        """Simulated account value at each close from the first fill: cash + units × close.

        A position is valued at its latest close on a date its own market did not trade.
        """
        fills = self.fill_list()
        if not fills:
            return pd.Series(dtype="float64", name="simulated_equity")
        start = pd.Timestamp(min(f.session for f in fills))
        closes = pd.DataFrame({t: p["close"] for t, p in prices_by_ticker.items()}).sort_index()
        closes = closes[closes.index >= start].ffill()
        values = []
        for stamp in closes.index:
            day = stamp.date().isoformat()
            cash = self.starting_cash
            units: dict[str, float] = {}
            for f in fills:
                if f.session > day:
                    continue
                sign = 1 if f.side == "buy" else -1
                cash += -sign * f.quantity * f.price - f.fee
                units[f.ticker] = units.get(f.ticker, 0.0) + sign * f.quantity
            held = {t: u for t, u in units.items() if abs(u) > 1e-12}
            missing = [t for t in held if t not in closes.columns or pd.isna(closes.at[stamp, t])]
            values.append(math.nan if missing else cash + sum(u * closes.at[stamp, t] for t, u in held.items()))
        return pd.Series(values, index=closes.index, name="simulated_equity").dropna()

    def portfolio_state(self, prices_by_ticker: dict[str, pd.DataFrame],
                        sectors: dict[str, str] | None = None) -> PortfolioState:
        """The simulated account as a :class:`quantlab.risk.PortfolioState` for limit checks.

        Open buy orders count as exposure (at their limit, or the latest close for market orders) and
        their estimated cost is taken from cash, so orders that have not filled yet cannot be used to
        exceed a limit. The equity peak includes the starting cash.
        """
        curve = self.equity_curve(prices_by_ticker)
        sectors = sectors or {}

        def last_close(ticker: str) -> float:
            frame = prices_by_ticker.get(ticker)
            if frame is None or frame.empty:
                raise ValueError(f"No prices supplied for the simulated account's {ticker}.")
            return float(frame["close"].iloc[-1])

        positions = []
        for row in self.positions().itertuples():
            if row.units > 1e-12:
                positions.append(Position(row.ticker, row.units, row.avg_cost, last_close(row.ticker),
                                          sectors.get(row.ticker)))
        cash = self.cash()
        equity = cash + sum(p.value for p in positions)
        pending_cost = 0.0
        for o in self.order_list(status="open"):
            if o.side == "buy":
                reference = o.limit_price if o.order_type == "limit" else last_close(o.ticker)
                positions.append(Position(o.ticker, o.quantity, reference, reference, sectors.get(o.ticker)))
                pending_cost += o.quantity * reference * (1 + self.fee_rate)
        fills = self.fill_list()
        latest = max((pd.Timestamp(p.index[-1]).date().isoformat() for p in prices_by_ticker.values() if len(p)),
                     default=None)
        today_pnl = 0.0 if latest is None else (self._realised_through(fills, "9999-12-31")
                                                - self._realised_through(fills, latest))
        start_of_day = float(curve.iloc[-2]) if len(curve) >= 2 else self.starting_cash
        peak = max([self.starting_cash, equity, *([float(curve.max())] if len(curve) else [])])
        return PortfolioState(equity=equity, cash=max(cash - pending_cost, 0.0), positions=positions,
                              peak_equity=peak, realised_pnl_today=today_pnl, start_of_day_equity=start_of_day)

    @staticmethod
    def _realised_through(fills: Iterable[PaperFill], last_session: str) -> float:
        """Realised P&L from fills strictly before ``last_session`` (average-cost method)."""
        state: dict[str, list[float]] = {}
        realised = 0.0
        for f in fills:
            if f.session >= last_session:
                continue
            units, cost = state.setdefault(f.ticker, [0.0, 0.0])
            if f.side == "buy":
                state[f.ticker] = [units + f.quantity, cost + f.quantity * f.price + f.fee]
            else:
                avg = cost / units if units else 0.0
                realised += f.quantity * (f.price - avg) - f.fee
                state[f.ticker] = [units - f.quantity, cost - avg * f.quantity]
        return realised
