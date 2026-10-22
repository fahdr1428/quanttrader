"""Simulated paper trading and post-trade review on hand-made daily bars (no broker, no network)."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from quantlab.experiments import _MIGRATIONS, SCHEMA_VERSION, ExperimentStore
from quantlab.paper import PaperBook, Thesis
from quantlab.postmortem import review_all, review_trade
from quantlab.risk import LimitCheck

SLIP, FEE = 0.0005, 0.001


class Clock:
    """The simulated wall clock: orders are decided on the clock's (New York) date."""

    def __init__(self, day: str):
        self.set(day)

    def set(self, day: str) -> None:
        self.now = datetime.fromisoformat(f"{day}T21:00:00+00:00")     # 16:00 in New York

    def __call__(self) -> datetime:
        return self.now


def bars(*rows, start="2024-01-02"):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.bdate_range(start, periods=len(rows)))


def thesis(stop=95.0, targets=(110.0,), days=5, confidence="moderate"):
    return Thesis("long", "Trend continuation", "Close above resistance", stop, targets,
                  "A close below the 50-day average", days, confidence)


def book(cash=10_000.0, day="2024-01-02"):
    clock = Clock(day)
    b = PaperBook(ExperimentStore.open_memory(), starting_cash=cash, fee_rate=FEE, slippage_rate=SLIP, clock=clock)
    b.clock = clock
    return b


def test_market_entry_fills_at_the_next_open_and_the_target_closes_it():
    b = book()
    entry = b.submit("acme", "buy", 10, thesis=thesis())
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (104, 111, 101, 109), (108, 109, 107, 108))
    fills = b.process("ACME", prices)
    buy, sell = fills
    assert (buy.session, buy.reference_price) == ("2024-01-03", 101)              # not on the decision bar
    assert buy.price == pytest.approx(101 * (1 + SLIP)) and buy.fee == pytest.approx(10 * buy.price * FEE)
    assert (sell.session, sell.price) == ("2024-01-04", 110)                       # limit: no slippage
    statuses = {o.role: o.status for o in b.order_list()}
    assert statuses == {"entry": "filled", "target": "filled", "stop": "cancelled"}  # one-cancels-other
    assert b.cash() == pytest.approx(10_000 - 10 * buy.price - buy.fee + 1100 - sell.fee)
    assert b.held("ACME") == 0 and b.positions().loc[0, "realised_pnl"] == pytest.approx(b.cash() - 10_000)

    r = review_trade(b, entry.id, prices)
    risk = buy.price - 95
    pnl = 10 * (110 - buy.price) - buy.fee - sell.fee
    assert r.outcome == "target" and r.holding_days == 1 and r.pnl == pytest.approx(pnl)
    assert r.realised_r == pytest.approx(pnl / (10 * risk))
    assert r.mfe_r == pytest.approx((111 - buy.price) / risk) and r.mae_r == pytest.approx((100 - buy.price) / risk)
    assert r.slippage_vs_assumed == pytest.approx(0.0)


def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    b = book()
    entry = b.submit("ACME", "buy", 10, thesis=thesis())
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (90, 92, 89, 91), (95, 100, 94, 99),
                  (100, 106, 99, 105))
    b.process("ACME", prices)
    stop_fill = b.fill_list()[-1]
    assert stop_fill.reference_price == 90 and stop_fill.price == pytest.approx(90 * (1 - SLIP))
    r = review_trade(b, entry.id, prices)
    assert r.outcome == "stop" and r.slippage_vs_assumed > 0.8                    # about a full R worse
    assert any("Gapped through the stop" in n for n in r.notes)
    assert any("closed above the entry" in n for n in r.notes)


def test_a_bar_hitting_both_stop_and_target_counts_as_stopped_and_the_entry_bar_checks_the_stop():
    both = book()
    both.submit("ACME", "buy", 10, thesis=thesis())
    both.process("ACME", bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 112, 94, 100)))
    assert {o.role: o.status for o in both.order_list()}["stop"] == "filled"
    same = book()
    same.submit("ACME", "buy", 10, thesis=thesis())
    fills = same.process("ACME", bars((100, 101, 99, 100), (101, 112, 94, 96)))
    assert [f.session for f in fills] == ["2024-01-03", "2024-01-03"]
    assert fills[1].reference_price == 95                                           # stop, not the 112 high


def test_limit_buys_fill_at_the_better_of_open_and_limit():
    b = book()
    first = b.submit("ACME", "buy", 5, "limit", limit=98.0)
    prices = bars((100, 101, 99, 100), (99, 100, 97, 98))
    b.process("ACME", prices)
    assert b.get_order(first.id).status == "filled" and b.fill_list()[0].price == 98
    b.clock.set("2024-01-03")
    second = b.submit("ACME", "buy", 5, "limit", limit=98.0)
    b.process("ACME", bars((96, 97, 95, 96), start="2024-01-04"))
    assert b.get_order(second.id).status == "filled" and b.fill_list()[-1].price == 96   # gapped below the limit


def test_the_book_refuses_shorts_margin_failed_checks_and_duplicates():
    b = book(cash=1_000)
    failed = [LimitCheck("daily loss", False, 0.03, 0.02, "loss limit reached")]
    assert b.submit("ACME", "buy", 1, risk_checks=failed).status == "rejected"
    assert "daily loss" in b.order_list()[0].reason
    assert b.submit("ACME", "sell", 1).status == "rejected"                          # nothing held: no shorting
    assert "no margin" in b.submit("ACME", "buy", 100, "limit", limit=50.0).reason
    big = b.submit("ACME", "buy", 100)                    # checked at the fill
    b.process("ACME", bars((100, 101, 99, 100), (100, 101, 99, 100)))
    assert b.get_order(big.id).status == "rejected" and "no margin" in b.get_order(big.id).reason
    once = b.submit("ACME", "buy", 1, client_order_id="idem-1")
    again = b.submit("ACME", "buy", 1, client_order_id="idem-1")
    assert once.id == again.id and len(b.order_list()) == 5
    with pytest.raises(ValueError):
        b.submit("ACME", "short", 1)
    with pytest.raises(ValueError, match="short"):
        Thesis("short", "x", "x", 95.0, (), "x", 5, "low")
    assert b.cancel(once.id).status == "cancelled"
    with pytest.raises(ValueError):
        b.cancel(once.id)


def test_every_ledger_row_is_simulated_by_construction():
    b = book()
    b.submit("ACME", "buy", 1)
    with pytest.raises(sqlite3.IntegrityError):
        with b.conn:
            b.conn.execute("UPDATE paper_orders SET simulated = 0")
    for table in ("paper_orders", "paper_accounts"):
        assert {r[0] for r in b.conn.execute(f"SELECT simulated FROM {table}")} == {1}


def test_equity_curve_and_portfolio_state_mark_positions_to_the_close():
    b = book()
    b.submit("ACME", "buy", 10)
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103))
    b.process("ACME", prices)
    buy = b.fill_list()[0]
    curve = b.equity_curve({"ACME": prices})
    cash = 10_000 - 10 * buy.price - buy.fee
    assert list(curve.index.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04"]
    assert curve.iloc[-1] == pytest.approx(cash + 10 * 103)
    state = b.portfolio_state({"ACME": prices}, {"ACME": "Tech"})
    assert state.equity == pytest.approx(cash + 1030) and state.positions[0].sector == "Tech"
    assert state.realised_pnl_today == 0


def test_review_all_summarises_r_calibration_and_sample_size():
    b = book(cash=100_000)
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (104, 111, 101, 109), (108, 109, 107, 108),
                  (90, 92, 89, 91), (95, 100, 94, 99))
    b.submit("WIN", "buy", 10, thesis=thesis(confidence="low"))
    b.submit("BARE", "buy", 1)
    b.clock.set("2024-01-04")
    b.submit("LOSS", "buy", 10, thesis=thesis(stop=99.5, targets=(130.0,), confidence="high"))
    for t in ("WIN", "LOSS", "BARE"):
        b.process(t, prices)
    reviews, summary = review_all(b, {t: prices for t in ("WIN", "LOSS", "BARE")})
    outcomes = {r.ticker: r.outcome for r in reviews}
    assert outcomes == {"WIN": "target", "LOSS": "stop", "BARE": "open"}
    assert summary["closed_with_r"] == 2 and summary["hit_rate"] == 0.5
    assert summary["calibration_by_confidence"]["high"]["average_r"] < 0 < summary["calibration_by_confidence"]["low"]["average_r"]
    assert any("not calibrated" in p for p in summary["patterns"])
    assert any("below 20" in w for w in summary["warnings"]) and any("no recorded thesis" in w for w in summary["warnings"])
    bare = next(r for r in reviews if r.ticker == "BARE")
    assert bare.realised_r is None and "No thesis" in bare.notes[0]


def test_existing_research_databases_are_migrated_to_the_paper_schema(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE schema_version (version INTEGER NOT NULL);" + _MIGRATIONS[1])
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.execute("INSERT INTO notes (title, created_at, updated_at) VALUES ('kept', 'x', 'x')")
    conn.commit()
    conn.close()
    store = ExperimentStore.open_file(path)
    assert store.schema_version() == SCHEMA_VERSION == 3 and store.list_notes()[0].title == "kept"
    PaperBook(store, starting_cash=5_000).submit("ACME", "buy", 1)
    assert PaperBook(store, starting_cash=999).starting_cash == 5_000                # the account keeps its cash


def test_a_manual_exit_closes_the_trade_and_cancels_its_protective_orders():
    """Stops and targets never outlive their position and act on a later one in the same ticker."""
    b = book(cash=100_000)
    first = b.submit("ACME", "buy", 10, thesis=thesis())
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103), (103, 104, 102, 103),
                  (103, 104, 102, 103), (96, 97, 94, 96))
    b.process("ACME", prices.iloc[:2])
    b.clock.set("2024-01-03")
    exit_order = b.submit("ACME", "sell", 10, note="invalidated: lost the 50-day")
    assert exit_order.parent_id == first.id
    b.process("ACME", prices.iloc[:3])
    assert {o.role: o.status for o in b.order_list() if o.parent_id == first.id} == {
        "stop": "cancelled", "target": "cancelled", "exit": "filled"}
    b.clock.set("2024-01-04")
    second = b.submit("ACME", "buy", 5, thesis=thesis(stop=90.0))
    b.process("ACME", prices)
    assert b.held("ACME") == 5                          # the old stop at 95 did not sell the new position
    review = review_trade(b, first.id, prices)
    assert review.outcome == "invalidated" and review.exit == pytest.approx(102 * (1 - SLIP))
    assert review_trade(b, second.id, prices).outcome == "open"


def test_orders_cannot_be_back_dated_and_bars_are_evaluated_forward_only():
    """No look-ahead: an order decided today never fills on old bars, and revised history never refills."""
    b = book()
    with pytest.raises(ValueError, match="past decision date"):
        b.submit("ACME", "buy", 10, decision_date="2023-12-29")
    b.submit("ACME", "buy", 10, thesis=thesis())
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 96, 103))
    first = b.process("ACME", prices)
    assert [f.session for f in first] == ["2024-01-03"] and b.evaluated_through("ACME") == "2024-01-04"
    revised = prices * 0.98                                   # e.g. back-adjusted for a dividend: low 96 -> 94.08
    assert b.process("ACME", revised) == [] and b.held("ACME") == 10      # the old bar is not re-evaluated


def test_cash_is_checked_on_every_later_session_across_tickers():
    """Processing one ticker ahead cannot let its later sale fund another ticker's earlier buy (no margin)."""
    b = book(cash=10_000)
    b.submit("AAA", "buy", 90, thesis=thesis(stop=80.0, targets=(105.0,)))
    b.submit("BBB", "buy", 90)
    prices = bars((100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (104, 106, 103, 105))
    b.process("AAA", prices)                                 # buys on 01-03, sells at the target on 01-05
    b.process("BBB", prices)                                 # its 01-03 buy would need cash that came later
    bbb = next(o for o in b.order_list() if o.ticker == "BBB")
    assert bbb.status == "rejected" and "no margin" in bbb.reason
    assert all(v >= 0 for v in _cash_by_session(b).values())


def _cash_by_session(b):
    cash, out = b.starting_cash, {}
    for f in b.fill_list():
        cash += (-1 if f.side == "buy" else 1) * f.quantity * f.price - f.fee
        out[f.session] = cash
    return out


def test_pending_buys_count_as_exposure_and_the_peak_includes_starting_cash():
    b = book(cash=100_000)
    b.submit("ACME", "buy", 100)                               # open: not filled yet
    prices = bars((100, 101, 99, 100))
    state = b.portfolio_state({"ACME": prices})
    assert state.holding_value("ACME") == pytest.approx(100 * 100)
    assert state.cash == pytest.approx(100_000 - 100 * 100 * (1 + FEE))
    loser = book(cash=100_000)
    loser.submit("ACME", "buy", 900)
    crash = bars((100, 101, 99, 100), (100, 100, 80, 82))
    loser.process("ACME", crash)
    s2 = loser.portfolio_state({"ACME": crash})
    assert s2.peak_equity == 100_000 and s2.drawdown() == pytest.approx(1 - s2.equity / 100_000)
    assert s2.drawdown() > 0.15


def test_a_partial_exit_keeps_the_remaining_units_protected_and_exits_attach_to_open_entries():
    b = book(cash=100_000)
    entry = b.submit("ACME", "buy", 10, thesis=thesis())
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103), (70, 71, 60, 61))
    b.process("ACME", prices.iloc[:2])
    b.clock.set("2024-01-03")
    part = b.submit("ACME", "sell", 4)
    assert part.parent_id == entry.id
    too_many = b.submit("ACME", "sell", 7)
    assert too_many.status == "rejected" and "Sell of 7" in too_many.reason        # 6 left after the open exit
    b.process("ACME", prices)
    protective = {o.role: o for o in b.order_list() if o.parent_id == entry.id and o.role in ("stop", "target")}
    assert protective["stop"].status == "filled" and protective["stop"].quantity == 6
    assert b.held("ACME") == 0                                                       # the crash was stopped out
    review = review_trade(b, entry.id, prices)
    assert review.outcome == "stop" and review.exit is not None


def test_a_bar_opening_above_the_target_fills_the_target_at_the_open():
    b = book()
    b.submit("ACME", "buy", 10, thesis=thesis())
    fills = b.process("ACME", bars((100, 101, 99, 100), (101, 103, 100, 102), (115, 116, 94, 96)))
    assert (fills[-1].side, fills[-1].reference_price) == ("sell", 115)               # target first, at the open
    assert {o.role: o.status for o in b.order_list()}["stop"] == "cancelled"


def test_a_reused_client_order_id_must_describe_the_same_order():
    b = book()
    b.submit("ACME", "buy", 1, client_order_id="k1")
    with pytest.raises(ValueError, match="already used"):
        b.submit("XYZ", "sell", 50, "limit", limit=10.0, client_order_id="k1")
