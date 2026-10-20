"""Simulated paper trading and post-trade review on hand-made daily bars (no broker, no network)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from quantlab.experiments import _MIGRATIONS, SCHEMA_VERSION, ExperimentStore
from quantlab.paper import PaperBook, Thesis
from quantlab.postmortem import review_all, review_trade
from quantlab.risk import LimitCheck

NOW = datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)
SLIP, FEE = 0.0005, 0.001


def bars(*rows, start="2024-01-02"):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.bdate_range(start, periods=len(rows)))


def thesis(stop=95.0, targets=(110.0,), days=5, confidence="moderate"):
    return Thesis("long", "Trend continuation", "Close above resistance", stop, targets,
                  "A close below the 50-day average", days, confidence)


def book(cash=10_000.0):
    return PaperBook(ExperimentStore.open_memory(), starting_cash=cash, fee_rate=FEE, slippage_rate=SLIP,
                     clock=lambda: NOW)


def test_market_entry_fills_at_the_next_open_and_the_target_closes_it():
    b = book()
    entry = b.submit("acme", "buy", 10, thesis=thesis(), decision_date="2024-01-02")
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
    entry = b.submit("ACME", "buy", 10, thesis=thesis(), decision_date="2024-01-02")
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
    both.submit("ACME", "buy", 10, thesis=thesis(), decision_date="2024-01-02")
    both.process("ACME", bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 112, 94, 100)))
    assert {o.role: o.status for o in both.order_list()}["stop"] == "filled"
    same = book()
    same.submit("ACME", "buy", 10, thesis=thesis(), decision_date="2024-01-02")
    fills = same.process("ACME", bars((100, 101, 99, 100), (101, 112, 94, 96)))
    assert [f.session for f in fills] == ["2024-01-03", "2024-01-03"]
    assert fills[1].reference_price == 95                                           # stop, not the 112 high


def test_limit_buys_fill_at_the_better_of_open_and_limit():
    b = book()
    first = b.submit("ACME", "buy", 5, "limit", limit=98.0, decision_date="2024-01-02")
    prices = bars((100, 101, 99, 100), (99, 100, 97, 98))
    b.process("ACME", prices)
    assert b.get_order(first.id).status == "filled" and b.fill_list()[0].price == 98
    second = b.submit("ACME", "buy", 5, "limit", limit=98.0, decision_date="2024-01-03")
    b.process("ACME", bars((96, 97, 95, 96), start="2024-01-04"))
    assert b.get_order(second.id).status == "filled" and b.fill_list()[-1].price == 96   # gapped below the limit


def test_the_book_refuses_shorts_margin_failed_checks_and_duplicates():
    b = book(cash=1_000)
    failed = [LimitCheck("daily loss", False, 0.03, 0.02, "loss limit reached")]
    assert b.submit("ACME", "buy", 1, risk_checks=failed).status == "rejected"
    assert "daily loss" in b.order_list()[0].reason
    assert b.submit("ACME", "sell", 1).status == "rejected"                          # nothing held: no shorting
    assert "no margin" in b.submit("ACME", "buy", 100, "limit", limit=50.0).reason
    big = b.submit("ACME", "buy", 100, decision_date="2024-01-02")                    # checked at the fill
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
    assert all(o["simulated"] for o in b.orders().to_dict(orient="records"))


def test_equity_curve_and_portfolio_state_mark_positions_to_the_close():
    b = book()
    b.submit("ACME", "buy", 10, decision_date="2024-01-02")
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
    b.submit("WIN", "buy", 10, thesis=thesis(confidence="low"), decision_date="2024-01-02")
    b.submit("LOSS", "buy", 10, thesis=thesis(stop=99.5, targets=(130.0,), confidence="high"),
             decision_date="2024-01-04")
    b.submit("BARE", "buy", 1, decision_date="2024-01-02")
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
    assert store.schema_version() == SCHEMA_VERSION == 2 and store.list_notes()[0].title == "kept"
    PaperBook(store, starting_cash=5_000).submit("ACME", "buy", 1)
    assert PaperBook(store, starting_cash=999).starting_cash == 5_000                # the account keeps its cash


def test_a_manual_exit_closes_the_trade_and_cancels_its_protective_orders():
    """Stops and targets never outlive their position and act on a later one in the same ticker."""
    b = book(cash=100_000)
    first = b.submit("ACME", "buy", 10, thesis=thesis(), decision_date="2024-01-02")
    prices = bars((100, 101, 99, 100), (101, 103, 100, 102), (102, 104, 101, 103), (103, 104, 102, 103),
                  (103, 104, 102, 103), (96, 97, 94, 96))
    b.process("ACME", prices.iloc[:2])
    exit_order = b.submit("ACME", "sell", 10, decision_date="2024-01-03", note="invalidated: lost the 50-day")
    assert exit_order.parent_id == first.id
    b.process("ACME", prices.iloc[:3])
    assert {o.role: o.status for o in b.order_list() if o.parent_id == first.id} == {
        "stop": "cancelled", "target": "cancelled", "exit": "filled"}
    second = b.submit("ACME", "buy", 5, thesis=thesis(stop=90.0), decision_date="2024-01-04")
    b.process("ACME", prices)
    assert b.held("ACME") == 5                          # the old stop at 95 did not sell the new position
    review = review_trade(b, first.id, prices)
    assert review.outcome == "invalidated" and review.exit == pytest.approx(102 * (1 - SLIP))
    assert review_trade(b, second.id, prices).outcome == "open"
