"""Position sizing and limit checks, checked by hand. No leverage, no shorts, no overriding a failed check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantlab.risk import (
    LimitCheck,
    PortfolioState,
    Position,
    RiskLimits,
    TradeSetup,
    check_limits,
    gap_risk,
    illustrative_limits,
    risk_reward,
    size_position,
    trade_allowed,
)

LIMITS = RiskLimits(account_equity=100_000)
SETUP = TradeSetup("ACME", entry=100.0, stop=95.0, targets=(110.0, 90.0), sector="Tech")
# entry_eff 100.05, stop_eff 94.9525, fees 0.001 × 195.0025: loss per unit 5.2925025
LOSS = 100.05 - 94.9525 + 0.001 * (100.05 + 94.9525)


def test_size_is_the_risk_budget_divided_by_the_loss_per_unit_at_the_stop():
    s = size_position(SETUP, LIMITS)
    assert s.loss_per_unit_at_stop == pytest.approx(LOSS)
    assert s.risk_budget == 500 and s.units == 94 == int(500 / LOSS)
    assert s.binding_constraint == "risk per trade" and s.notional == pytest.approx(94 * 100.05)
    assert s.est_loss_at_stop == pytest.approx(94 * LOSS) and s.est_loss_at_stop <= 500
    assert s.caps["position size"] == 99 and not s.illustrative
    json.dumps(s.to_dict())


@pytest.mark.parametrize(("change", "limits", "units", "binding"), [
    ({"stop": 99.0}, LIMITS, 99, "position size"),                                  # tight stop: notional cap binds
    ({}, RiskLimits(account_equity=100_000, available_cash=5_000), 49, "available cash"),
    ({"lot_size": 10.0}, LIMITS, 90, "risk per trade"),                             # whole lots only
    ({"fx_rate_to_account": 1.25}, LIMITS, 75, "risk per trade"),                   # loss in the account currency
    ({"gap_allowance": 5.0}, LIMITS, 48, "risk per trade"),                         # a gap counts against the budget
])
def test_caps_name_the_binding_limit(change, limits, units, binding):
    s = size_position(TradeSetup(**{**SETUP.__dict__, **change}), limits)
    assert (s.units, s.binding_constraint) == (units, binding)


def test_sector_and_gross_exposure_count_existing_positions():
    book = PortfolioState(equity=100_000, cash=50_000, positions=[Position("MEGA", 290, 100, 100, "Tech")])
    s = size_position(SETUP, LIMITS, book)
    assert s.binding_constraint == "sector" and s.units == 9                       # 1,000 of room / 100.05
    nearly = PortfolioState(equity=100_000, cash=1_000, positions=[Position("ALL", 990, 100, 100, "Energy")])
    assert size_position(SETUP, LIMITS, nearly).caps["gross exposure"] == 9          # 1,000 of room left
    full = PortfolioState(equity=100_000, cash=0, positions=[Position("ALL", 1000, 100, 100, "Energy")])
    s = size_position(SETUP, LIMITS, full)
    assert s.units == 0 and s.binding_constraint in {"gross exposure", "available cash"} and "No position" in s.warnings[0]


def test_invalid_setups_size_to_zero_with_the_reason():
    s = size_position(TradeSetup("ACME", 100.0, 100.0), LIMITS)
    assert s.units == 0 and s.binding_constraint == "invalid setup" and "not below the entry" in s.warnings[0]
    assert not trade_allowed(check_limits(TradeSetup("ACME", 100.0, 100.0), s, LIMITS))
    with pytest.raises(ValueError, match="short"):
        TradeSetup("ACME", 100.0, 105.0, side="short")
    with pytest.raises(ValueError):
        TradeSetup("ACME", float("nan"), 95.0)


def test_limits_refuse_leverage_and_label_illustrative_defaults():
    with pytest.raises(ValidationError, match="leverage"):
        RiskLimits(account_equity=100_000, max_gross_exposure_pct=2.0)
    with pytest.raises(ValidationError, match="margin"):
        RiskLimits(account_equity=100_000, available_cash=150_000)
    with pytest.raises(ValidationError):
        RiskLimits(account_equity=0)
    demo = illustrative_limits()
    assert demo.illustrative and "ILLUSTRATIVE" in demo.describe()[0]
    assert "ILLUSTRATIVE" in size_position(SETUP, demo).warnings[0]


FLAT = PortfolioState(equity=100_000, cash=100_000, peak_equity=100_000, start_of_day_equity=100_000)


def test_every_check_passes_for_a_trade_inside_the_limits_with_a_known_account():
    checks = check_limits(SETUP, size_position(SETUP, LIMITS, FLAT), LIMITS, FLAT)
    assert trade_allowed(checks), [c for c in checks if not c.ok]
    assert {c.name for c in checks} >= {"risk per trade", "position size", "cash (no margin)", "gross exposure",
                                        "sector", "daily loss", "drawdown halt", "correlated exposure",
                                        "sizing figures consistent"}


def test_limits_that_cannot_be_checked_fail_closed():
    """Without account state the daily-loss, drawdown and correlation limits cannot be confirmed: do not trade."""
    checks = {c.name: c for c in check_limits(SETUP, size_position(SETUP, LIMITS), LIMITS)}
    for name in ("daily loss", "drawdown halt", "correlated exposure"):
        assert not checks[name].ok and checks[name].detail.startswith("Not checked"), name
    assert not trade_allowed(list(checks.values()))
    no_peak = PortfolioState(equity=100_000, cash=100_000)
    halt = next(c for c in check_limits(SETUP, size_position(SETUP, LIMITS, no_peak), LIMITS, no_peak)
                if c.name == "drawdown halt")
    assert not halt.ok and "peak is unknown" in halt.detail
    with pytest.raises(ValueError):
        PortfolioState(equity=100_000, cash=0, peak_equity=float("nan"))
    with pytest.raises(ValueError):
        PortfolioState(equity=100_000, cash=0, realised_pnl_today=float("nan"))


def test_an_existing_holding_in_the_same_ticker_counts_toward_the_position_limit():
    held = PortfolioState(equity=100_000, cash=90_100, positions=[Position("ACME", 99, 100, 100, "Tech")],
                          peak_equity=100_000, start_of_day_equity=100_000)
    s = size_position(SETUP, LIMITS, held)
    assert s.binding_constraint == "position size" and s.units == int(100 / 100.05)   # 100 of room left
    checks = {c.name: c for c in check_limits(SETUP, s, LIMITS, held)}
    assert checks["position size"].ok and "already held" in checks["position size"].detail
    again = s.__class__(**{**s.__dict__, "units": 99, "notional": 99 * 100.05, "est_loss_with_gap": 99 * LOSS})
    assert not {c.name: c for c in check_limits(SETUP, again, LIMITS, held)}["position size"].ok


def test_limits_apply_to_the_lower_of_configured_and_current_equity_and_daily_loss_counts_unrealised():
    down = PortfolioState(equity=88_000, cash=88_000, peak_equity=100_000, start_of_day_equity=90_000)
    s = size_position(SETUP, LIMITS, down)
    assert s.risk_budget == pytest.approx(440) and any("current account equity" in w for w in s.warnings)
    loss = next(c for c in check_limits(SETUP, s, LIMITS, down) if c.name == "daily loss")
    assert loss.value == pytest.approx(2_000 / 90_000) and not loss.ok and "unrealised" in loss.detail


def test_daily_loss_drawdown_and_correlated_exposure_block_new_risk():
    corr = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=["ACME", "PEER"], columns=["ACME", "PEER"])
    book = PortfolioState(equity=84_000, cash=59_000, positions=[Position("PEER", 250, 100, 100, "Industrials")],
                          peak_equity=100_000, realised_pnl_today=-2_500)
    checks = {c.name: c for c in check_limits(SETUP, size_position(SETUP, LIMITS, book), LIMITS, book, corr)}
    assert not checks["daily loss"].ok and checks["daily loss"].value == pytest.approx(2_500 / 84_000)
    assert not checks["drawdown halt"].ok and checks["drawdown halt"].value == pytest.approx(0.16)
    assert not checks["correlated exposure"].ok and "PEER (corr +0.80)" in checks["correlated exposure"].detail
    assert not trade_allowed(list(checks.values()))


def test_unknown_sectors_and_correlations_are_counted_as_concentrated():
    """Conservative: a holding of unknown sector or correlation counts as the same sector / correlated."""
    book = PortfolioState(equity=100_000, cash=60_000, positions=[Position("MSFT", 400, 100, 100, None)],
                          peak_equity=100_000, start_of_day_equity=100_000)
    checks = {c.name: c for c in check_limits(SETUP, size_position(SETUP, LIMITS, book), LIMITS, book)}
    assert not checks["sector"].ok and "unknown sectors counted" in checks["sector"].detail
    assert not checks["correlated exposure"].ok and "MSFT (correlation unknown)" in checks["correlated exposure"].detail
    independent = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["ACME", "MSFT"], columns=["ACME", "MSFT"])
    known = PortfolioState(equity=100_000, cash=60_000, positions=[Position("MSFT", 400, 100, 100, "Software")],
                           peak_equity=100_000, start_of_day_equity=100_000)
    ok = {c.name: c for c in check_limits(SETUP, size_position(SETUP, LIMITS, known), LIMITS, known, independent)}
    assert ok["sector"].ok and ok["correlated exposure"].ok


def test_a_hand_made_oversized_sizing_fails_its_checks():
    """The checks recompute the numbers from the units: an inflated size cannot slip through."""
    honest = size_position(SETUP, LIMITS, FLAT)
    inflated = honest.__class__(**{**honest.__dict__, "units": 500})
    failing = {c.name for c in check_limits(SETUP, inflated, LIMITS, FLAT) if not c.ok}
    assert failing == {"sizing figures consistent", "risk per trade", "position size", "sector",
                       "correlated exposure"}
    assert not trade_allowed([LimitCheck("x", True, None, None, ""), LimitCheck("y", False, 1, 0, "")])
    assert not trade_allowed([])


def test_gap_risk_counts_opens_through_the_stop():
    idx = pd.bdate_range("2023-01-02", periods=300)
    opens = np.full(300, 100.0)
    opens[[50, 100, 150, 200, 250]] = 90.0                                          # five −10% gaps
    prices = pd.DataFrame({"open": opens, "high": 101.0, "low": 89.0, "close": 100.0, "volume": 1e6}, index=idx)
    g = gap_risk(prices, 0.05)
    assert g["sessions"] == 299 and g["gaps_through_stop"] == 5
    assert g["prob_gap_through_stop"] == pytest.approx(5 / 299) and g["worst_gap"] == pytest.approx(-0.10)
    # Worst case: a -10% gap from a close just above the stop lands 10% below the stop = 9.5% of the entry.
    assert g["adverse_gap_p99"] == pytest.approx(0.10) and g["suggested_gap_allowance_pct"] == pytest.approx(0.095)
    assert gap_risk(prices.iloc[:10], 0.05)["prob_gap_through_stop"] is None
    with pytest.raises(ValueError):
        gap_risk(prices, 5)


def test_risk_reward_reports_gross_and_net_r_and_rejects_targets_below_entry():
    rr = risk_reward(SETUP)
    first, second = rr["targets"]
    exit_eff = 110 * 0.9995
    net = exit_eff - 100.05 - 0.001 * (100.05 + exit_eff)
    assert first["gross_r"] == pytest.approx(2.0) and first["net_r"] == pytest.approx(net / LOSS)
    assert second["gross_r"] is None and "not above" in second["note"]
    assert not risk_reward(TradeSetup("ACME", 100.0, 101.0))["valid"]
