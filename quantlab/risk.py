"""Risk before opportunity: position sizing, gap risk, risk/reward and hard limit checks.

Every size the agent may quote comes from :func:`size_position`; every "may I take
this trade" answer comes from :func:`check_limits`. Both are plain arithmetic on
the user's own limits, so the language model can neither invent a size nor talk
its way past a failed check.

Sizing (long only; prices in the instrument's currency, money in the account's)::

    entry_eff      = entry × (1 + slippage)            what a buy is assumed to cost
    stop_eff       = stop × (1 − slippage)             what a stop sale is assumed to fetch
    loss_per_unit  = entry_eff − stop_eff + fees on both legs        (at the stop)
    sizing_loss    = loss_per_unit + gap_allowance                   (a gap through the stop)
    risk_budget    = account_equity × max_risk_per_trade_pct
    units          = floor(risk_budget / (sizing_loss × fx) / lot) × lot

then capped by the position, cash, gross-exposure and sector limits; the binding
cap is named. No leverage: gross exposure can never exceed equity, and a buy can
never exceed available cash. A stop is a plan, not a guarantee: the market can
open below it, which is why the gap allowance and :func:`gap_risk` exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantlab.serialization import clean_json
from quantlab.technicals import overnight_gaps

CORRELATION_THRESHOLD = 0.7
_EPS = 1e-12


class RiskLimits(BaseModel):
    """The user's risk limits. ``illustrative`` marks defaults the user did not choose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_equity: float = Field(gt=0, allow_inf_nan=False)
    available_cash: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_risk_per_trade_pct: float = Field(default=0.005, gt=0, le=0.05, allow_inf_nan=False)
    max_daily_loss_pct: float = Field(default=0.02, gt=0, le=0.25, allow_inf_nan=False)
    max_position_pct: float = Field(default=0.10, gt=0, le=1.0, allow_inf_nan=False)
    max_gross_exposure_pct: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_sector_pct: float = Field(default=0.30, gt=0, le=1.0, allow_inf_nan=False)
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=1.0, allow_inf_nan=False)
    illustrative: bool = False

    @model_validator(mode="after")
    def _no_leverage(self) -> RiskLimits:
        if self.max_gross_exposure_pct > 1.0:
            raise ValueError("max_gross_exposure_pct above 1.0 would mean leverage, which is not supported.")
        if self.available_cash is not None and self.available_cash > self.account_equity * (1 + 1e-9):
            raise ValueError("available_cash cannot exceed account_equity (no margin).")
        return self

    @property
    def cash(self) -> float:
        return self.account_equity if self.available_cash is None else self.available_cash

    def describe(self) -> list[str]:
        e = self.account_equity
        label = " (ILLUSTRATIVE: not supplied by the user)" if self.illustrative else ""
        return [
            f"Account equity {e:,.2f}{label}; available cash {self.cash:,.2f}.",
            f"Risk per trade at most {self.max_risk_per_trade_pct:.2%} of equity ({e * self.max_risk_per_trade_pct:,.2f}).",
            f"New risk stops for the day once losses reach {self.max_daily_loss_pct:.2%} of equity.",
            f"One position at most {self.max_position_pct:.0%} of equity; one sector at most {self.max_sector_pct:.0%}.",
            f"Gross exposure at most {self.max_gross_exposure_pct:.0%} of equity (no leverage, no short selling).",
            f"New risk halts in a drawdown of {self.max_drawdown_pct:.0%} or more from the equity peak.",
        ]


def illustrative_limits(account_equity: float = 100_000.0) -> RiskLimits:
    """Conservative defaults for when the user has not given limits — always labelled as such."""
    return RiskLimits(account_equity=account_equity, illustrative=True)


@dataclass(frozen=True)
class TradeSetup:
    """A proposed long trade. A stop at or above the entry is accepted here and refused by the sizing."""

    ticker: str
    entry: float
    stop: float
    targets: tuple[float, ...] = ()
    side: str = "long"
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    lot_size: float = 1.0
    gap_allowance: float = 0.0          # extra adverse move per unit beyond the stop, in price units
    fx_rate_to_account: float = 1.0     # account currency per unit of the instrument's currency
    sector: str | None = None

    def __post_init__(self) -> None:
        if self.side != "long":
            raise ValueError("Only long trades are supported: no short selling. A bearish view means avoid or reduce.")
        for name in ("entry", "stop", "gap_allowance"):
            if not _is_number(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        if not all(_is_number(t) for t in self.targets):
            raise ValueError("targets must be finite numbers")
        object.__setattr__(self, "targets", tuple(float(t) for t in self.targets))
        if not 0 <= self.fee_rate < 0.05 or not 0 <= self.slippage_rate < 0.05:
            raise ValueError("fee_rate and slippage_rate must be in [0, 0.05)")
        if not (_is_number(self.lot_size) and self.lot_size > 0):
            raise ValueError("lot_size must be positive")
        if self.gap_allowance < 0:
            raise ValueError("gap_allowance cannot be negative")
        if not (_is_number(self.fx_rate_to_account) and self.fx_rate_to_account > 0):
            raise ValueError("fx_rate_to_account must be positive")

    @property
    def entry_eff(self) -> float:
        return self.entry * (1 + self.slippage_rate)

    @property
    def stop_eff(self) -> float:
        return self.stop * (1 - self.slippage_rate)

    @property
    def loss_per_unit(self) -> float:
        """Loss per unit if stopped out at the stop: slippage and both legs' fees included (instrument currency)."""
        return self.entry_eff - self.stop_eff + self.fee_rate * (self.entry_eff + self.stop_eff)

    def problems(self) -> list[str]:
        out = []
        if self.entry <= 0 or self.stop <= 0:
            out.append("Prices must be positive.")
        elif self.stop >= self.entry:
            out.append(f"The stop ({self.stop:g}) is not below the entry ({self.entry:g}); a long trade needs a stop "
                       "below the entry.")
        return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


@dataclass
class Position:
    ticker: str
    units: float
    entry_price: float
    last_price: float
    sector: str | None = None

    @property
    def value(self) -> float:
        return self.units * self.last_price


@dataclass
class PortfolioState:
    """Current account state. Long-only and unlevered: cash cannot be negative."""

    equity: float
    cash: float
    positions: list[Position] = field(default_factory=list)
    peak_equity: float | None = None
    realised_pnl_today: float = 0.0

    def __post_init__(self) -> None:
        if not (_is_number(self.equity) and self.equity > 0):
            raise ValueError("equity must be positive")
        if not (_is_number(self.cash) and self.cash >= 0):
            raise ValueError("cash cannot be negative (no margin)")
        if any(p.units < 0 for p in self.positions):
            raise ValueError("positions must be long (no short selling)")

    def gross_exposure(self) -> float:
        return float(sum(p.value for p in self.positions))

    def sector_exposure(self, sector: str | None) -> float:
        if not sector:
            return 0.0
        key = sector.strip().lower()
        return float(sum(p.value for p in self.positions if p.sector and p.sector.strip().lower() == key))

    def drawdown(self) -> float:
        """Fraction below the equity peak (0 when at or above it, or when no peak is known)."""
        if not self.peak_equity or self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


@dataclass(frozen=True)
class SizingResult:
    units: float
    notional: float                 # units × entry_eff, account currency (fees excluded)
    risk_budget: float
    loss_per_unit_at_stop: float    # account currency, slippage and fees included, no gap
    est_loss_at_stop: float
    est_loss_with_gap: float
    binding_constraint: str
    formula: str
    warnings: tuple[str, ...]
    illustrative: bool
    caps: dict[str, float | None] = field(default_factory=dict)   # units allowed by each limit

    def to_dict(self) -> dict[str, Any]:
        return clean_json({
            "units": self.units, "notional": self.notional, "risk_budget": self.risk_budget,
            "loss_per_unit_at_stop": self.loss_per_unit_at_stop, "est_loss_at_stop": self.est_loss_at_stop,
            "est_loss_with_gap": self.est_loss_with_gap, "binding_constraint": self.binding_constraint,
            "units_allowed_by_each_limit": self.caps, "formula": self.formula, "warnings": list(self.warnings),
            "illustrative_limits": self.illustrative,
        })


def _floor_lots(units: float, lot: float) -> float:
    if not math.isfinite(units) or units <= 0:
        return 0.0
    lots = math.floor(units / lot + 1e-9)
    return float(lots * lot)


def size_position(setup: TradeSetup, limits: RiskLimits, portfolio: PortfolioState | None = None) -> SizingResult:
    """The largest long position whose loss at the stop (plus the gap allowance) fits the risk budget and every cap."""
    fx = setup.fx_rate_to_account
    equity = limits.account_equity
    budget = equity * limits.max_risk_per_trade_pct
    formula = ("units = floor(risk_budget / ((entry×(1+slippage) − stop×(1−slippage) + fees on both legs + "
               "gap_allowance) × fx) / lot) × lot; risk_budget = account_equity × max_risk_per_trade_pct; then "
               "capped by position, cash, gross-exposure and sector limits")
    warnings: list[str] = []
    if limits.illustrative:
        warnings.append("Sized with ILLUSTRATIVE limits the user did not supply; not a recommendation for any account.")
    problems = setup.problems()
    if problems:
        return SizingResult(0.0, 0.0, budget, 0.0, 0.0, 0.0, "invalid setup", formula, tuple(problems + warnings),
                            limits.illustrative)

    loss = setup.loss_per_unit * fx
    sizing_loss = (setup.loss_per_unit + setup.gap_allowance) * fx
    price = setup.entry_eff * fx
    cash = limits.cash if portfolio is None else min(limits.cash, portfolio.cash)
    gross_now = portfolio.gross_exposure() if portfolio else 0.0
    sector_now = portfolio.sector_exposure(setup.sector) if portfolio else 0.0

    caps: dict[str, float | None] = {
        "risk per trade": budget / sizing_loss,
        "position size": limits.max_position_pct * equity / price,
        "available cash": cash / (price * (1 + setup.fee_rate)),
        "gross exposure": max(0.0, limits.max_gross_exposure_pct * equity - gross_now) / price,
        "sector": (max(0.0, limits.max_sector_pct * equity - sector_now) / price) if setup.sector else None,
    }
    if not setup.sector:
        warnings.append("Sector unknown: the sector limit was not applied.")
    binding = min((k for k, v in caps.items() if v is not None), key=lambda k: caps[k])
    units = _floor_lots(caps[binding], setup.lot_size)
    caps = {k: (None if v is None else _floor_lots(v, setup.lot_size)) for k, v in caps.items()}
    if units == 0:
        warnings.append(f"No position: the {binding} limit allows less than one lot of {setup.lot_size:g}.")
    stop_pct = (setup.entry - setup.stop) / setup.entry
    if stop_pct < 0.005:
        warnings.append(f"The stop is only {stop_pct:.2%} below the entry: normal daily noise may trigger it.")
    if setup.gap_allowance == 0:
        warnings.append("No gap allowance: an open below the stop would lose more than the estimate.")
    return SizingResult(
        units=units, notional=units * price, risk_budget=budget, loss_per_unit_at_stop=loss,
        est_loss_at_stop=units * loss, est_loss_with_gap=units * sizing_loss, binding_constraint=binding,
        formula=formula, warnings=tuple(warnings), illustrative=limits.illustrative, caps=caps,
    )


def gap_risk(prices: pd.DataFrame, stop_distance_pct: float) -> dict[str, Any]:
    """How often this asset has opened through a stop this far below the prior close.

    Uses the asset's own overnight gaps (``open_t / close_{t−1} − 1``). The suggested
    allowance is the 99th-percentile adverse gap beyond the stop distance, as a
    fraction of price (multiply by the entry for a per-unit ``gap_allowance``). History
    is a sample, not a bound: a larger gap than any seen is always possible.
    """
    if not _is_number(stop_distance_pct) or not 0 < stop_distance_pct < 1:
        raise ValueError("stop_distance_pct must be between 0 and 1 (e.g. 0.05 for a stop 5% below)")
    gaps = overnight_gaps(prices).dropna()
    gaps = gaps[np.isfinite(gaps)]
    n = int(len(gaps))
    out: dict[str, Any] = {"sessions": n, "stop_distance_pct": float(stop_distance_pct),
                           "definition": "overnight gap = open / previous close − 1; adverse = below −stop distance"}
    if n < 20:
        out.update(prob_gap_through_stop=None, worst_gap=None, adverse_gap_p99=None,
                   suggested_gap_allowance_pct=None, warnings=[f"Only {n} overnight gaps: too few to estimate."])
        return clean_json(out)
    through = int((gaps < -stop_distance_pct).sum())
    p99 = float(-np.quantile(gaps.to_numpy(), 0.01))
    warnings = []
    if n < 250:
        warnings.append(f"Based on {n} sessions (under a year): the estimate is rough.")
    if through == 0:
        warnings.append("No gap through this stop distance in the sample; that does not make one impossible.")
    out.update(gaps_through_stop=through, prob_gap_through_stop=through / n, worst_gap=float(gaps.min()),
               adverse_gap_p99=max(0.0, p99), suggested_gap_allowance_pct=max(0.0, p99 - stop_distance_pct),
               warnings=warnings)
    return clean_json(out)


def risk_reward(setup: TradeSetup) -> dict[str, Any]:
    """Per target: the gross R multiple (price distances) and the net R after slippage and fees on both legs."""
    problems = setup.problems()
    if problems:
        return {"ticker": setup.ticker, "valid": False, "reason": " ".join(problems), "targets": []}
    risk = setup.entry - setup.stop
    rows = []
    for target in setup.targets:
        if target <= setup.entry:
            rows.append({"target": target, "gross_r": None, "net_r": None,
                         "note": "target is not above the entry"})
            continue
        exit_eff = target * (1 - setup.slippage_rate)
        net_gain = exit_eff - setup.entry_eff - setup.fee_rate * (setup.entry_eff + exit_eff)
        rows.append({"target": target, "gross_r": (target - setup.entry) / risk,
                     "net_r": net_gain / setup.loss_per_unit, "note": ""})
    return clean_json({
        "ticker": setup.ticker, "valid": True, "entry": setup.entry, "stop": setup.stop,
        "risk_per_unit": risk, "net_loss_per_unit_at_stop": setup.loss_per_unit, "targets": rows,
        "formula": "gross R = (target − entry) / (entry − stop); net R = gain after slippage and fees / loss at "
                   "the stop after slippage and fees",
        "note": "R multiples describe the payoff if a level is reached; they say nothing about how likely that is.",
    })


@dataclass(frozen=True)
class LimitCheck:
    name: str
    ok: bool
    value: float | None
    limit: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return clean_json({"name": self.name, "ok": self.ok, "value": self.value, "limit": self.limit,
                           "detail": self.detail})


def check_limits(setup: TradeSetup, sizing: SizingResult, limits: RiskLimits,
                 portfolio: PortfolioState | None = None, correlations: pd.DataFrame | None = None) -> list[LimitCheck]:
    """Every limit, checked in code. One failing check means do not trade; nothing can override it."""
    equity = limits.account_equity
    checks: list[LimitCheck] = []
    problems = setup.problems()
    checks.append(LimitCheck("valid setup", not problems and sizing.units > 0, sizing.units, None,
                             " ".join(problems) or ("Size is positive." if sizing.units > 0 else
                                                    f"Size is zero ({sizing.binding_constraint} limit).")))
    risk = sizing.est_loss_with_gap / equity
    checks.append(LimitCheck("risk per trade", risk <= limits.max_risk_per_trade_pct + _EPS, risk,
                             limits.max_risk_per_trade_pct,
                             f"Estimated loss at the stop incl. gap allowance {sizing.est_loss_with_gap:,.2f} = "
                             f"{risk:.3%} of equity."))
    position = sizing.notional / equity
    checks.append(LimitCheck("position size", position <= limits.max_position_pct + _EPS, position,
                             limits.max_position_pct, f"Notional {sizing.notional:,.2f} = {position:.2%} of equity."))
    cash = limits.cash if portfolio is None else min(limits.cash, portfolio.cash)
    cost = sizing.notional * (1 + setup.fee_rate)
    checks.append(LimitCheck("cash (no margin)", cost <= cash + 1e-6, cost, cash,
                             f"Cost incl. fees {cost:,.2f} against available cash {cash:,.2f}."))
    gross_now = portfolio.gross_exposure() if portfolio else 0.0
    gross = (gross_now + sizing.notional) / equity
    checks.append(LimitCheck("gross exposure", gross <= limits.max_gross_exposure_pct + _EPS, gross,
                             limits.max_gross_exposure_pct, f"Gross exposure after the trade {gross:.2%} of equity."))
    if setup.sector:
        sector = ((portfolio.sector_exposure(setup.sector) if portfolio else 0.0) + sizing.notional) / equity
        checks.append(LimitCheck("sector", sector <= limits.max_sector_pct + _EPS, sector, limits.max_sector_pct,
                                 f"{setup.sector} exposure after the trade {sector:.2%} of equity."))
    else:
        checks.append(LimitCheck("sector", True, None, limits.max_sector_pct, "Not checked: the sector is unknown."))
    if portfolio is None:
        checks.append(LimitCheck("daily loss", True, None, limits.max_daily_loss_pct,
                                 "Not checked: no account state supplied (today's P&L unknown)."))
        checks.append(LimitCheck("drawdown halt", True, None, limits.max_drawdown_pct,
                                 "Not checked: no account state supplied (equity peak unknown)."))
    else:
        lost = max(0.0, -portfolio.realised_pnl_today) / equity
        checks.append(LimitCheck("daily loss", lost < limits.max_daily_loss_pct, lost, limits.max_daily_loss_pct,
                                 f"Realised loss today {lost:.2%} of equity; new risk stops at "
                                 f"{limits.max_daily_loss_pct:.2%}."))
        dd = portfolio.drawdown()
        checks.append(LimitCheck("drawdown halt", dd < limits.max_drawdown_pct, dd, limits.max_drawdown_pct,
                                 f"Drawdown from the equity peak {dd:.2%}; new risk halts at {limits.max_drawdown_pct:.0%}."))
    checks.append(_correlated(setup, sizing, limits, portfolio, correlations))
    return checks


def _correlated(setup: TradeSetup, sizing: SizingResult, limits: RiskLimits, portfolio: PortfolioState | None,
                correlations: pd.DataFrame | None) -> LimitCheck:
    """Positions moving with the new one (|corr| >= 0.7) count together against the sector limit."""
    name = "correlated exposure"
    if portfolio is None or not portfolio.positions:
        return LimitCheck(name, True, None, limits.max_sector_pct, "No existing positions.")
    if correlations is None or setup.ticker not in correlations.index:
        return LimitCheck(name, True, None, limits.max_sector_pct,
                          "Not checked: no return correlations with the existing positions were supplied.")
    linked = []
    for p in portfolio.positions:
        if p.ticker == setup.ticker:
            linked.append((p, 1.0))
            continue
        if p.ticker in correlations.columns:
            corr = correlations.at[setup.ticker, p.ticker]
            if pd.notna(corr) and abs(float(corr)) >= CORRELATION_THRESHOLD:
                linked.append((p, float(corr)))
    combined = (sizing.notional + sum(p.value for p, _ in linked)) / limits.account_equity
    detail = (f"Combined with {', '.join(f'{p.ticker} (corr {c:+.2f})' for p, c in linked)}: {combined:.2%} of equity."
              if linked else f"No existing position has |correlation| >= {CORRELATION_THRESHOLD}.")
    return LimitCheck(name, combined <= limits.max_sector_pct + _EPS, combined, limits.max_sector_pct, detail)


def trade_allowed(checks: list[LimitCheck]) -> bool:
    return bool(checks) and all(c.ok for c in checks)
