"""PORTFOLIO REVIEW: concentration, sectors, correlation, risk, scenarios and liquidity of current holdings.

Weights come from each holding's latest validated close. Risk figures apply today's
weights to the past ``lookback`` sessions of returns: that is a *hypothetical*
constant-weight portfolio, not the account's actual history, and is labelled as such.
A holding whose data cannot be loaded or fails validation is listed under
``data_failures`` with the reason and is never silently dropped; the figures then
cover only the holdings that loaded, and a warning says so.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantlab.data import MarketDataset
from quantlab.inference import conditional_value_at_risk, value_at_risk
from quantlab.metrics import TRADING_DAYS_PER_YEAR
from quantlab.serialization import clean_json
from quantlab.technicals import average_dollar_volume, load_validated
from quantlab.validation import InputValidationError, validate_ticker

MAX_HOLDINGS = 200
MAX_CSV_BYTES = 1_000_000
MIN_COMMON_SESSIONS = 60
HYPOTHETICAL = ("Hypothetical: today's weights applied to past returns (no rebalancing, costs or cash flows); "
                "not the account's actual history.")


@dataclass(frozen=True)
class Holding:
    ticker: str
    quantity: float
    cost_basis: float | None = None     # per unit, instrument currency
    sector: str | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.quantity, (int, float)) and math.isfinite(self.quantity) and self.quantity > 0):
            raise ValueError(f"{self.ticker}: quantity must be a positive number (long positions only)")
        if self.cost_basis is not None and not (math.isfinite(self.cost_basis) and self.cost_basis > 0):
            raise ValueError(f"{self.ticker}: cost_basis must be positive when given")


def parse_holdings_csv(content: bytes) -> list[Holding]:
    """Holdings from a CSV with columns ``ticker, quantity[, cost_basis, sector]`` (header names case-insensitive).

    Every bad row is reported with its line number; nothing is guessed. Repeated tickers are
    combined (quantities added, cost basis quantity-weighted when every lot has one).
    """
    if len(content) > MAX_CSV_BYTES:
        raise ValueError(f"The holdings file is larger than {MAX_CSV_BYTES // 1000} kB.")
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the holdings CSV: {exc}") from exc
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    aliases = {"symbol": "ticker", "shares": "quantity", "qty": "quantity", "units": "quantity",
               "cost": "cost_basis", "avg_cost": "cost_basis", "price_paid": "cost_basis"}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns and v not in frame.columns})
    missing = {"ticker", "quantity"} - set(frame.columns)
    if missing:
        raise ValueError(f"The holdings CSV needs columns ticker and quantity; missing: {', '.join(sorted(missing))}.")
    if len(frame) > MAX_HOLDINGS:
        raise ValueError(f"At most {MAX_HOLDINGS} holdings are supported.")
    errors: list[str] = []
    lots: dict[str, list[Holding]] = {}
    for i, row in frame.iterrows():
        line = int(i) + 2   # header is line 1
        try:
            ticker = validate_ticker(row["ticker"])
            quantity = _number(row["quantity"], "quantity")
            cost = _number(row.get("cost_basis", ""), "cost_basis", optional=True)
            sector = str(row.get("sector", "")).strip()[:60] or None
            lots.setdefault(ticker, []).append(Holding(ticker, quantity, cost, sector))
        except (InputValidationError, ValueError) as exc:
            errors.append(f"line {line}: {exc}")
    if errors:
        raise ValueError("The holdings CSV has invalid rows — " + "; ".join(errors[:10]))
    if not lots:
        raise ValueError("The holdings CSV has no rows.")
    out = []
    for ticker, group in lots.items():
        quantity = sum(h.quantity for h in group)
        cost = (sum(h.quantity * h.cost_basis for h in group) / quantity
                if all(h.cost_basis is not None for h in group) else None)
        sector = next((h.sector for h in group if h.sector), None)
        out.append(Holding(ticker, quantity, cost, sector))
    return out


def _number(raw: Any, name: str, optional: bool = False) -> float | None:
    text = str(raw).strip().replace(",", "")
    if not text:
        if optional:
            return None
        raise ValueError(f"{name} is empty")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} {raw!r} is not a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


@dataclass(frozen=True)
class PortfolioReview:
    as_of: date | None
    holdings: pd.DataFrame
    concentration: dict[str, Any]
    sector_exposure: pd.DataFrame
    correlation: pd.DataFrame
    risk: dict[str, Any]
    scenarios: pd.DataFrame
    liquidity: pd.DataFrame
    warnings: tuple[str, ...]
    data_failures: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return frame.reset_index().to_dict(orient="records") if not frame.empty else []

        return clean_json({
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "holdings": records(self.holdings), "concentration": self.concentration,
            "sector_exposure": records(self.sector_exposure),
            "correlation": {str(k): {str(c): v for c, v in row.items()} for k, row in
                            self.correlation.round(3).to_dict(orient="index").items()},
            "risk": self.risk, "scenarios": records(self.scenarios), "liquidity": records(self.liquidity),
            "warnings": list(self.warnings), "data_failures": self.data_failures,
        })


def _bench_close(bench: pd.Series | pd.DataFrame | MarketDataset | None) -> pd.Series | None:
    if bench is None:
        return None
    prices = getattr(bench, "prices", bench)
    series = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    return pd.to_numeric(series, errors="coerce").astype("float64").dropna()


def review_portfolio(holdings: list[Holding], load_prices: Callable[[str], MarketDataset],
                     bench_prices: pd.Series | pd.DataFrame | MarketDataset | None = None,
                     lookback: int = 252) -> PortfolioReview:
    """Review current holdings on validated end-of-day data (see the module docstring for the conventions)."""
    if not holdings:
        raise ValueError("No holdings to review.")
    if lookback < MIN_COMMON_SESSIONS:
        raise ValueError(f"lookback must be at least {MIN_COMMON_SESSIONS} sessions")
    failures: dict[str, str] = {}
    warnings: list[str] = []
    rows, closes, frames = [], {}, {}
    for h in holdings:
        outcome = load_validated(h.ticker, load_prices)
        if not outcome.ok:
            failures[h.ticker] = outcome.reason
            continue
        prices = outcome.dataset.prices
        last = prices.index[-1]
        close = float(prices["close"].iloc[-1])
        value = h.quantity * close
        rows.append({"ticker": h.ticker, "quantity": h.quantity, "last_close": close, "as_of": last.date(),
                     "value": value, "sector": h.sector or "Unknown", "cost_basis": h.cost_basis,
                     "unrealised_pnl": None if h.cost_basis is None else (close - h.cost_basis) * h.quantity,
                     "data_status": outcome.status})
        closes[h.ticker] = prices["close"].astype("float64")
        frames[h.ticker] = prices
    if failures:
        warnings.append(f"{len(failures)} holding(s) could not be evaluated and are excluded from every figure: "
                        + ", ".join(sorted(failures)) + ".")
    if not rows:
        empty = pd.DataFrame()
        return PortfolioReview(None, empty, {}, empty, empty, {}, empty, empty,
                               tuple(warnings + ["No holding could be evaluated."]), failures)

    table = pd.DataFrame(rows).set_index("ticker")
    total = float(table["value"].sum())
    table["weight"] = table["value"] / total
    as_of = max(table["as_of"])
    lagging = sorted(t for t, d in table["as_of"].items() if d < as_of)
    if lagging:
        warnings.append(f"Latest closes are from different dates ({', '.join(lagging)} older than {as_of}); "
                        "weights mix dates.")
    if table["cost_basis"].isna().any():
        warnings.append("Cost basis missing for some holdings: unrealised P&L is shown only where it was given.")
    if any(r["sector"] == "Unknown" for r in rows):
        warnings.append("Sector missing for some holdings: they are grouped as 'Unknown'.")

    weights = table["weight"]
    hhi = float((weights**2).sum())
    ranked = weights.sort_values(ascending=False)
    concentration = {"positions": len(weights), "total_value": total, "hhi": hhi, "effective_positions": 1 / hhi,
                     "largest": ranked.index[0], "largest_weight": float(ranked.iloc[0]),
                     "top_3_weight": float(ranked.iloc[:3].sum()),
                     "definition": "HHI = Σ weight²; effective positions = 1 / HHI"}
    sectors = (table.groupby("sector")["weight"].sum().sort_values(ascending=False).to_frame("weight"))

    returns = pd.DataFrame(closes).pct_change(fill_method=None).dropna(how="any").iloc[-lookback:]
    risk: dict[str, Any] = {"sessions": int(len(returns)), "label": HYPOTHETICAL}
    correlation = pd.DataFrame()
    scenarios = pd.DataFrame()
    if len(returns) < MIN_COMMON_SESSIONS:
        warnings.append(f"Only {len(returns)} sessions are common to every holding (need {MIN_COMMON_SESSIONS}); "
                        "volatility, VaR, correlation and scenarios are not computed.")
    else:
        if len(returns) < lookback:
            warnings.append(f"Risk uses {len(returns)} common sessions, fewer than the {lookback} requested.")
        port = returns[weights.index].mul(weights, axis=1).sum(axis=1)
        equity = (1 + port).cumprod()
        risk.update(
            start=returns.index[0].date(), end=returns.index[-1].date(),
            annualised_volatility=float(port.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)),
            var_95_1d=value_at_risk(port, 0.95, "historical"), cvar_95_1d=conditional_value_at_risk(port, 0.95),
            max_drawdown=float((equity / equity.cummax() - 1).min()),
            definitions="VaR/CVaR: historical one-day 5% quantile / mean beyond it, as returns (losses negative).",
        )
        risk["var_95_1d_value"] = None if risk["var_95_1d"] is None else risk["var_95_1d"] * total
        correlation = returns.corr()
        monthly = (1 + port).groupby(port.index.to_period("M")).prod() - 1
        complete = monthly.iloc[1:-1] if len(monthly) > 2 else monthly.iloc[0:0]   # partial first/last months dropped
        scenario_rows = []
        if len(complete):
            worst = complete.idxmin()
            scenario_rows.append({"scenario": f"Worst calendar month in the window ({worst})",
                                  "portfolio_return": float(complete.min()),
                                  "value_change": float(complete.min()) * total, "basis": HYPOTHETICAL})
        bench = _bench_close(bench_prices)
        if bench is not None:
            bench_ret = bench.pct_change(fill_method=None).dropna()
            common = returns.index.intersection(bench_ret.index)
            if len(common) >= MIN_COMMON_SESSIONS:
                b = bench_ret.loc[common]
                betas = {t: float(np.cov(returns.loc[common, t], b, ddof=1)[0, 1] / b.var(ddof=1)) for t in weights.index}
                table["beta"] = pd.Series(betas)
                port_beta = float((weights * table["beta"]).sum())
                risk.update(beta=port_beta, beta_sessions=int(len(common)))
                for shock in (-0.10, -0.20):
                    scenario_rows.append({"scenario": f"Benchmark {shock:+.0%} (beta-implied)",
                                          "portfolio_return": port_beta * shock, "value_change": port_beta * shock * total,
                                          "basis": f"portfolio beta {port_beta:.2f} × shock; betas over {len(common)} "
                                                   "sessions; real losses can be larger (betas rise in sell-offs)"})
            else:
                warnings.append("Too few sessions in common with the benchmark to estimate betas.")
        scenarios = pd.DataFrame(scenario_rows).set_index("scenario") if scenario_rows else pd.DataFrame()

    liquidity_rows = []
    for t in table.index:
        adv = average_dollar_volume(frames[t], 20).iloc[-1]
        adv = None if pd.isna(adv) or adv <= 0 else float(adv)
        days = None if adv is None else float(table.at[t, "value"] / adv)
        liquidity_rows.append({"ticker": t, "value": float(table.at[t, "value"]), "avg_dollar_volume_20": adv,
                               "position_to_adv": days,
                               "flag": "unknown volume" if adv is None else ("above 10% of ADV" if days > 0.10 else "")})
    liquidity = pd.DataFrame(liquidity_rows).set_index("ticker")
    if (liquidity["flag"] == "above 10% of ADV").any():
        warnings.append("Some positions exceed 10% of their 20-day average dollar volume: exiting could move the price.")
    warnings.append(HYPOTHETICAL)
    return PortfolioReview(as_of, table, concentration, sectors, correlation, clean_json(risk), scenarios, liquidity,
                           tuple(warnings), failures)
