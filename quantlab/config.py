"""Configuration objects and project-wide conventions.

Every number that changes how a backtest is calculated lives here, so that a
saved configuration fully describes how a result was produced.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

APP_NAME = "Quant Lab Pro"
APP_VERSION = "1.0.0"

# Bump METHODOLOGY_VERSION whenever a calculation convention changes (timing,
# cost model, metric definition). Saved experiments record it so results
# produced under different conventions are never silently compared.
METHODOLOGY_VERSION = "1.0.0"

# Annualisation convention: US equity markets trade roughly 252 sessions a year.
TRADING_DAYS_PER_YEAR = 252

# Below this many daily return observations, annualised figures are an
# extrapolation of less than one year of data and are flagged as such.
MIN_OBS_FOR_ANNUALISED = TRADING_DAYS_PER_YEAR

# Below this many observations a ratio's standard error is so large that the
# figure is shown with an "insufficient sample" note.
MIN_OBS_FOR_RATIO = 60

# Closed round-trip trades needed before win rate / average trade statistics
# are described as more than anecdotal.
MIN_TRADES_FOR_STATS = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Directory for local caches and databases (overridable for deployments)."""
    return Path(os.environ.get("QLP_DATA_DIR", PROJECT_ROOT / "data"))


class BacktestConfig(BaseModel):
    """Portfolio and execution assumptions for a single backtest.

    Rates are decimals: ``fee_rate=0.001`` means 0.10% of traded notional.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_capital: float = Field(
        default=10_000.0, gt=0, le=1e12, description="Starting cash in USD."
    )
    fee_rate: float = Field(
        default=0.001,
        ge=0,
        le=0.05,
        description="Commission charged on traded notional per execution (0.001 = 0.10%).",
    )
    slippage_rate: float = Field(
        default=0.0005,
        ge=0,
        le=0.05,
        description="Adverse price movement per execution (0.0005 = 0.05%). Buys fill "
        "above the reference price, sells below it.",
    )
    fractional_shares: bool = Field(
        default=True,
        description="If False, only whole shares are traded and leftover cash stays uninvested.",
    )
    risk_free_rate: float = Field(
        default=0.0,
        ge=-0.05,
        le=0.25,
        description="Annual risk-free rate used in Sharpe and Sortino ratios. Cash itself "
        "earns zero interest in this version regardless of this setting.",
    )
    periods_per_year: int = Field(default=TRADING_DAYS_PER_YEAR, ge=1, le=366)

    @field_validator("initial_capital")
    @classmethod
    def _capital_is_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("initial capital must be a finite number")
        return value

    def without_costs(self) -> "BacktestConfig":
        """The same configuration with zero fees and slippage (the 'gross' run)."""
        return self.model_copy(update={"fee_rate": 0.0, "slippage_rate": 0.0})

    def describe(self) -> dict[str, str]:
        """Human-readable summary used in the UI and in exports."""
        return {
            "Initial capital": f"${self.initial_capital:,.2f}",
            "Transaction fee": f"{self.fee_rate:.4%} of traded notional per execution",
            "Slippage": f"{self.slippage_rate:.4%} per execution (adverse to the trader)",
            "Share sizing": "Fractional shares" if self.fractional_shares else "Whole shares only",
            "Risk-free rate (ratios only)": f"{self.risk_free_rate:.2%} per year",
            "Cash interest": "None (cash earns 0%)",
            "Execution": "Signal at session close t, executed at the open of session t+1",
            "Position model": "Long-only, unleveraged, all-in or all-cash",
            "Annualisation": f"{self.periods_per_year} sessions per year",
        }


def storage_mode() -> str:
    """Where experiments and notes are stored.

    ``"session"`` (the default) keeps records in an in-memory database private to
    one browser session — safe for a public deployment because one visitor can
    never see another visitor's notes. ``"sqlite"`` persists to a local file and
    is intended for running the app on your own computer. Set it with the
    ``QLP_STORAGE`` environment variable.
    """
    mode = os.environ.get("QLP_STORAGE", "session").strip().lower()
    return mode if mode in {"session", "sqlite"} else "session"
