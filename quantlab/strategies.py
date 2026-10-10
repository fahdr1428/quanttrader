"""Strategy framework and the initial research strategies.

A strategy turns prices into a *target exposure* series:

* ``1.0`` — after the close of session ``t`` the strategy wants to be fully
  invested; the engine buys at the open of session ``t + 1`` if it is not.
* ``0.0`` — the strategy wants to hold cash.
* ``NaN`` — no valid signal yet (indicator warm-up). The engine treats this as cash.

Strategies never see execution prices, cash or positions. That separation keeps
signal logic testable on its own and makes it impossible for a strategy to peek
at the price it will trade at.

These strategies are research hypotheses, not recommendations, and nothing here
claims that any of them is profitable.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from quantlab import indicators as ind


class StrategyParameterError(ValueError):
    """Raised when a strategy is configured with invalid or nonsensical parameters."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: Literal["int", "float", "choice"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = ()
    help: str = ""

    def coerce(self, value: Any) -> Any:
        if self.kind == "choice":
            if value not in self.choices:
                raise StrategyParameterError(
                    f"{self.label}: '{value}' is not one of {', '.join(map(str, self.choices))}."
                )
            return value
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyParameterError(f"{self.label} must be a number, got {value!r}.") from exc
        if not math.isfinite(number):
            raise StrategyParameterError(f"{self.label} must be finite.")
        if self.kind == "int":
            if number != int(number):
                raise StrategyParameterError(f"{self.label} must be a whole number, got {value!r}.")
            number = int(number)
        if self.min is not None and number < self.min:
            raise StrategyParameterError(f"{self.label} must be at least {self.min:g}, got {number:g}.")
        if self.max is not None and number > self.max:
            raise StrategyParameterError(f"{self.label} must be at most {self.max:g}, got {number:g}.")
        return number


@dataclass(frozen=True)
class StrategyDoc:
    summary: str
    entry: str
    exit: str
    assumptions: tuple[str, ...]
    failure_modes: tuple[str, ...]
    references: tuple[str, ...] = field(default_factory=tuple)


class Strategy(ABC):
    """Base class. Subclasses declare parameters and implement ``_compute``."""

    key: ClassVar[str]
    name: ClassVar[str]
    family: ClassVar[str]
    doc: ClassVar[StrategyDoc]
    param_specs: ClassVar[tuple[ParamSpec, ...]] = ()
    # Indicator columns drawn on the price chart, and in a separate oscillator panel.
    overlay_columns: ClassVar[tuple[str, ...]] = ()
    oscillator_columns: ClassVar[tuple[str, ...]] = ()
    # Constrained search space used by sensitivity analysis and walk-forward research.
    search_space: ClassVar[dict[str, tuple[Any, ...]]] = {}

    def __init__(self, **params: Any):
        self.params = self.validate_params(params)

    # ---- parameters ---------------------------------------------------------
    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {spec.name: spec.default for spec in cls.param_specs}

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        known = {spec.name for spec in cls.param_specs}
        unknown = set(params) - known
        if unknown:
            raise StrategyParameterError(
                f"Unknown parameter(s) for {cls.name}: {', '.join(sorted(unknown))}."
            )
        merged = {**cls.defaults(), **params}
        clean = {spec.name: spec.coerce(merged[spec.name]) for spec in cls.param_specs}
        cls._check_constraints(clean)
        return clean

    @classmethod
    def _check_constraints(cls, params: dict[str, Any]) -> None:
        """Cross-parameter rules (e.g. short window < long window)."""

    @classmethod
    def is_valid(cls, params: dict[str, Any]) -> bool:
        try:
            cls.validate_params(params)
            return True
        except StrategyParameterError:
            return False

    @classmethod
    def parameter_grid(cls, space: dict[str, tuple[Any, ...]] | None = None) -> list[dict[str, Any]]:
        """All valid combinations of a (constrained) search space, in a deterministic order."""
        space = space or cls.search_space
        if not space:
            return [cls.defaults()]
        names = list(space)
        combos = []
        for values in itertools.product(*(space[n] for n in names)):
            candidate = {**cls.defaults(), **dict(zip(names, values))}
            if cls.is_valid(candidate):
                combos.append(cls.validate_params(candidate))
        return combos

    def uses_default_parameters(self) -> bool:
        return self.params == self.defaults()

    @property
    @abstractmethod
    def warmup_sessions(self) -> int:
        """Sessions of history needed before the first valid signal."""

    @property
    def label(self) -> str:
        return self.name

    # ---- signals --------------------------------------------------------------
    @abstractmethod
    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return indicator columns plus a ``target`` column (1.0 / 0.0 / NaN)."""

    def _reason(self, row: pd.Series, target: float) -> str:
        return "Enter long" if target == 1.0 else "Exit to cash"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators, target exposure and a human-readable reason per change.

        The ``reason`` column is filled only on sessions where the target differs
        from the previous session (warm-up NaN is treated as cash), i.e. where an
        order would be generated.
        """
        if "close" not in prices.columns:
            raise ValueError("prices must contain a 'close' column")
        frame = self._compute(prices)
        target = frame["target"]
        valid = target.dropna()
        if not valid.isin([0.0, 1.0]).all():
            raise ValueError(f"{self.name} produced targets other than 0, 1 or NaN")
        effective = target.fillna(0.0)
        changed = effective.ne(effective.shift(1).fillna(0.0))
        reasons = pd.Series("", index=frame.index, dtype=object)
        for ts in frame.index[changed]:
            reasons.loc[ts] = self._reason(frame.loc[ts], float(effective.loc[ts]))
        frame["reason"] = reasons
        return frame


def _hysteresis(enter: np.ndarray, exit_: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Stateful long/cash switch used by the mean-reversion strategies.

    Flat -> long when ``enter`` is true; long -> flat when ``exit_`` is true;
    otherwise the previous state is held. Holding the state is what stops a
    persistent condition (e.g. RSI staying below 30 for a week) from generating a
    new buy order every day.
    """
    out = np.full(len(enter), np.nan)
    position = 0.0
    started = False
    for i in range(len(enter)):
        if not valid[i]:
            out[i] = position if started else np.nan
            continue
        started = True
        if position == 0.0 and enter[i]:
            position = 1.0
        elif position == 1.0 and exit_[i]:
            position = 0.0
        out[i] = position
    return out


# --------------------------------------------------------------------------- #
# A. Buy and hold
# --------------------------------------------------------------------------- #


class BuyAndHold(Strategy):
    key = "buy_and_hold"
    name = "Buy & Hold"
    family = "Passive"
    doc = StrategyDoc(
        summary=(
            "Invest all capital at the first opportunity and hold until the end of the sample. "
            "It is the natural benchmark: an active strategy must justify its extra trading, "
            "costs and complexity against it."
        ),
        entry=(
            "A target of 100% invested is set at the first session's close, so the purchase "
            "executes at the next session's open — the same timing convention every other "
            "strategy uses."
        ),
        exit="Never exits. The position is valued at the final close (mark-to-market).",
        assumptions=(
            "The asset is held through every drawdown with no risk management.",
            "Dividends are captured only when total-return-adjusted prices are used.",
        ),
        failure_modes=(
            "Suffers the full depth of bear markets (e.g. 2008, 2020, 2022).",
            "A single asset can underperform cash for many years.",
        ),
    )

    @property
    def warmup_sessions(self) -> int:
        return 0

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"target": 1.0}, index=prices.index)

    def _reason(self, row: pd.Series, target: float) -> str:
        return "Buy-and-hold: invest at the first available opportunity"


# --------------------------------------------------------------------------- #
# B. Moving-average trend following
# --------------------------------------------------------------------------- #


class MovingAverageCrossover(Strategy):
    key = "ma_crossover"
    name = "Moving-Average Trend"
    family = "Trend following"
    doc = StrategyDoc(
        summary=(
            "Trend following: be invested while the short-term average price is above the "
            "long-term average (an up-trend), otherwise hold cash. The aim is to sidestep "
            "prolonged declines at the cost of some whipsaw trades."
        ),
        entry="Long when the short moving average closes above the long moving average.",
        exit="Cash when the short moving average closes at or below the long moving average.",
        assumptions=(
            "Trends persist long enough for a lagging signal to capture part of them.",
            "Both averages use closing prices up to and including the signal session.",
            "No signal is produced until the long average has a full window of data.",
        ),
        failure_modes=(
            "Sideways, choppy markets cause repeated false crossovers (whipsaws) and cost drag.",
            "Moving averages lag: sharp V-shaped recoveries are re-entered late.",
            "Overnight gaps mean exits can fill well below the signal-day close.",
        ),
        references=("Faber (2007), 'A Quantitative Approach to Tactical Asset Allocation'.",),
    )
    param_specs = (
        ParamSpec("short_window", "Short window (sessions)", "int", 20, 2, 250, 1,
                  help="Length of the fast moving average."),
        ParamSpec("long_window", "Long window (sessions)", "int", 50, 3, 400, 1,
                  help="Length of the slow moving average. Must exceed the short window."),
        ParamSpec("ma_type", "Average type", "choice", "SMA", choices=("SMA", "EMA"),
                  help="Simple (equal weights) or exponential (recent prices weigh more)."),
    )
    overlay_columns = ("ma_short", "ma_long")
    search_space = {"short_window": (10, 20, 30, 50), "long_window": (50, 100, 150, 200)}

    @classmethod
    def _check_constraints(cls, params: dict[str, Any]) -> None:
        if params["short_window"] >= params["long_window"]:
            raise StrategyParameterError(
                "The short window must be shorter than the long window "
                f"(got {params['short_window']} and {params['long_window']})."
            )

    @property
    def warmup_sessions(self) -> int:
        return self.params["long_window"] - 1

    @property
    def label(self) -> str:
        p = self.params
        return f"{self.name} ({p['ma_type']} {p['short_window']}/{p['long_window']})"

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        avg = ind.sma if p["ma_type"] == "SMA" else ind.ema
        short = avg(prices["close"], p["short_window"])
        long = avg(prices["close"], p["long_window"])
        target = (short > long).astype("float64").where(short.notna() & long.notna())
        return pd.DataFrame({"ma_short": short, "ma_long": long, "target": target})

    def _reason(self, row: pd.Series, target: float) -> str:
        p = self.params
        rel = "above" if target == 1.0 else "at/below"
        return (
            f"{p['ma_type']}{p['short_window']} ({row['ma_short']:.2f}) closed {rel} "
            f"{p['ma_type']}{p['long_window']} ({row['ma_long']:.2f})"
        )


# --------------------------------------------------------------------------- #
# C. Time-series momentum
# --------------------------------------------------------------------------- #


class TimeSeriesMomentum(Strategy):
    key = "ts_momentum"
    name = "Time-Series Momentum"
    family = "Trend following"
    doc = StrategyDoc(
        summary=(
            "Absolute (time-series) momentum: if the asset's own trailing return is above a "
            "threshold, assume the trend continues and be invested; otherwise hold cash."
        ),
        entry="Long when the trailing return over the lookback window exceeds the threshold.",
        exit="Cash when the trailing return is at or below the threshold.",
        assumptions=(
            "Past returns over months carry information about the next month's direction.",
            "The trailing return at session t uses closes up to t (or t - skip) only.",
            "The threshold is compared with the raw return, not a risk-adjusted one.",
        ),
        failure_modes=(
            "Momentum crashes: sharp trend reversals after long rallies or sell-offs.",
            "A long lookback reacts slowly; a short one trades frequently and pays more costs.",
            "Results are sensitive to the exact lookback — a classic overfitting trap.",
        ),
        references=("Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE.",),
    )
    param_specs = (
        ParamSpec("lookback", "Lookback (sessions)", "int", 252, 2, 504, 1,
                  help="Window for the trailing return. 252 sessions is about 12 months."),
        ParamSpec("skip", "Skip most recent (sessions)", "int", 0, 0, 63, 1,
                  help="Exclude the latest sessions from the return (e.g. 21 for '12-1' momentum)."),
        ParamSpec("threshold", "Return threshold", "float", 0.0, -0.5, 0.5, 0.01,
                  help="Minimum trailing return to be invested (0.0 = any positive return)."),
    )
    search_space = {"lookback": (63, 126, 189, 252), "skip": (0, 21)}

    @classmethod
    def _check_constraints(cls, params: dict[str, Any]) -> None:
        if params["skip"] >= params["lookback"]:
            raise StrategyParameterError("Skip must be smaller than the lookback window.")

    @property
    def warmup_sessions(self) -> int:
        return self.params["lookback"] + self.params["skip"]

    @property
    def label(self) -> str:
        p = self.params
        skip = f", skip {p['skip']}" if p["skip"] else ""
        return f"{self.name} ({p['lookback']}d{skip}, >{p['threshold']:.0%})"

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        tr = ind.trailing_return(prices["close"], p["lookback"], p["skip"])
        target = (tr > p["threshold"]).astype("float64").where(tr.notna())
        return pd.DataFrame({"trailing_return": tr, "target": target})

    def _reason(self, row: pd.Series, target: float) -> str:
        p = self.params
        rel = "above" if target == 1.0 else "at/below"
        return f"{p['lookback']}-session trailing return {row['trailing_return']:+.2%} {rel} threshold {p['threshold']:+.2%}"


# --------------------------------------------------------------------------- #
# D. RSI mean reversion
# --------------------------------------------------------------------------- #


class RSIMeanReversion(Strategy):
    key = "rsi_reversion"
    name = "RSI Mean Reversion"
    family = "Mean reversion"
    doc = StrategyDoc(
        summary=(
            "Buy weakness: when the Relative Strength Index signals the asset is oversold, buy "
            "and wait for a rebound; sell once momentum has recovered to the exit level."
        ),
        entry="Long when RSI closes below the entry threshold while flat.",
        exit="Cash when RSI closes above the exit threshold while invested.",
        assumptions=(
            "Short-term price moves partially reverse (mean reversion).",
            "Between the entry and exit levels the position is simply held — a persistent "
            "oversold reading does not create additional orders.",
            "RSI uses Wilder's smoothing on closing prices up to the signal session.",
        ),
        failure_modes=(
            "'Catching a falling knife': in a crash RSI stays oversold while prices keep falling.",
            "No stop-loss: losses are unbounded until RSI recovers.",
            "Long periods in cash during steady up-trends, lagging buy-and-hold.",
        ),
        references=("Wilder (1978), 'New Concepts in Technical Trading Systems'.",),
    )
    param_specs = (
        ParamSpec("window", "RSI window (sessions)", "int", 14, 2, 100, 1,
                  help="Smoothing period for Wilder's RSI."),
        ParamSpec("entry_threshold", "Entry: RSI below", "float", 30.0, 1.0, 99.0, 1.0,
                  help="Buy when RSI closes below this oversold level."),
        ParamSpec("exit_threshold", "Exit: RSI above", "float", 50.0, 1.0, 99.0, 1.0,
                  help="Sell when RSI closes above this level. Must exceed the entry level."),
    )
    oscillator_columns = ("rsi",)
    search_space = {"window": (7, 14, 21), "entry_threshold": (20.0, 25.0, 30.0, 35.0), "exit_threshold": (50.0, 60.0, 70.0)}

    @classmethod
    def _check_constraints(cls, params: dict[str, Any]) -> None:
        if params["entry_threshold"] >= params["exit_threshold"]:
            raise StrategyParameterError(
                "The RSI entry threshold must be below the exit threshold "
                f"(got {params['entry_threshold']:g} and {params['exit_threshold']:g})."
            )

    @property
    def warmup_sessions(self) -> int:
        return self.params["window"]

    @property
    def label(self) -> str:
        p = self.params
        return f"{self.name} (RSI{p['window']} <{p['entry_threshold']:g} / >{p['exit_threshold']:g})"

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi = ind.rsi_wilder(prices["close"], p["window"])
        values = rsi.to_numpy()
        valid = ~np.isnan(values)
        with np.errstate(invalid="ignore"):
            target = _hysteresis(values < p["entry_threshold"], values > p["exit_threshold"], valid)
        return pd.DataFrame({"rsi": rsi, "target": target}, index=prices.index)

    def _reason(self, row: pd.Series, target: float) -> str:
        p = self.params
        if target == 1.0:
            return f"RSI{p['window']} {row['rsi']:.1f} closed below entry level {p['entry_threshold']:g}"
        return f"RSI{p['window']} {row['rsi']:.1f} closed above exit level {p['exit_threshold']:g}"


# --------------------------------------------------------------------------- #
# E. Bollinger-band mean reversion
# --------------------------------------------------------------------------- #


class BollingerMeanReversion(Strategy):
    key = "bollinger_reversion"
    name = "Bollinger Mean Reversion"
    family = "Mean reversion"
    doc = StrategyDoc(
        summary=(
            "Buy when the price closes unusually far below its recent average (beneath the "
            "lower Bollinger band) and sell when it has reverted to the average."
        ),
        entry="Long when the close is below the lower band while flat.",
        exit="Cash when the close rises above the middle band (or the upper band, if selected).",
        assumptions=(
            "Over the band window, prices fluctuate around a stable local mean.",
            "Band width (rolling standard deviation) is a fair measure of 'unusual'; it assumes "
            "roughly symmetric, stable volatility, which real returns do not have.",
            "Bands use closes up to and including the signal session (population std, ddof=0).",
        ),
        failure_modes=(
            "Trending markets: price can 'walk the band' downward for weeks.",
            "Volatility regime shifts widen the bands after the damage is done.",
            "No stop-loss; a structural break (e.g. bankruptcy) is treated as a bargain.",
        ),
        references=("Bollinger (2001), 'Bollinger on Bollinger Bands'.",),
    )
    param_specs = (
        ParamSpec("window", "Band window (sessions)", "int", 20, 5, 200, 1,
                  help="Moving-average and standard-deviation window."),
        ParamSpec("num_std", "Band width (std devs)", "float", 2.0, 0.5, 4.0, 0.1,
                  help="Distance of the bands from the middle in standard deviations."),
        ParamSpec("exit_band", "Exit at", "choice", "middle", choices=("middle", "upper"),
                  help="Sell when the close crosses above the middle band or the upper band."),
    )
    overlay_columns = ("bb_middle", "bb_upper", "bb_lower")
    search_space = {"window": (10, 20, 30, 40), "num_std": (1.5, 2.0, 2.5)}

    @property
    def warmup_sessions(self) -> int:
        return self.params["window"] - 1

    @property
    def label(self) -> str:
        p = self.params
        return f"{self.name} ({p['window']}d, {p['num_std']:g}σ, exit {p['exit_band']})"

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        bands = ind.bollinger_bands(prices["close"], p["window"], p["num_std"])
        close = prices["close"].to_numpy()
        lower = bands["bb_lower"].to_numpy()
        exit_level = bands["bb_middle" if p["exit_band"] == "middle" else "bb_upper"].to_numpy()
        valid = ~np.isnan(lower)
        with np.errstate(invalid="ignore"):
            target = _hysteresis(close < lower, close > exit_level, valid)
        out = bands.copy()
        out["target"] = target
        return out

    def _reason(self, row: pd.Series, target: float) -> str:
        if target == 1.0:
            return f"Close fell below lower band ({row['bb_lower']:.2f})"
        level = "middle" if self.params["exit_band"] == "middle" else "upper"
        return f"Close rose above {level} band ({row['bb_' + level]:.2f})"


STRATEGY_CLASSES: tuple[type[Strategy], ...] = (
    BuyAndHold,
    MovingAverageCrossover,
    TimeSeriesMomentum,
    RSIMeanReversion,
    BollingerMeanReversion,
)
STRATEGIES: dict[str, type[Strategy]] = {cls.key: cls for cls in STRATEGY_CLASSES}


def create_strategy(key: str, params: dict[str, Any] | None = None) -> Strategy:
    if key not in STRATEGIES:
        raise StrategyParameterError(f"Unknown strategy '{key}'. Available: {', '.join(STRATEGIES)}.")
    return STRATEGIES[key](**(params or {}))
