"""Strategy framework and the initial research strategies.

A strategy turns prices into a *target exposure* series — the fraction of the
portfolio it wants invested after the close of session ``t``:

* ``1.0`` — fully invested; ``0.0`` — all cash; values in between are partial
  positions (used by the volatility-targeted and model-based strategies).
* ``NaN`` — no valid signal yet (indicator warm-up). The engine treats this as cash.

The engine trades at the open of session ``t + 1`` when the target differs from
the holding (see :mod:`quantlab.backtest` for the rebalance band).

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
from quantlab import ml


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
    # Indicator columns drawn on the price chart.
    overlay_columns: ClassVar[tuple[str, ...]] = ()
    # True when targets are partial weights rather than strictly 0 or 1.
    continuous_exposure: ClassVar[bool] = False
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
        """Cross-parameter rules (e.g. short window < long window). Override when needed."""
        return None

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

    def oscillator_spec(self) -> dict[str, Any] | None:
        """Optional lower chart panel: ``{"title", "columns": {col: label}, "range", "levels"}``."""
        return None

    # ---- signals --------------------------------------------------------------
    @abstractmethod
    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return indicator columns plus a ``target`` column (weights in [0, 1], or NaN)."""

    def _reason(self, row: dict[str, Any], target: float) -> str:
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
        if ((valid < 0) | (valid > 1)).any():
            raise ValueError(f"{self.name} produced target weights outside [0, 1]")
        effective = target.fillna(0.0).to_numpy()
        previous = np.r_[0.0, effective[:-1]]
        columns = {c: frame[c].to_numpy() for c in frame.columns}
        reasons = np.full(len(frame), "", dtype=object)
        for i in np.flatnonzero(effective != previous):
            row = {c: values[i] for c, values in columns.items()}
            reasons[i] = self._reason(row, float(effective[i]))
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


def _deadband_steps(raw: np.ndarray, step: float) -> np.ndarray:
    """Quantise a continuous target to ``step`` increments with hysteresis.

    The target only moves when the raw signal is at least one full step away from
    the current target, which stops a signal hovering near a rounding boundary
    from flipping (and trading) every session. Causal: each value depends only on
    the current and earlier raw values.
    """
    out = np.full(len(raw), np.nan)
    current: float | None = None
    for i, x in enumerate(raw):
        if np.isnan(x):
            out[i] = np.nan if current is None else current
            continue
        if current is None or abs(x - current) >= step - 1e-12:
            current = float(np.clip(np.round(x / step) * step, 0.0, 1.0))
        out[i] = current
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

    def _reason(self, row: dict[str, Any], target: float) -> str:
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

    def _reason(self, row: dict[str, Any], target: float) -> str:
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

    def _reason(self, row: dict[str, Any], target: float) -> str:
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

    def oscillator_spec(self) -> dict[str, Any]:
        p = self.params
        return {
            "title": "RSI",
            "columns": {"rsi": f"RSI({p['window']})"},
            "range": (0, 100),
            "levels": [(p["entry_threshold"], "entry"), (p["exit_threshold"], "exit")],
        }

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi = ind.rsi_wilder(prices["close"], p["window"])
        values = rsi.to_numpy()
        valid = ~np.isnan(values)
        with np.errstate(invalid="ignore"):
            target = _hysteresis(values < p["entry_threshold"], values > p["exit_threshold"], valid)
        return pd.DataFrame({"rsi": rsi, "target": target}, index=prices.index)

    def _reason(self, row: dict[str, Any], target: float) -> str:
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

    def _reason(self, row: dict[str, Any], target: float) -> str:
        if target == 1.0:
            return f"Close fell below lower band ({row['bb_lower']:.2f})"
        level = "middle" if self.params["exit_band"] == "middle" else "upper"
        return f"Close rose above {level} band ({row['bb_' + level]:.2f})"


# --------------------------------------------------------------------------- #
# F. Adaptive trend ensemble with volatility targeting
# --------------------------------------------------------------------------- #


class AdaptiveTrendEnsemble(Strategy):
    key = "adaptive_ensemble"
    name = "Adaptive Trend Ensemble"
    family = "Multi-signal, risk-managed"
    continuous_exposure = True
    doc = StrategyDoc(
        summary=(
            "A diversified, risk-managed trend model. Five independent trend votes (time-series "
            "momentum over 1, 3, 6 and 12 months, plus price above its 200-session average) set "
            "how bullish the model is. Exposure is then scaled down when recent volatility is above "
            "a target, so the position shrinks in turbulent markets and grows in calm ones. An "
            "optional pullback component adds exposure after short-term sell-offs inside an intact "
            "up-trend."
        ),
        entry=(
            "Target weight = min(1, vol scale x (trend score + pullback boost)), moved in whole "
            "steps only once the raw value is a full step away from the current target. Trend "
            "score = share of the five trend votes that are positive."
        ),
        exit=(
            "Exposure falls as trend votes turn negative and as volatility rises; it reaches 0% when "
            "every vote is negative. The engine trades only when the held weight drifts outside the "
            "rebalance band."
        ),
        assumptions=(
            "Trends at several horizons carry information (Moskowitz, Ooi & Pedersen 2012; Hurst, "
            "Ooi & Pedersen 2017). Averaging horizons avoids betting on one look-back.",
            "Volatility is persistent, so recent volatility forecasts near-term volatility, and "
            "scaling exposure inversely to it improves risk-adjusted returns (Moreira & Muir 2017).",
            "Every component uses closes up to the signal session only; volatility is an "
            "exponentially weighted estimate of daily close-to-close returns.",
            "Long-only and capped at 100%: volatility targeting can reduce but never lever exposure.",
        ),
        failure_modes=(
            "Sharp V-shaped reversals: trend votes flip late, so the model sells near lows and "
            "re-buys after the rebound.",
            "Volatility spikes after a crash cut exposure just before the recovery.",
            "Range-bound markets produce partial, frequently adjusted positions and cost drag.",
            "Five votes share the same price history, so they are correlated — the ensemble is "
            "less diversified than it looks.",
        ),
        references=(
            "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE.",
            "Moreira & Muir (2017), 'Volatility-Managed Portfolios', Journal of Finance.",
            "Hurst, Ooi & Pedersen (2017), 'A Century of Evidence on Trend-Following Investing'.",
        ),
    )
    param_specs = (
        ParamSpec("target_vol", "Target volatility (annual)", "float", 0.15, 0.03, 0.60, 0.01,
                  help="Exposure is scaled by target / recent volatility (capped at 100%)."),
        ParamSpec("vol_span", "Volatility half-life span (sessions)", "int", 20, 5, 126, 1,
                  help="Span of the exponentially weighted volatility estimate."),
        ParamSpec("pullback_boost", "Pullback boost", "float", 0.25, 0.0, 1.0, 0.05,
                  help="Extra exposure when RSI(3) < 20 while price is above its 200-session average."),
        ParamSpec("step", "Exposure step", "float", 0.2, 0.05, 0.5, 0.05,
                  help="Targets move in increments of this size, and only when the raw signal is a "
                       "full step away — larger steps mean fewer trades."),
    )
    search_space = {"target_vol": (0.10, 0.15, 0.20), "pullback_boost": (0.0, 0.25)}
    HORIZONS = (21, 63, 126, 252)

    @property
    def warmup_sessions(self) -> int:
        return max(self.HORIZONS)

    @property
    def label(self) -> str:
        p = self.params
        return f"{self.name} (vol {p['target_vol']:.0%}, boost {p['pullback_boost']:g})"

    def oscillator_spec(self) -> dict[str, Any]:
        return {
            "title": "Model state",
            "columns": {"trend_score": "Trend score", "vol_scale": "Volatility scale"},
            "range": (0, 1.05),
            "levels": [],
        }

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = prices["close"]
        votes = [ind.trailing_return(close, h) > 0 for h in self.HORIZONS]
        sma200 = ind.sma(close, 200)
        votes.append(close > sma200)
        valid = pd.concat([ind.trailing_return(close, h) for h in self.HORIZONS] + [sma200], axis=1).notna().all(axis=1)
        trend_score = pd.concat(votes, axis=1).astype("float64").mean(axis=1)

        daily_ret = close.pct_change()
        realised_vol = daily_ret.ewm(span=p["vol_span"], adjust=False, min_periods=p["vol_span"]).std() * math.sqrt(252)
        vol_scale = (p["target_vol"] / realised_vol.where(realised_vol > 0)).clip(upper=1.0)

        rsi3 = ind.rsi_wilder(close, 3)
        pullback = ((rsi3 < 20) & (close > sma200)).astype("float64")
        raw = (vol_scale * (trend_score + p["pullback_boost"] * pullback)).clip(0.0, 1.0)
        ok = valid & vol_scale.notna() & rsi3.notna()
        target = pd.Series(_deadband_steps(raw.where(ok).to_numpy(), p["step"]), index=prices.index)
        return pd.DataFrame(
            {
                "trend_score": trend_score.where(ok),
                "vol_scale": vol_scale.where(ok),
                "realised_vol": realised_vol.where(ok),
                "pullback": pullback.where(ok),
                "target": target.where(ok),
            },
            index=prices.index,
        )

    def _reason(self, row: dict[str, Any], target: float) -> str:
        boost = " + pullback boost" if row.get("pullback") == 1.0 and self.params["pullback_boost"] > 0 else ""
        return (
            f"Target {target:.0%}: trend score {row['trend_score']:.0%}{boost}, volatility "
            f"{row['realised_vol']:.1%} vs target {self.params['target_vol']:.0%} (scale {row['vol_scale']:.2f})"
        )


# --------------------------------------------------------------------------- #
# G. Walk-forward machine-learning classifier
# --------------------------------------------------------------------------- #


class WalkForwardLogistic(Strategy):
    key = "ml_logistic"
    name = "Walk-Forward ML Classifier"
    family = "Machine learning"
    continuous_exposure = True
    doc = StrategyDoc(
        summary=(
            "A regularised logistic-regression model estimates the probability that the asset will "
            "be higher in N sessions, from twelve causal features (multi-horizon returns, volatility, "
            "volatility regime, distance from moving averages, RSI, drawdown). It is retrained on a "
            "rolling window as time passes, using only outcomes that were already known, and its "
            "confidence sets the position size."
        ),
        entry=(
            "Target weight rises linearly from 0% at a 50% predicted probability of a rise to 100% "
            "at the 'full-confidence' probability, in 20% steps that only change once the "
            "probability has moved a full step (to avoid trading on noise)."
        ),
        exit="Weight falls to 0% whenever the predicted probability of a rise is 50% or lower.",
        assumptions=(
            "Labels are purged: a refit at session r uses only samples whose N-session outcome was "
            "known by r. Feature scaling is estimated on the training window only.",
            "The model is refit every few sessions and frozen in between, so every prediction is "
            "out-of-sample. Its hit rate and AUC are reported on the Strategy Lab page.",
            "Relationships between these features and future returns are weak and unstable; strong "
            "regularisation and a small feature set limit overfitting.",
        ),
        failure_modes=(
            "Regime change: patterns learned in the training window may stop working.",
            "Overlapping labels make the effective sample size about N times smaller than it "
            "looks, so apparent accuracy edges are noisy.",
            "A model trained mostly in bull markets learns to be bullish — compare its hit rate "
            "with the 'always up' base rate before reading anything into it.",
            "Tuning horizon, window and regularisation on the same data reintroduces overfitting "
            "through the back door — use the Walk-Forward page.",
        ),
        references=(
            "López de Prado (2018), 'Advances in Financial Machine Learning' (purging and embargo).",
            "Gu, Kelly & Xiu (2020), 'Empirical Asset Pricing via Machine Learning', RFS.",
        ),
    )
    param_specs = (
        ParamSpec("horizon", "Forecast horizon (sessions)", "int", 21, 1, 126, 1,
                  help="The model predicts whether the price is higher this many sessions ahead."),
        ParamSpec("train_window", "Training window (sessions)", "int", 756, 126, 2520, 21,
                  help="Rolling window of labelled history used at each refit (756 ≈ 3 years)."),
        ParamSpec("retrain_every", "Refit every (sessions)", "int", 21, 1, 252, 1,
                  help="How often the model is refit; it is frozen between refits."),
        ParamSpec("ridge", "Regularisation strength", "float", 0.1, 0.0, 10.0, 0.01,
                  help="L2 penalty on standardised coefficients. Higher = simpler, more stable model."),
        ParamSpec("full_confidence", "Full-position probability", "float", 0.60, 0.51, 0.95, 0.01,
                  help="Predicted probability of a rise at which the model is 100% invested."),
        ParamSpec("smoothing", "Probability smoothing (sessions)", "int", 5, 1, 63, 1,
                  help="Span of an exponential average of predicted probabilities (1 = none). "
                       "Reduces trading on day-to-day noise at the cost of some lag."),
    )
    search_space = {"horizon": (10, 21), "ridge": (0.1, 1.0)}

    @property
    def warmup_sessions(self) -> int:
        # Features need 252 sessions; the first model needs 252 labelled samples plus the horizon.
        return ml.FEATURE_WARMUP + 252 + self.params["horizon"] - 1

    @property
    def label(self) -> str:
        p = self.params
        return f"{self.name} ({p['horizon']}d horizon, refit {p['retrain_every']}d, ridge {p['ridge']:g})"

    def oscillator_spec(self) -> dict[str, Any]:
        return {
            "title": "P(up)",
            "columns": {"prob_up": f"P(up over {self.params['horizon']}d)"},
            "range": None,
            "levels": [(0.5, "neutral"), (self.params["full_confidence"], "full")],
        }

    def walk_forward_fit(self, prices: pd.DataFrame) -> ml.WalkForwardFit:
        p = self.params
        return ml.walk_forward_probabilities(
            prices, horizon=p["horizon"], train_window=p["train_window"],
            retrain_every=p["retrain_every"], ridge=p["ridge"], min_train=252,
        )

    def _compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        raw_prob = self.walk_forward_fit(prices).probabilities
        prob = raw_prob.ewm(span=p["smoothing"], adjust=False).mean().where(raw_prob.notna())
        scaled = ((prob - 0.5) / (p["full_confidence"] - 0.5)).clip(0.0, 1.0)
        target = pd.Series(_deadband_steps(scaled.to_numpy(), 0.2), index=prices.index)
        return pd.DataFrame({"prob_up": prob, "prob_up_raw": raw_prob, "target": target.where(prob.notna())},
                            index=prices.index)

    def _reason(self, row: dict[str, Any], target: float) -> str:
        return (
            f"Model P(up over {self.params['horizon']}d) = {row['prob_up']:.1%} -> target {target:.0%}"
        )


STRATEGY_CLASSES: tuple[type[Strategy], ...] = (
    BuyAndHold,
    MovingAverageCrossover,
    TimeSeriesMomentum,
    RSIMeanReversion,
    BollingerMeanReversion,
    AdaptiveTrendEnsemble,
    WalkForwardLogistic,
)
STRATEGIES: dict[str, type[Strategy]] = {cls.key: cls for cls in STRATEGY_CLASSES}


def create_strategy(key: str, params: dict[str, Any] | None = None) -> Strategy:
    if key not in STRATEGIES:
        raise StrategyParameterError(f"Unknown strategy '{key}'. Available: {', '.join(STRATEGIES)}.")
    return STRATEGIES[key](**(params or {}))
