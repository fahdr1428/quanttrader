"""Session state, sidebar controls and cached computations.

Everything that determines a backtest is collected here into two objects:

* :class:`ResearchData` — the validated prices (asset and benchmark, aligned to
  common sessions) and the evaluation window.
* :class:`RunSpec` — the strategy, its parameters and the cost/capital settings.

Pages read these two objects and never re-implement any calculation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.analytics import StrategyRun, run_strategy
from quantlab.backtest import BacktestResult, run_buy_and_hold
from quantlab.config import BacktestConfig, data_dir, storage_mode
from quantlab.data import (
    ADJUSTMENT_LABELS,
    AdjustmentMode,
    CacheStatus,
    CSVProvider,
    DataProviderError,
    DiskCache,
    MarketDataset,
    SyntheticProvider,
    YahooFinanceProvider,
    align_benchmark,
    fingerprint_prices,
    load_market_data,
)
from quantlab.experiments import ExperimentStore
from quantlab.research import per_period_sharpe
from quantlab.simulation import DEMO_SEEDS, SIM_END, SIM_START, SYNTHETIC_MARKETS, demo_ticker
from quantlab.strategies import STRATEGIES, ParamSpec, StrategyParameterError, create_strategy
from quantlab.validation import InputValidationError, validate_date_range

DEFAULT_TICKER = "SPY"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_START = date(2015, 1, 2)
DEFAULT_WARMUP_DAYS = 800  # ~550 sessions: enough for the ML model's warm-up
DEFAULT_STRATEGY = "ma_crossover"

SOURCE_YAHOO = "Yahoo Finance"
SOURCE_CSV = "CSV file"
SOURCE_DEMO = "Synthetic demo"
BENCH_SAME = "Same asset (buy & hold)"
BENCH_TICKER = "Another ticker"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class ResearchData:
    asset: MarketDataset
    benchmark_dataset: MarketDataset | None
    benchmark_ticker: str
    prices: pd.DataFrame
    bench_prices: pd.DataFrame
    dropped_dates: pd.DatetimeIndex
    fingerprint: str
    bench_fingerprint: str
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    requested_start: date
    requested_end: date
    warmup_days: int
    notes: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        bench_ok = self.benchmark_dataset is None or self.benchmark_dataset.is_usable
        return self.asset.is_usable and bench_ok and len(self.eval_prices) >= 2

    @property
    def benchmark_label(self) -> str:
        if self.benchmark_dataset is None:
            return f"{self.asset.ticker} buy & hold"
        return f"{self.benchmark_ticker} buy & hold"

    @property
    def same_asset_benchmark(self) -> bool:
        return self.benchmark_dataset is None

    @property
    def eval_prices(self) -> pd.DataFrame:
        p = self.prices
        return p.loc[(p.index >= self.eval_start) & (p.index <= self.eval_end)]

    @property
    def warmup_sessions_available(self) -> int:
        return int((self.prices.index < self.eval_start).sum())

    @cached_property
    def sample_key(self) -> str:
        """Identity of the evaluated sample, used to group trials.

        Built from the prices inside the evaluation window only, so loading more
        or less warm-up history does not reset the multiple-testing count.
        """
        bench_eval = self.bench_prices.loc[self.eval_prices.index] if len(self.eval_prices) else self.bench_prices
        return (f"{fingerprint_prices(self.eval_prices)[:16]}|{fingerprint_prices(bench_eval)[:16]}|"
                f"{self.eval_start.date()}|{self.eval_end.date()}")


@dataclass(frozen=True)
class RunSpec:
    strategy_key: str
    params: dict[str, Any]
    config: BacktestConfig
    declared_preselected: bool = False

    @property
    def strategy(self):
        return create_strategy(self.strategy_key, self.params)

    @property
    def uses_defaults(self) -> bool:
        return self.params == STRATEGIES[self.strategy_key].defaults()

    @property
    def parameter_selection(self) -> str:
        if self.uses_defaults:
            return "Pre-specified defaults (fixed before viewing this sample)"
        if self.declared_preselected:
            return "Custom — declared fixed before viewing results (self-reported)"
        return "Custom — may have been tuned after viewing results (treat as in-sample)"

    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)


# --------------------------------------------------------------------------- #
# Cached data loading (shared across sessions: public market data only)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=32)
def _fetch_yahoo(ticker: str, start: str, end: str, adjustment: str, retries: int, _force: bool = False) -> MarketDataset:
    return load_market_data(
        ticker,
        start,
        end,
        YahooFinanceProvider(retries=retries),
        adjustment=AdjustmentMode(adjustment),
        cache=DiskCache(data_dir() / "cache"),
        force_refresh=_force,
    )


def _yahoo(ticker: str, start: str, end: str, adjustment: str, retries: int = 2, force: bool = False) -> MarketDataset:
    """Shared, memoised Yahoo download.

    A forced refresh replaces the shared memo entry (so the refreshed data is
    what every later load sees), and a stale-cache fallback is never memoised,
    so the next load tries the provider again.
    """
    args = (ticker, start, end, adjustment, retries)
    if force:
        _fetch_yahoo.clear(*args)
    dataset = _fetch_yahoo(*args, _force=force)
    if dataset.cache_status is CacheStatus.STALE_CACHE:
        _fetch_yahoo.clear(*args)
    return dataset


def _load_csv(content: bytes, filename: str, ticker: str, start: str, end: str, adjustment: str) -> MarketDataset:
    return load_market_data(ticker, start, end, CSVProvider(content, filename), adjustment=AdjustmentMode(adjustment))


@st.cache_data(show_spinner=False, max_entries=16)
def _load_synthetic(ticker: str, start: str, end: str) -> MarketDataset:
    return load_market_data(ticker, start, end, SyntheticProvider(), adjustment=AdjustmentMode.PRICE_ONLY,
                            today=SIM_END.date())


def load_research_data(
    *,
    source: str,
    ticker: str,
    benchmark_mode: str,
    benchmark_ticker: str,
    start: date,
    end: date,
    adjustment: str,
    warmup_days: int,
    csv_content: bytes | None = None,
    csv_name: str = "upload.csv",
    force_refresh: bool = False,
    demo_market: str = "random",
    demo_seed: int = 1,
    retries: int = 2,
) -> ResearchData:
    """Fetch, validate and align asset and benchmark prices. Raises on failure."""
    notes: list[str] = []
    if source == SOURCE_DEMO:
        if end > SIM_END.date():
            end = SIM_END.date()
            notes.append(f"The synthetic series ends on {SIM_END:%Y-%m-%d}; the evaluation window stops there.")
        if start < (SIM_START + timedelta(days=int(warmup_days))).date():
            start = (SIM_START + timedelta(days=int(warmup_days))).date()
            notes.append(f"The synthetic series starts on {SIM_START:%Y-%m-%d}; the evaluation start was moved "
                         "so the full warm-up history is available.")
        if benchmark_mode == BENCH_TICKER:
            notes.append("Synthetic demo data is always compared with buy & hold of the same simulated series.")
            benchmark_mode = BENCH_SAME
    # The minimum sample length applies to the evaluation window itself, not to the
    # warm-up-inclusive download range.
    validate_date_range(start, end)
    fetch_start = (pd.Timestamp(start) - timedelta(days=int(warmup_days))).date()
    if source == SOURCE_CSV:
        if not csv_content:
            raise InputValidationError("Choose a CSV file to upload first.")
        if ticker.strip().upper() == DEFAULT_TICKER:
            # Do not label someone's file with a real security's symbol it may not contain.
            stem = re.sub(r"[^A-Za-z0-9.\-]", "", Path(csv_name).stem).upper().lstrip(".-")[:15]
            ticker = stem or "UPLOAD"
            notes.append(f"Series labelled '{ticker}' from the file name; enter a label in the Ticker box to change it.")
        asset = _load_csv(csv_content, csv_name, ticker, str(fetch_start), str(end), adjustment)
    elif source == SOURCE_DEMO:
        asset = _load_synthetic(demo_ticker(demo_market, demo_seed), str(fetch_start), str(end))
    else:
        asset = _yahoo(ticker.strip().upper(), str(fetch_start), str(end), adjustment, retries, force_refresh)

    bench_ds: MarketDataset | None = None
    prices = asset.prices
    bench_prices = prices
    dropped = pd.DatetimeIndex([])
    bench_symbol = asset.ticker
    if benchmark_mode == BENCH_TICKER and benchmark_ticker.strip().upper() != asset.ticker:
        bench_symbol = benchmark_ticker.strip().upper()
        try:
            bench_ds = _yahoo(bench_symbol, str(fetch_start), str(end), adjustment, retries, force_refresh)
        except DataProviderError as exc:
            raise type(exc)(f"The benchmark '{bench_symbol}' could not be loaded: {exc.user_message} "
                            "Choose 'Same asset (buy & hold)' as the benchmark to continue without it.",
                            detail=exc.detail) from exc
        if bench_ds.is_usable and asset.is_usable:
            prices, bench_prices, dropped = align_benchmark(asset.prices, bench_ds.prices)
            lo = max(asset.prices.index[0], bench_ds.prices.index[0])
            hi = min(asset.prices.index[-1], bench_ds.prices.index[-1])
            inside = dropped[(dropped >= lo) & (dropped <= hi)]
            outside = len(dropped) - len(inside)
            if outside:
                notes.append(f"The sample is limited to the dates both series cover, {lo:%Y-%m-%d} to {hi:%Y-%m-%d} "
                             f"({outside} session(s) of {asset.ticker} or {bench_symbol} history outside that "
                             "range are not used).")
            if len(inside):
                notes.append(
                    f"{len(inside)} session(s) inside the common range were present in only one of "
                    f"{asset.ticker} / {bench_symbol} and were dropped from both, so that strategy and benchmark "
                    "use identical dates."
                )
            if bench_ds.adjustment != asset.adjustment:
                notes.append(
                    f"Price conventions differ: {asset.ticker} is "
                    f"'{ADJUSTMENT_LABELS.get(asset.adjustment, asset.adjustment)}' but {bench_symbol} is "
                    f"'{ADJUSTMENT_LABELS.get(bench_ds.adjustment, bench_ds.adjustment)}'. Relative performance is "
                    "not like-for-like (dividends are counted for one series only)."
                )

    eval_mask = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    eval_index = prices.index[eval_mask]
    if asset.is_usable and len(eval_index) < 2:
        raise InputValidationError("No observations fall inside the selected evaluation period.")
    return ResearchData(
        asset=asset,
        benchmark_dataset=bench_ds,
        benchmark_ticker=bench_symbol,
        prices=prices,
        bench_prices=bench_prices,
        dropped_dates=dropped,
        fingerprint=fingerprint_prices(prices),
        bench_fingerprint=fingerprint_prices(bench_prices),
        eval_start=eval_index[0] if len(eval_index) else pd.Timestamp(start),
        eval_end=eval_index[-1] if len(eval_index) else pd.Timestamp(end),
        requested_start=start,
        requested_end=end,
        warmup_days=int(warmup_days),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Cached backtests
# --------------------------------------------------------------------------- #


# The caches below are shared by every session in the process. Each strategy run
# holds the net result and its zero-cost twin (about 1 MB per 10 years of data),
# so the number of entries and their lifetime are bounded; the benchmark run is
# cached once rather than inside every strategy entry.


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _cached_bench(_bench_prices: pd.DataFrame, fingerprint: str, bench_fingerprint: str, config_json: str,
                  eval_start: str, eval_end: str) -> BacktestResult:
    config = BacktestConfig.model_validate_json(config_json)
    meta = {"data_fingerprint": fingerprint, "benchmark_fingerprint": bench_fingerprint}
    return run_buy_and_hold(_bench_prices, config, pd.Timestamp(eval_start), pd.Timestamp(eval_end), meta=meta)


@st.cache_data(show_spinner=False, max_entries=64, ttl=3600)
def _cached_run(
    _prices: pd.DataFrame,
    _bench: BacktestResult,
    fingerprint: str,
    bench_fingerprint: str,
    strategy_key: str,
    params_json: str,
    config_json: str,
    eval_start: str,
    eval_end: str,
) -> StrategyRun:
    config = BacktestConfig.model_validate_json(config_json)
    start, end = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    meta = {"data_fingerprint": fingerprint, "benchmark_fingerprint": bench_fingerprint}
    return run_strategy(
        _prices, create_strategy(strategy_key, json.loads(params_json)), config, start, end, benchmark=_bench, meta=meta
    )


def benchmark_for(data: ResearchData, config: BacktestConfig) -> BacktestResult:
    return _cached_bench(data.bench_prices, data.fingerprint, data.bench_fingerprint, config.model_dump_json(),
                         str(data.eval_start.date()), str(data.eval_end.date()))


def run_for(data: ResearchData, strategy_key: str, params: dict[str, Any], config: BacktestConfig,
            register: bool = True) -> tuple[StrategyRun, BacktestResult]:
    bench = benchmark_for(data, config)
    run = _cached_run(
        data.prices,
        bench,
        data.fingerprint,
        data.bench_fingerprint,
        strategy_key,
        json.dumps(params, sort_keys=True),
        config.model_dump_json(),
        str(data.eval_start.date()),
        str(data.eval_end.date()),
    )
    if register:
        register_trial(data, strategy_key, params, config, run)
    return run, bench


def active_run(data: ResearchData, spec: RunSpec) -> tuple[StrategyRun, BacktestResult]:
    return run_for(data, spec.strategy_key, spec.params, spec.config)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _cached_cost_sensitivity(_prices: pd.DataFrame, _bench_prices: pd.DataFrame, fingerprint: str,
                             bench_fingerprint: str, strategy_key: str, params_json: str, config_json: str,
                             eval_start: str, eval_end: str) -> pd.DataFrame:
    from quantlab.analytics import cost_sensitivity

    return cost_sensitivity(_prices, create_strategy(strategy_key, json.loads(params_json)),
                            BacktestConfig.model_validate_json(config_json), pd.Timestamp(eval_start),
                            pd.Timestamp(eval_end), benchmark_prices=_bench_prices)


def cost_sensitivity_for(data: ResearchData, spec: RunSpec) -> pd.DataFrame:
    return _cached_cost_sensitivity(data.prices, data.bench_prices, data.fingerprint, data.bench_fingerprint,
                                    spec.strategy_key, spec.params_json(), spec.config.model_dump_json(),
                                    str(data.eval_start.date()), str(data.eval_end.date()))


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def _cached_permutation(_prices: pd.DataFrame, _result: BacktestResult, fingerprint: str, strategy_key: str,
                        params_json: str, config_json: str, eval_start: str, eval_end: str, n_shifts: int,
                        min_shift: int):
    from quantlab.nulls import timing_permutation_test

    return timing_permutation_test(_prices, _result, _result.config, n_shifts=n_shifts, min_shift=min_shift)


def permutation_for(data: ResearchData, spec: RunSpec, run: StrategyRun, n_shifts: int = 200):
    """Timing permutation test of the active run (cached). The minimum shift exceeds the
    strategy's look-back where the window allows, so wrapped signals cannot see their own returns."""
    n = len(run.net.daily)
    min_shift = int(min(max(63, run.strategy.warmup_sessions + 1), max(63, (n - 2 * 40) // 2)))
    return _cached_permutation(data.prices, run.net, data.fingerprint, spec.strategy_key, spec.params_json(),
                               spec.config.model_dump_json(), str(data.eval_start.date()),
                               str(data.eval_end.date()), n_shifts, min_shift)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def _cached_ml_fit(_prices: pd.DataFrame, fingerprint: str, params_json: str, eval_end: str):
    model = create_strategy("ml_logistic", json.loads(params_json))
    return model.walk_forward_fit(_prices.loc[: pd.Timestamp(eval_end)])


def ml_fit_for(data: ResearchData, params: dict[str, Any]):
    """The ML strategy's walk-forward fit on the loaded history up to the evaluation end (cached)."""
    return _cached_ml_fit(data.prices, data.fingerprint, json.dumps(params, sort_keys=True), str(data.eval_end.date()))


@st.cache_data(show_spinner=False, max_entries=32)
def bootstrap_for(strategy_returns: pd.Series, benchmark_returns: pd.Series, n_boot: int, mean_block: int,
                  risk_free_rate: float) -> pd.DataFrame:
    from quantlab.robustness import bootstrap_metrics

    return bootstrap_metrics(strategy_returns, benchmark_returns, n_boot=n_boot, mean_block=mean_block,
                             risk_free_rate=risk_free_rate)


# --------------------------------------------------------------------------- #
# Trial registry (multiple-testing awareness)
# --------------------------------------------------------------------------- #


def trial_key(data: ResearchData, strategy_key: str, params: dict[str, Any], config: BacktestConfig) -> str:
    """Identity of one evaluated configuration on one sample.

    Parameters are canonicalised through the strategy's own validation, so the
    sidebar run and an identical sensitivity-grid point map to the same trial,
    and every parameter (not only the ones on the grid) is part of the identity.
    """
    canonical = STRATEGIES[strategy_key].validate_params(params)
    payload = f"{data.sample_key}|{strategy_key}|{json.dumps(canonical, sort_keys=True)}|{config.model_dump_json()}"
    return hashlib.sha1(payload.encode()).hexdigest()


def _returns_signature(returns: pd.Series) -> list[float]:
    """Summary of a return stream that is equal (to floating-point tolerance) only for identical streams."""
    x = returns.to_numpy(dtype="float64")
    weights = np.arange(1, len(x) + 1, dtype="float64")
    return [float(len(x)), float(x.sum()), float((x * x).sum()), float((weights * x).sum())]


def _same_returns(a: list[float] | None, b: list[float] | None) -> bool:
    return a is not None and b is not None and bool(np.allclose(a, b, rtol=1e-9, atol=1e-12))


def register_trial(data: ResearchData, strategy_key: str, params: dict[str, Any], config: BacktestConfig,
                   run: StrategyRun) -> None:
    """Count a configuration as a trial on this sample.

    A setting that does not change the return stream (e.g. capital with
    fractional shares) is the same trial statistically, so configurations are
    also de-duplicated on their daily returns.
    """
    trials: dict[str, dict[str, Any]] = st.session_state.setdefault("trials", {})
    key = trial_key(data, strategy_key, params, config)
    signature = _returns_signature(run.net.returns)
    if key not in trials and not any(_same_returns(t.get("returns_signature"), signature) for t in trials_for(data)):
        trials[key] = {
            "returns_signature": signature,
            "sample": data.sample_key,
            "label": run.net.strategy_label,
            "strategy_key": strategy_key,
            "sharpe": run.report.value("sharpe"),
            # Per-period excess-return Sharpe: the unit the deflated Sharpe ratio works in.
            "sharpe_pp": per_period_sharpe(run.net.returns, config.risk_free_rate, config.periods_per_year),
        }


def trials_for(data: ResearchData) -> list[dict[str, Any]]:
    return [t for t in st.session_state.get("trials", {}).values() if t["sample"] == data.sample_key]


def register_trials_bulk(data: ResearchData, strategy_key: str, config: BacktestConfig,
                         rows: list[dict[str, Any]]) -> int:
    """Record grid-search evaluations (sensitivity analysis) as trials too.

    Each row needs the full ``params`` it was run with, plus ``label`` and the
    annualised excess-return ``sharpe``, and optionally its daily ``returns``:
    with them, a grid point whose return stream matches an existing trial is not
    counted again, exactly as for :func:`register_trial`. Returns the number of
    new trials.
    """
    trials: dict[str, dict[str, Any]] = st.session_state.setdefault("trials", {})
    added = 0
    for row in rows:
        key = trial_key(data, strategy_key, row["params"], config)
        signature = _returns_signature(row["returns"]) if row.get("returns") is not None else None
        if key in trials or (signature is not None and any(
                _same_returns(t.get("returns_signature"), signature) for t in trials_for(data))):
            continue
        sharpe = row.get("sharpe")
        finite = sharpe is not None and math.isfinite(sharpe)
        trials[key] = {
            "returns_signature": signature,
            "sample": data.sample_key,
            "label": row["label"],
            "strategy_key": strategy_key,
            "sharpe": sharpe if finite else None,
            "sharpe_pp": sharpe / math.sqrt(config.periods_per_year) if finite else None,
        }
        added += 1
    return added


def register_walk_forward(data: ResearchData, signature: str, result: Any) -> None:
    """Record a walk-forward specification (windows, grid, objective, costs) as a trial.

    Re-running walk-forward with different settings and keeping the best
    out-of-sample result is itself a selection step, so these runs are counted
    separately from full-sample trials (their Sharpes cover different windows).
    """
    runs: dict[str, dict[str, Any]] = st.session_state.setdefault("wf_trials", {})
    key = hashlib.sha1(signature.encode()).hexdigest()
    returns_signature = _returns_signature(result.oos.returns)
    # A setting that leaves the out-of-sample returns unchanged (e.g. capital with fractional shares) is the
    # same specification statistically.
    if key not in runs and not any(_same_returns(t.get("returns_signature"), returns_signature)
                                   for t in walk_forward_trials_for(data)):
        cfg = result.oos.config
        runs[key] = {
            "returns_signature": returns_signature,
            "sample": data.sample_key,
            "sharpe_pp": per_period_sharpe(result.oos.returns, cfg.risk_free_rate, cfg.periods_per_year),
        }


def walk_forward_trials_for(data: ResearchData) -> list[dict[str, Any]]:
    return [t for t in st.session_state.get("wf_trials", {}).values() if t["sample"] == data.sample_key]


# --------------------------------------------------------------------------- #
# Experiment store
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def _file_store() -> ExperimentStore:
    return ExperimentStore.open_file(data_dir() / "quant_lab_research.db")


def get_store() -> ExperimentStore:
    if storage_mode() == "sqlite":
        return _file_store()
    if "session_store" not in st.session_state:
        st.session_state["session_store"] = ExperimentStore.open_memory()
    return st.session_state["session_store"]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def _param_widget(container, key: str, spec: ParamSpec):
    # persist_state keeps a strategy's custom values while another strategy is active.
    wkey = f"p_{key}_{spec.name}"
    if spec.kind == "choice":
        return container.selectbox(spec.label, spec.choices, index=spec.choices.index(spec.default),
                                   key=wkey, help=spec.help, persist_state="session")
    if spec.kind == "int":
        return container.number_input(spec.label, min_value=int(spec.min), max_value=int(spec.max),
                                      value=int(spec.default), step=int(spec.step or 1), key=wkey, help=spec.help,
                                      persist_state="session")
    return container.number_input(spec.label, min_value=float(spec.min), max_value=float(spec.max),
                                  value=float(spec.default), step=float(spec.step or 0.1), key=wkey,
                                  help=spec.help, format="%.2f", persist_state="session")


def strategy_params_from_state(key: str) -> dict[str, Any]:
    """Current widget values for a strategy (defaults if never touched)."""
    cls = STRATEGIES[key]
    values = {}
    for spec in cls.param_specs:
        values[spec.name] = st.session_state.get(f"p_{key}_{spec.name}", spec.default)
    return values


def _attempt_load(**kwargs) -> None:
    try:
        with st.spinner("Loading and validating market data…"):
            st.session_state["research_data"] = load_research_data(**kwargs)
        st.session_state.pop("data_error", None)
    except (DataProviderError, InputValidationError) as exc:
        message = exc.user_message if isinstance(exc, DataProviderError) else str(exc)
        detail = getattr(exc, "detail", None)
        st.session_state["data_error"] = {"message": message, "detail": detail, "kind": getattr(exc, "kind", "input")}
    st.session_state["load_attempted"] = True


def _follow_strategy() -> None:
    """Keep the Strategy Lab guide on the sidebar's strategy when that changes."""
    st.session_state["lab_guide_strategy"] = st.session_state["w_strategy"]


def render_sidebar() -> RunSpec | None:
    sb = st.sidebar
    sb.markdown(
        '<div class="qlp-brand">Quant Lab Pro<small>Systematic equity research</small></div>',
        unsafe_allow_html=True,
    )

    # ---- market data -----------------------------------------------------------
    sb.markdown("#### Market data")
    # Initialised through session state (not `default=`) because callbacks such as the demo button and
    # "Restore configuration" also set it, and a widget must not get its value from both.
    st.session_state.setdefault("w_source", SOURCE_YAHOO)
    source = sb.segmented_control(
        "Data source", [SOURCE_YAHOO, SOURCE_CSV, SOURCE_DEMO], key="w_source",
        help="Yahoo Finance downloads free end-of-day history (not live prices). Upload a CSV to "
             "work offline or with another vendor's data. Synthetic demo uses simulated prices with a known "
             "data-generating process — not market data.",
    ) or SOURCE_YAHOO
    with sb.form("data_form", border=False):
        uploaded = None
        demo_market, demo_seed = "random", 1
        if source == SOURCE_CSV:
            uploaded = st.file_uploader(
                "Daily OHLCV CSV", type=["csv"], key="w_csv",
                help="Columns: Date, Open, High, Low, Close, Volume (optional: Adj Close).",
            )
        if source == SOURCE_DEMO:
            st.caption("Simulated prices, **not market data**. Use them to see whether a strategy finds an "
                       "edge that is (or is not) really there.")
            demo_market = st.selectbox("Synthetic market", list(SYNTHETIC_MARKETS), key="w_demo_market",
                                       format_func=lambda k: SYNTHETIC_MARKETS[k].name)
            demo_seed = st.selectbox("Random seed", DEMO_SEEDS, key="w_demo_seed",
                                     help="Each seed is a different path from the same process. Comparing seeds "
                                          "shows how much of a result is luck.")
            ticker = demo_ticker(demo_market, demo_seed)
        else:
            ticker = st.text_input("Ticker", value=DEFAULT_TICKER, key="w_ticker", max_chars=15,
                                   help="US-listed stock or ETF symbol as used by Yahoo Finance, e.g. SPY, QQQ, "
                                        "AAPL, BRK-B.")
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", value=DEFAULT_START, min_value=date(1990, 1, 1),
                              max_value=date.today(), key="w_start", format="YYYY-MM-DD")
        end = c2.date_input("End", value=date.today(), min_value=date(1990, 1, 2),
                            max_value=date.today(), key="w_end", format="YYYY-MM-DD")
        if source == SOURCE_DEMO:
            bench_mode, bench_ticker = BENCH_SAME, DEFAULT_BENCHMARK
            st.caption("Benchmark: buy & hold of the same simulated series.")
        else:
            bench_mode = st.radio("Benchmark", [BENCH_SAME, BENCH_TICKER], key="w_bench_mode", horizontal=False,
                                  help="Every strategy is compared with buy-and-hold of the benchmark, run through "
                                       "the same engine, costs and sessions.")
            bench_ticker = st.text_input("Benchmark ticker", value=DEFAULT_BENCHMARK, key="w_bench_ticker",
                                         help="Used only when 'Another ticker' is selected.")
        with st.expander("Advanced data options"):
            adjustment = st.selectbox(
                "Price convention",
                [AdjustmentMode.TOTAL_RETURN.value, AdjustmentMode.PRICE_ONLY.value],
                format_func=lambda v: ADJUSTMENT_LABELS[AdjustmentMode(v)],
                key="w_adjustment",
                help="Total return scales open/high/low/close by Adj Close / Close so dividends are "
                     "reinvested. Price return ignores dividends.",
            )
            warmup_days = st.number_input(
                "Indicator warm-up history (calendar days)", min_value=0, max_value=1500,
                value=DEFAULT_WARMUP_DAYS, step=50, key="w_warmup",
                help="Extra history loaded before the start date so indicators are ready on day one of the "
                     "evaluation. It is never traded or scored.",
            )
            force = st.checkbox("Bypass cache and re-download", key="w_force")
        submitted = st.form_submit_button("Load data", type="primary", width="stretch")

    load_kwargs = dict(
        source=source,
        ticker=ticker,
        benchmark_mode=bench_mode,
        benchmark_ticker=bench_ticker,
        start=start,
        end=end,
        adjustment=adjustment,
        warmup_days=int(warmup_days),
        csv_content=uploaded.getvalue() if uploaded is not None else None,
        csv_name=uploaded.name if uploaded is not None else "upload.csv",
        force_refresh=bool(force),
        demo_market=demo_market,
        demo_seed=int(demo_seed),
    )
    loaded_now = st.session_state.get("pending_load", False) or submitted
    if st.session_state.pop("pending_load", False):
        _attempt_load(**load_kwargs)  # a saved experiment's configuration was restored
    elif submitted:
        _attempt_load(**load_kwargs)
    elif not st.session_state.get("load_attempted") and source == SOURCE_YAHOO:
        # First visit: load the default example automatically, without retries, so an
        # unreachable provider fails fast and the page can offer the synthetic demo instead.
        _attempt_load(**{**load_kwargs, "retries": 0})

    notice = st.session_state.pop("restore_notice", None)
    if notice:
        sb.info(notice, icon=":material/settings_backup_restore:")
    data: ResearchData | None = st.session_state.get("research_data")
    expected_fp = st.session_state.get("restore_fingerprint")
    if expected_fp and data is not None and loaded_now:
        if data.fingerprint != expected_fp:
            sb.warning(f"Data fingerprint {data.fingerprint[:16]} differs from the saved experiment's "
                       f"{expected_fp[:16]}: the prices are not identical (provider revision, a different file, "
                       "or a different date range).", icon=":material/difference:")
        st.session_state.pop("restore_fingerprint", None)
    if data is not None and data.is_usable:
        ds = data.asset
        sb.caption(
            f"**{ds.ticker}** · {data.eval_start:%Y-%m-%d} → {data.eval_end:%Y-%m-%d} · "
            f"{len(data.eval_prices):,} sessions · last obs {ds.last_date:%Y-%m-%d}"
        )

    # ---- strategy --------------------------------------------------------------
    sb.markdown("#### Strategy")
    keys = list(STRATEGIES)
    strategy_key = sb.selectbox(
        "Active strategy", keys, index=keys.index(DEFAULT_STRATEGY), key="w_strategy", on_change=_follow_strategy,
        format_func=lambda k: STRATEGIES[k].name,
        help="The strategy analysed on the Overview, Backtesting and Performance pages.",
    )
    cls = STRATEGIES[strategy_key]
    params = {spec.name: _param_widget(sb, strategy_key, spec) for spec in cls.param_specs}
    declared = False
    params_error = None
    try:
        params = cls.validate_params(params)
    except StrategyParameterError as exc:
        params_error = str(exc)
    if params_error:
        sb.error(params_error, icon=":material/error:")
    elif params != cls.defaults():
        declared = sb.checkbox(
            "These parameters were fixed before I viewed results for this sample",
            key=f"w_declared_{strategy_key}", persist_state="session",
            help="Honest disclosure. Parameters chosen after looking at results are in-sample and "
                 "overstate expected performance.",
        )
    if cls.param_specs:
        def _reset(k=strategy_key):
            for spec in STRATEGIES[k].param_specs:
                st.session_state[f"p_{k}_{spec.name}"] = spec.default
        sb.button("Reset to defaults", on_click=_reset, key="w_reset", type="tertiary")

    # ---- execution & costs -----------------------------------------------------
    with sb.expander("Capital & execution costs", expanded=False):
        capital = st.number_input("Initial capital ($)", min_value=100.0, max_value=1e9, value=10_000.0,
                                  step=1_000.0, key="w_capital", format="%.0f")
        fee_pct = st.number_input("Fee (% of traded notional)", min_value=0.0, max_value=2.0, value=0.10,
                                  step=0.01, key="w_fee", format="%.3f",
                                  help="Commission per execution, charged on actual traded value.")
        slip_pct = st.number_input("Slippage (% per execution)", min_value=0.0, max_value=2.0, value=0.05,
                                   step=0.01, key="w_slip", format="%.3f",
                                   help="Buys fill this much above the open, sells this much below.")
        rf_pct = st.number_input("Risk-free rate (% p.a., for ratios)", min_value=-5.0, max_value=20.0,
                                 value=0.0, step=0.25, key="w_rf", format="%.2f",
                                 help="Used in Sharpe and Sortino ratios.")
        cash_pct = st.number_input("Interest on idle cash (% p.a.)", min_value=0.0, max_value=20.0,
                                   value=0.0, step=0.25, key="w_cash_rate", format="%.2f",
                                   help="Uninvested cash accrues this rate each session. Strategies that "
                                        "spend long periods in cash are penalised unrealistically at 0%.")
        band_pct = st.number_input("Rebalance band (percentage points)", min_value=0.0, max_value=50.0,
                                   value=5.0, step=0.5, key="w_band", format="%.1f",
                                   help="For partial-position strategies: trade only when the held weight "
                                        "drifts this far from the target. Entries and full exits always trade.")
        fractional = st.toggle("Allow fractional shares", value=True, key="w_fractional",
                               help="Off: only whole shares are bought and leftover cash sits idle.")

    if params_error:
        return None
    config = BacktestConfig(
        initial_capital=float(capital),
        fee_rate=float(fee_pct) / 100,
        slippage_rate=float(slip_pct) / 100,
        risk_free_rate=float(rf_pct) / 100,
        cash_interest_rate=float(cash_pct) / 100,
        rebalance_threshold=float(band_pct) / 100,
        fractional_shares=bool(fractional),
    )
    return RunSpec(strategy_key=strategy_key, params=params, config=config, declared_preselected=declared)


def start_synthetic_demo(market_key: str = "random") -> None:
    """Widget callback: switch the data source to the synthetic demo and load it."""
    ss = st.session_state
    ss["w_source"] = SOURCE_DEMO
    ss["w_demo_market"] = market_key
    ss["pending_load"] = True


def restore_configuration(record) -> None:
    """Widget callback: load a saved experiment's inputs into the sidebar and reload data.

    Must run as an ``on_click`` callback, because widget values can only be set
    before the widgets are drawn in a script run. Every field is validated; a
    field that cannot be restored is listed in a notice rather than silently
    replaced by a default, and an experiment run on an uploaded CSV is never
    re-run on downloaded data that merely shares its ticker.
    """
    from quantlab.simulation import parse_demo_ticker

    ss = st.session_state
    cfg = record.config or {}
    skipped: list[str] = []
    source = record.data_source or ""
    if source.startswith("Synthetic"):
        parsed = parse_demo_ticker(record.ticker)
        if parsed is None:
            skipped.append(f"synthetic market '{record.ticker}'")
        else:
            ss["w_source"] = SOURCE_DEMO
            ss["w_demo_market"], ss["w_demo_seed"] = parsed
    elif source.startswith("Yahoo"):
        ss["w_source"] = SOURCE_YAHOO
        ss["w_ticker"] = record.ticker
    else:
        ss["w_source"] = SOURCE_CSV
        ss["w_ticker"] = record.ticker
    for key, value in (("w_start", record.start_date), ("w_end", record.end_date)):
        try:
            ss[key] = date.fromisoformat(str(value))
        except ValueError:
            skipped.append(f"date '{value}'")
    same = not record.benchmark or record.benchmark == record.ticker
    ss["w_bench_mode"] = BENCH_SAME if same else BENCH_TICKER
    if not same:
        ss["w_bench_ticker"] = record.benchmark
    if record.price_adjustment in {m.value for m in AdjustmentMode}:
        ss["w_adjustment"] = record.price_adjustment
    if "warmup_days" in cfg:
        ss["w_warmup"] = int(cfg["warmup_days"])
    if record.strategy_key in STRATEGIES:
        cls = STRATEGIES[record.strategy_key]
        ss["w_strategy"] = record.strategy_key
        known = {p.name for p in cls.param_specs}
        for name, value in record.params.items():
            if name in known:
                ss[f"p_{record.strategy_key}_{name}"] = value
            else:
                skipped.append(f"parameter '{name}'")
    else:
        skipped.append(f"strategy '{record.strategy_key}' (not available in this version)")
    for key, field_name, scale, default in (
        ("w_capital", "initial_capital", 1, 10_000.0), ("w_fee", "fee_rate", 100, 0.001),
        ("w_slip", "slippage_rate", 100, 0.0005), ("w_rf", "risk_free_rate", 100, 0.0),
        ("w_cash_rate", "cash_interest_rate", 100, 0.0), ("w_band", "rebalance_threshold", 100, 0.05),
    ):
        if field_name not in cfg:
            skipped.append(f"{field_name} (default {default * scale:g} used)")
        ss[key] = float(cfg.get(field_name, default)) * scale
    ss["w_fractional"] = bool(cfg.get("fractional_shares", True))
    notice = []
    if ss["w_source"] == SOURCE_CSV:
        notice.append(f"This experiment used an uploaded file. Upload the same file and press Load data; "
                      f"its data fingerprint should be {record.data_fingerprint[:16]}.")
    else:
        ss["pending_load"] = True
    if skipped:
        notice.append("Not restored: " + "; ".join(skipped) + ".")
    ss["restore_notice"] = " ".join(notice) or None
    ss["restore_fingerprint"] = record.data_fingerprint


def current_spec() -> RunSpec | None:
    return st.session_state.get("run_spec")


def current_data() -> ResearchData | None:
    return st.session_state.get("research_data")
