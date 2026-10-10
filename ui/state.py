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
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from quantlab.analytics import StrategyRun, run_strategy
from quantlab.backtest import BacktestResult, run_buy_and_hold
from quantlab.config import BacktestConfig, data_dir, storage_mode
from quantlab.data import (
    ADJUSTMENT_LABELS,
    AdjustmentMode,
    CSVProvider,
    DataProviderError,
    DiskCache,
    MarketDataset,
    YahooFinanceProvider,
    align_benchmark,
    fingerprint_prices,
    load_market_data,
)
from quantlab.experiments import ExperimentStore
from quantlab.research import per_period_sharpe
from quantlab.strategies import STRATEGIES, ParamSpec, StrategyParameterError, create_strategy
from quantlab.validation import InputValidationError

DEFAULT_TICKER = "SPY"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_START = date(2015, 1, 2)
DEFAULT_WARMUP_DAYS = 400
DEFAULT_STRATEGY = "ma_crossover"

SOURCE_YAHOO = "Yahoo Finance"
SOURCE_CSV = "CSV file"
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

    @property
    def sample_key(self) -> str:
        return f"{self.fingerprint[:16]}|{self.bench_fingerprint[:16]}|{self.eval_start.date()}|{self.eval_end.date()}"


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
def _fetch_yahoo(ticker: str, start: str, end: str, adjustment: str, refresh_token: int) -> MarketDataset:
    return load_market_data(
        ticker,
        start,
        end,
        YahooFinanceProvider(),
        adjustment=AdjustmentMode(adjustment),
        cache=DiskCache(data_dir() / "cache"),
        force_refresh=refresh_token > 0,
    )


def _load_csv(content: bytes, filename: str, ticker: str, start: str, end: str, adjustment: str) -> MarketDataset:
    return load_market_data(ticker, start, end, CSVProvider(content, filename), adjustment=AdjustmentMode(adjustment))


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
    refresh_token: int = 0,
) -> ResearchData:
    """Fetch, validate and align asset and benchmark prices. Raises on failure."""
    if end <= start:
        raise InputValidationError("End date must be after the start date.")
    fetch_start = (pd.Timestamp(start) - timedelta(days=int(warmup_days))).date()
    if source == SOURCE_CSV:
        if not csv_content:
            raise InputValidationError("Choose a CSV file to upload first.")
        asset = _load_csv(csv_content, csv_name, ticker, str(fetch_start), str(end), adjustment)
    else:
        asset = _fetch_yahoo(ticker.strip().upper(), str(fetch_start), str(end), adjustment, refresh_token)

    notes: list[str] = []
    bench_ds: MarketDataset | None = None
    prices = asset.prices
    bench_prices = prices
    dropped = pd.DatetimeIndex([])
    bench_symbol = asset.ticker
    if benchmark_mode == BENCH_TICKER and benchmark_ticker.strip().upper() != asset.ticker:
        bench_symbol = benchmark_ticker.strip().upper()
        bench_ds = _fetch_yahoo(bench_symbol, str(fetch_start), str(end), adjustment, refresh_token)
        if bench_ds.is_usable and asset.is_usable:
            prices, bench_prices, dropped = align_benchmark(asset.prices, bench_ds.prices)
            if len(dropped):
                notes.append(
                    f"{len(dropped)} session(s) present in only one of {asset.ticker} / {bench_symbol} were "
                    "dropped from both so that strategy and benchmark use identical dates."
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


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_run(
    _prices: pd.DataFrame,
    _bench_prices: pd.DataFrame,
    fingerprint: str,
    bench_fingerprint: str,
    strategy_key: str,
    params_json: str,
    config_json: str,
    eval_start: str,
    eval_end: str,
) -> tuple[StrategyRun, BacktestResult]:
    config = BacktestConfig.model_validate_json(config_json)
    start, end = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    meta = {"data_fingerprint": fingerprint, "benchmark_fingerprint": bench_fingerprint}
    bench = run_buy_and_hold(_bench_prices, config, start, end, meta=meta)
    run = run_strategy(
        _prices, create_strategy(strategy_key, json.loads(params_json)), config, start, end, benchmark=bench, meta=meta
    )
    return run, bench


def run_for(data: ResearchData, strategy_key: str, params: dict[str, Any], config: BacktestConfig,
            register: bool = True) -> tuple[StrategyRun, BacktestResult]:
    run, bench = _cached_run(
        data.prices,
        data.bench_prices,
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


# --------------------------------------------------------------------------- #
# Trial registry (multiple-testing awareness)
# --------------------------------------------------------------------------- #


def register_trial(data: ResearchData, strategy_key: str, params: dict[str, Any], config: BacktestConfig,
                   run: StrategyRun) -> None:
    trials: dict[str, dict[str, Any]] = st.session_state.setdefault("trials", {})
    key = hashlib.sha1(
        f"{data.sample_key}|{strategy_key}|{json.dumps(params, sort_keys=True)}|{config.model_dump_json()}".encode()
    ).hexdigest()
    if key not in trials:
        trials[key] = {
            "sample": data.sample_key,
            "label": run.net.strategy_label,
            "strategy_key": strategy_key,
            "sharpe": run.report.value("sharpe"),
            "sharpe_pp": per_period_sharpe(run.net.returns),
        }


def trials_for(data: ResearchData) -> list[dict[str, Any]]:
    return [t for t in st.session_state.get("trials", {}).values() if t["sample"] == data.sample_key]


def register_trials_bulk(data: ResearchData, rows: list[dict[str, Any]]) -> None:
    """Record grid-search evaluations (sensitivity analysis) as trials too."""
    trials: dict[str, dict[str, Any]] = st.session_state.setdefault("trials", {})
    for row in rows:
        key = hashlib.sha1(f"{data.sample_key}|{row['id']}".encode()).hexdigest()
        trials.setdefault(key, {"sample": data.sample_key, **row})


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
    wkey = f"p_{key}_{spec.name}"
    if spec.kind == "choice":
        return container.selectbox(spec.label, spec.choices, index=spec.choices.index(spec.default),
                                   key=wkey, help=spec.help)
    if spec.kind == "int":
        return container.number_input(spec.label, min_value=int(spec.min), max_value=int(spec.max),
                                      value=int(spec.default), step=int(spec.step or 1), key=wkey, help=spec.help)
    return container.number_input(spec.label, min_value=float(spec.min), max_value=float(spec.max),
                                  value=float(spec.default), step=float(spec.step or 0.1), key=wkey,
                                  help=spec.help, format="%.2f")


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


def render_sidebar() -> RunSpec | None:
    sb = st.sidebar
    sb.markdown(
        '<div class="qlp-brand">Quant Lab Pro<small>Systematic equity research</small></div>',
        unsafe_allow_html=True,
    )

    # ---- market data -----------------------------------------------------------
    sb.markdown("#### Market data")
    source = sb.segmented_control(
        "Data source", [SOURCE_YAHOO, SOURCE_CSV], default=SOURCE_YAHOO, key="w_source",
        help="Yahoo Finance downloads free end-of-day history (not live prices). Upload a CSV to "
             "work offline or with another vendor's data.",
    ) or SOURCE_YAHOO
    with sb.form("data_form", border=False):
        uploaded = None
        if source == SOURCE_CSV:
            uploaded = st.file_uploader(
                "Daily OHLCV CSV", type=["csv"], key="w_csv",
                help="Columns: Date, Open, High, Low, Close, Volume (optional: Adj Close).",
            )
        ticker = st.text_input("Ticker", value=DEFAULT_TICKER, key="w_ticker", max_chars=15,
                               help="US-listed stock or ETF symbol as used by Yahoo Finance, e.g. SPY, QQQ, AAPL, BRK-B.")
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", value=DEFAULT_START, min_value=date(1990, 1, 1),
                              max_value=date.today(), key="w_start", format="YYYY-MM-DD")
        end = c2.date_input("End", value=date.today(), min_value=date(1990, 1, 2),
                            max_value=date.today(), key="w_end", format="YYYY-MM-DD")
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
        refresh_token=st.session_state.get("refresh_counter", 0) + 1 if force else 0,
    )
    if st.session_state.pop("pending_load", False):
        _attempt_load(**load_kwargs)  # a saved experiment's configuration was restored
    elif submitted:
        if force:
            st.session_state["refresh_counter"] = st.session_state.get("refresh_counter", 0) + 1
        _attempt_load(**load_kwargs)
    elif not st.session_state.get("load_attempted") and source == SOURCE_YAHOO:
        _attempt_load(**load_kwargs)  # first visit: load the default example automatically

    data: ResearchData | None = st.session_state.get("research_data")
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
        "Active strategy", keys, index=keys.index(DEFAULT_STRATEGY), key="w_strategy",
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
            key=f"w_declared_{strategy_key}",
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
        rf_pct = st.number_input("Risk-free rate (% p.a., ratios only)", min_value=-5.0, max_value=20.0,
                                 value=0.0, step=0.25, key="w_rf", format="%.2f",
                                 help="Used in Sharpe/Sortino. Cash in the portfolio still earns 0%.")
        fractional = st.toggle("Allow fractional shares", value=True, key="w_fractional",
                               help="Off: only whole shares are bought and leftover cash sits idle.")

    if params_error:
        return None
    config = BacktestConfig(
        initial_capital=float(capital),
        fee_rate=float(fee_pct) / 100,
        slippage_rate=float(slip_pct) / 100,
        risk_free_rate=float(rf_pct) / 100,
        fractional_shares=bool(fractional),
    )
    return RunSpec(strategy_key=strategy_key, params=params, config=config, declared_preselected=declared)


def restore_configuration(record) -> None:
    """Widget callback: load a saved experiment's inputs into the sidebar and reload data.

    Must run as an ``on_click`` callback, because widget values can only be set
    before the widgets are drawn in a script run.
    """
    ss = st.session_state
    cfg = record.config
    if record.data_source.startswith("Yahoo"):
        ss["w_source"] = SOURCE_YAHOO
    ss["w_ticker"] = record.ticker
    ss["w_start"] = date.fromisoformat(record.start_date)
    ss["w_end"] = date.fromisoformat(record.end_date)
    same = not record.benchmark or record.benchmark == record.ticker
    ss["w_bench_mode"] = BENCH_SAME if same else BENCH_TICKER
    if not same:
        ss["w_bench_ticker"] = record.benchmark
    if record.price_adjustment in {m.value for m in AdjustmentMode}:
        ss["w_adjustment"] = record.price_adjustment
    if "warmup_days" in cfg:
        ss["w_warmup"] = int(cfg["warmup_days"])
    ss["w_strategy"] = record.strategy_key
    for name, value in record.params.items():
        ss[f"p_{record.strategy_key}_{name}"] = value
    ss["w_capital"] = float(cfg.get("initial_capital", 10_000.0))
    ss["w_fee"] = float(cfg.get("fee_rate", 0.001)) * 100
    ss["w_slip"] = float(cfg.get("slippage_rate", 0.0005)) * 100
    ss["w_rf"] = float(cfg.get("risk_free_rate", 0.0)) * 100
    ss["w_fractional"] = bool(cfg.get("fractional_shares", True))
    ss["pending_load"] = True


def current_spec() -> RunSpec | None:
    return st.session_state.get("run_spec")


def current_data() -> ResearchData | None:
    return st.session_state.get("research_data")
