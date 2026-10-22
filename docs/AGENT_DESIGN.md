# Research agent — design and contracts

This extends Quant Lab Pro with an **AI quantitative research agent**: a deterministic, tested
research engine (data → validation → analysis → report) plus an optional language-model layer
that orchestrates it. The engine works without any API key; the model only chooses which tools
to call and writes the narrative, and **every number it may state comes from a tool result**.

The agent's role, rules and output format are in [`agent_role.md`](agent_role.md) (the system
prompt). This document fixes the module boundaries and the contracts between them.

## Principles (non-negotiable, enforced in code where possible)

1. **Research and simulation only.** There is no broker integration and no tool that can place,
   modify or cancel a real order, move funds or change account settings. Paper trading writes to
   a local simulated ledger only. This boundary is technical: the tool registry contains no
   execution tools, and a test asserts it.
2. **No fabricated data.** Providers raise on failure; nothing returns placeholder values. A
   missing value is `None` with a reason, never zero. Every fact carries a
   `quantlab.provenance.Provenance`.
3. **Risk before opportunity.** Position sizes come only from `quantlab.risk`; limits are checked
   in code and cannot be overridden by the model. No leverage, no short selling (the engine is
   long-only; a bearish thesis becomes "avoid / reduce", not a short).
4. **Data-quality gating.** If critical data fails validation, analysis that depends on it is
   withheld and the reason is shown.
5. **Untrusted input.** News, filings text and uploaded files are data, never instructions. They
   reach the model wrapped in `<untrusted_data>` and the system prompt says so.
6. **No secrets in code, logs or records.** The API key comes from the environment
   (`ANTHROPIC_API_KEY`) or Streamlit secrets; logs are redacted.
7. **Honest uncertainty.** Confidence is an evidence-quality rating (Low / Moderate / High) from a
   documented rubric, never a probability of profit.

## Package layout

```
quantlab/
  provenance.py      Provenance, Fact, Calculated, Timeliness, Adjustment   (done)
  fundamentals.py    statements, profile, events, news, filings; Yahoo + SEC EDGAR providers;
                     fundamental metrics with formulas                       (agent A)
  technicals.py      MACD, ATR, relative strength, volume anomalies, 52-week range,
                     support/resistance heuristics; TechnicalSnapshot         (agent B)
  signals.py         SignalSpec (hypothesis … limitations) for every strategy; current state (agent B)
  scan.py            SCAN mode: documented criteria over a user-defined universe (agent B)
  alerts.py          ALERTS mode: validated conditions on the latest data     (agent B)
  risk.py            RiskLimits, position sizing, gap risk, risk/reward, limit checks (agent C)
  holdings.py        PORTFOLIO REVIEW: concentration, sector, correlation, VaR, scenarios (agent C)
  paper.py           PAPER TRADE: simulated orders against daily bars, SQLite ledger (agent D)
  postmortem.py      POST-TRADE REVIEW: thesis vs outcome, systematic errors (agent D)
  research_report.py RESEARCH mode: the 10-section report, confidence rubric  (integration)
  agent/             LLM layer: tools, guardrails, runner, audit log, prompts (integration)
pages/9_Research_Agent.py                                                    (integration)
```

All `quantlab` modules stay free of Streamlit. Network access goes through injectable
callables (`ticker_factory`, `http_get`) so tests run offline with fakes; Yahoo and SEC are
blocked in the build sandbox, so **no test may touch the network**.

## Shared conventions

* Time: UTC `datetime` (tz-aware) for retrieval/publication times; exchange-local `date` for
  sessions and fiscal periods.
* Money: float in the instrument's currency; the currency code travels in `Provenance.currency`.
* A function that cannot compute a value returns `None` (or a `Calculated` with `value=None` and
  a `reason`) — never 0, never NaN in a user-facing result.
* Errors: data-source failures raise `quantlab.data.DataProviderError` subclasses
  (`TickerNotFoundError`, `RateLimitError`, `ProviderConnectionError`, `MalformedDataError`).
* Every public result type has `to_dict()` returning JSON-serialisable data (used by the agent's
  tools and the audit log). Use `quantlab.serialization.clean_json` if helpful.
* Tests: synthetic data only (`tests/helpers.py` `random_walk`, `make_prices`;
  `quantlab.simulation`), deterministic seeds, each behaviour pinned by a test with a docstring.

## Contracts

### A. `quantlab/fundamentals.py`

```python
class StatementPeriod(str, Enum): ANNUAL = "annual"; QUARTERLY = "quarterly"

@dataclass(frozen=True)
class FinancialStatements:
    ticker: str
    currency: str | None
    income: pd.DataFrame      # rows: canonical line items (below); columns: fiscal period end (Timestamp), newest first
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    period: StatementPeriod
    provenance: Provenance    # timeliness PERIODIC_FILING
    filed: dict[pd.Timestamp, datetime]   # period end -> filing time, when known (EDGAR); empty for Yahoo

# Canonical line items (index labels), absent rows simply missing:
INCOME_ITEMS = ("revenue", "gross_profit", "operating_income", "net_income", "ebitda",
                "depreciation_amortization", "eps_diluted", "diluted_shares")
BALANCE_ITEMS = ("total_assets", "total_liabilities", "total_equity", "cash", "total_debt",
                 "current_assets", "current_liabilities", "shares_outstanding")
CASHFLOW_ITEMS = ("operating_cash_flow", "capital_expenditure", "free_cash_flow", "dividends_paid")

@dataclass(frozen=True)
class CompanyProfile:
    ticker: str; name: str | None; exchange: str | None; sector: str | None; industry: str | None
    currency: str | None; country: str | None; shares_outstanding: float | None; provenance: Provenance

@dataclass(frozen=True)
class CorporateEvent:      # earnings dates, dividends, splits
    ticker: str; kind: str; date: date; confirmed: bool | None; detail: str; provenance: Provenance

@dataclass(frozen=True)
class NewsItem:            # UNTRUSTED text
    ticker: str; title: str; publisher: str | None; url: str | None; published_at: datetime | None
    provenance: Provenance

@dataclass(frozen=True)
class Filing:              # SEC filings index
    ticker: str; cik: str; form: str; filed: date; period: date | None; url: str; provenance: Provenance

@dataclass(frozen=True)
class InsiderTransaction:
    ticker: str; insider: str | None; relation: str | None; date: date | None; transaction: str | None
    shares: float | None; value: float | None; provenance: Provenance

class FundamentalsProvider(ABC):
    key: str; name: str
    def profile(self, ticker) -> CompanyProfile
    def statements(self, ticker, period=StatementPeriod.ANNUAL) -> FinancialStatements
    def events(self, ticker) -> list[CorporateEvent]          # may return []
    def news(self, ticker, limit=20) -> list[NewsItem]       # may return []
    def filings(self, ticker, forms=("10-K","10-Q","8-K"), limit=20) -> list[Filing]   # may raise NotSupported
    def insider_transactions(self, ticker, limit=50) -> list[InsiderTransaction]       # may return []

class YahooFundamentalsProvider(FundamentalsProvider)   # yfinance Ticker(...).income_stmt, quarterly_*, info,
                                                         # calendar/get_earnings_dates, news, insider_transactions
class SecEdgarProvider(FundamentalsProvider)            # data.sec.gov companyfacts + submissions; ticker->CIK via
    # https://www.sec.gov/files/company_tickers.json. Requires a User-Agent "Name email" from env
    # QLP_SEC_USER_AGENT (never hard-coded); without it raise ProviderConnectionError with a clear message.
    # Throttle to <= 5 requests/s; cache JSON responses on disk (DiskCache-style dir under QLP_DATA_DIR).
    # Uses the `filed` date of each XBRL fact -> point-in-time `filed` map; `statements(..., as_of=None)`
    # optional kwarg excludes facts filed after as_of.

def fundamental_metrics(statements: FinancialStatements, price: Fact, shares: Fact | None = None,
                        ttm: FinancialStatements | None = None) -> dict[str, Calculated]
# keys (each a Calculated with a formula string and inputs; None + reason when not computable):
#   revenue_growth_yoy, earnings_growth_yoy, gross_margin, operating_margin, net_margin,
#   free_cash_flow, fcf_margin, market_cap, enterprise_value, pe_trailing, price_to_sales,
#   ev_to_ebitda, fcf_yield, net_debt_to_ebitda, debt_to_equity, current_ratio, cash_conversion
# Ratios with a non-positive denominator (e.g. P/E with negative earnings) are None with a reason.
def statements_quality(statements) -> list[str]   # warnings: missing items, stale periods, sign anomalies
```

### B. `quantlab/technicals.py`, `signals.py`, `scan.py`, `alerts.py`

```python
# technicals.py (pure functions on validated daily OHLCV; causal; NaN warm-up, never back-filled)
def macd(close, fast=12, slow=26, signal=9) -> pd.DataFrame          # macd, signal, histogram (uses indicators.ema)
def atr(prices, window=14) -> pd.Series                               # Wilder true range
def relative_strength(close, bench_close, lookbacks=(63,126,252)) -> dict[int, float | None]  # asset minus bench return
def volume_zscore(volume, window=50) -> pd.Series                     # on log volume; NaN where volume missing/zero
def range_52w(prices) -> dict                                         # high, low, pct_from_high, pct_from_low
def support_resistance(prices, windows=(20, 60)) -> dict              # rolling min low / max high — labelled heuristic
def overnight_gaps(prices) -> pd.Series                               # open_t / close_{t-1} - 1

@dataclass(frozen=True)
class TechnicalSnapshot:
    ticker: str; as_of: date; close: Fact; values: dict[str, Calculated]; trend: str; notes: tuple[str, ...]
    def to_dict(self) -> dict
def technical_snapshot(ticker, prices, provenance, bench_prices=None) -> TechnicalSnapshot
# values: sma_50, sma_200, pct_vs_sma_50, pct_vs_sma_200, rsi_14, macd, macd_signal, macd_hist, atr_14,
#   atr_pct, realised_vol_63, rs_63/126/252, volume_z, high_52w, low_52w, pct_from_high_52w, support_20, resistance_20 …
# trend: "uptrend" | "downtrend" | "mixed" by a documented rule (close vs SMA50 vs SMA200).

# signals.py
@dataclass(frozen=True)
class SignalSpec:
    key: str; name: str; hypothesis: str; inputs: tuple[str, ...]; calculation: str; entry: str; exit: str
    holding_period: str; invalidation: str; benchmark: str; limitations: tuple[str, ...]
def signal_spec(strategy_key, params=None) -> SignalSpec       # for every strategy in quantlab.strategies.STRATEGIES
def current_signal(strategy, prices) -> dict                     # latest target, date it last changed, condition values,
                                                                 # whether the last signal is still unexecuted

# scan.py — SCAN mode
@dataclass(frozen=True)
class ScanCriteria:   # every criterion optional; documented in describe()
    min_price: float | None = 5.0; min_avg_dollar_volume: float | None = 5e6; min_history_sessions: int = 252
    trend: str | None = None            # "uptrend" | "downtrend" | None
    rsi_below: float | None = None; rsi_above: float | None = None
    min_rs_126: float | None = None     # relative strength vs benchmark over 126 sessions
    max_atr_pct: float | None = None
    def describe(self) -> list[str]
@dataclass(frozen=True)
class ScanResult: rows: pd.DataFrame; criteria: ScanCriteria; failures: dict[str, str]; notes: tuple[str, ...]
def scan_universe(universe: list[str], load_prices: Callable[[str], MarketDataset], criteria, bench_prices=None) -> ScanResult
# rows: one per ticker that loaded — each criterion value, pass/fail per criterion, overall pass, data-quality flag.
# Tickers that fail to load or fail validation go to `failures` with the reason (never silently dropped).
# notes always include the survivorship-bias caveat (the universe is user-supplied, today's constituents).

# alerts.py — ALERTS mode (evaluated on demand on the latest validated end-of-day data; no notifications sent)
@dataclass(frozen=True)
class AlertRule:  ticker: str; kind: str; params: dict      # kinds: price_above, price_below, rsi_below, rsi_above,
                                                             # sma_cross_up, sma_cross_down, volume_z_above,
                                                             # new_52w_high, new_52w_low, signal_change
@dataclass(frozen=True)
class AlertEvent: rule: AlertRule; triggered: bool; as_of: date; value: float | None; detail: str; data_status: str
def evaluate_alerts(rules, load_prices) -> list[AlertEvent]  # one event per rule; data failures -> triggered False + status
```

### C. `quantlab/risk.py`, `holdings.py`

```python
class RiskLimits(BaseModel):            # user-configured; validated
    account_equity: float               # > 0
    available_cash: float | None = None # defaults to equity
    max_risk_per_trade_pct: float = 0.005      # of equity
    max_daily_loss_pct: float = 0.02
    max_position_pct: float = 0.10             # notional / equity
    max_gross_exposure_pct: float = 1.0        # <= 1.0 always: no leverage (validator rejects > 1)
    max_sector_pct: float = 0.30
    max_drawdown_pct: float = 0.15             # new risk halted beyond this drawdown
    illustrative: bool = False                 # True when the user did not supply limits — labelled in output

@dataclass(frozen=True)
class TradeSetup:
    ticker: str; entry: float; stop: float; targets: tuple[float, ...] = ()
    side: str = "long"                  # only "long" is supported; anything else -> ValueError
    fee_rate: float = 0.001; slippage_rate: float = 0.0005; lot_size: float = 1.0
    gap_allowance: float = 0.0          # extra adverse move per unit beyond the stop (e.g. from gap_risk())
    fx_rate_to_account: float = 1.0     # account currency per instrument currency
    sector: str | None = None

@dataclass(frozen=True)
class SizingResult:
    units: float; notional: float; risk_budget: float; loss_per_unit_at_stop: float
    est_loss_at_stop: float; est_loss_with_gap: float; binding_constraint: str
    formula: str; warnings: tuple[str, ...]; illustrative: bool
    def to_dict(self) -> dict
def size_position(setup: TradeSetup, limits: RiskLimits, portfolio: PortfolioState | None = None) -> SizingResult
# units = floor(risk_budget / loss_per_unit / lot) * lot, where
#   loss_per_unit = entry*(1+slip) - stop*(1-slip) + gap_allowance + fees(entry and exit) ; risk_budget = equity * max_risk_per_trade_pct
# then capped by max_position_pct, available cash, gross exposure, sector limit; binding_constraint names the cap.
# A stop at or above the entry, a non-positive price, or zero units -> SizingResult with units 0 and the reason.

def gap_risk(prices, stop_distance_pct) -> dict
# historical overnight-gap distribution: P(adverse gap > stop distance), worst gap, 99th-pct adverse gap,
# suggested gap_allowance; based on the asset's own history (state the sample size).

def risk_reward(setup: TradeSetup) -> dict    # per target: gross R multiple and net R multiple after costs

@dataclass
class Position: ticker: str; units: float; entry_price: float; last_price: float; sector: str | None = None
@dataclass
class PortfolioState:
    equity: float; cash: float; positions: list[Position]; peak_equity: float | None = None
    realised_pnl_today: float = 0.0
    def gross_exposure(self) -> float; def drawdown(self) -> float
@dataclass(frozen=True)
class LimitCheck: name: str; ok: bool; value: float | None; limit: float | None; detail: str
def check_limits(setup, sizing, limits, portfolio, correlations: pd.DataFrame | None = None) -> list[LimitCheck]
# per-trade risk, daily loss (blocks new risk when breached), position %, gross exposure, sector %,
# drawdown halt, correlated exposure (|corr| >= 0.7 with an existing position -> combined exposure shown)
# A single failing check means "do not trade"; the model cannot override it.

# holdings.py — PORTFOLIO REVIEW
@dataclass(frozen=True)
class Holding: ticker: str; quantity: float; cost_basis: float | None = None; sector: str | None = None
def parse_holdings_csv(content: bytes) -> list[Holding]               # columns ticker, quantity[, cost_basis, sector]
@dataclass(frozen=True)
class PortfolioReview:
    as_of: date; holdings: pd.DataFrame; concentration: dict; sector_exposure: pd.DataFrame
    correlation: pd.DataFrame; risk: dict; scenarios: pd.DataFrame; liquidity: pd.DataFrame
    warnings: tuple[str, ...]; data_failures: dict[str, str]
    def to_dict(self) -> dict
def review_portfolio(holdings, load_prices, bench_prices=None, lookback=252) -> PortfolioReview
# weights from latest validated closes; HHI and top-N weight; volatility, historical VaR/CVaR 95% (quantlab.inference),
# max drawdown of today's weights over the lookback (labelled hypothetical), beta to the benchmark,
# scenarios: benchmark -10% / -20% via betas, worst historical month for current weights;
# liquidity: position value / 20-day average dollar volume. Holdings that fail to load are listed, never dropped.
```

### D. `quantlab/paper.py`, `postmortem.py`

```python
@dataclass(frozen=True)
class Thesis:       # recorded with every paper order (for post-trade review)
    direction: str; summary: str; entry_rationale: str; stop: float; targets: tuple[float, ...]
    invalidation: str; expected_holding_days: int; confidence: str      # "low" | "moderate" | "high"

class PaperBook:    # SIMULATED ONLY — never connects to a broker
    def __init__(self, store: ExperimentStore, account: str = "default", starting_cash: float = 100_000.0)
    def submit(self, ticker, side, quantity, order_type="market", limit=None, stop=None, thesis=None,
               client_order_id: str | None = None, risk_checks: list[LimitCheck] | None = None) -> PaperOrder
    # duplicate client_order_id -> returns the existing order (idempotent); failed risk checks -> REJECTED with reasons;
    # long-only: a sell larger than the position is rejected; no margin: buys beyond cash are rejected.
    def process(self, ticker, prices: pd.DataFrame) -> list[PaperFill]
    # fills open orders against daily bars AFTER the order's creation date: market -> next open; limit buy -> when
    # low <= limit (at min(open, limit)); stop sell -> when low <= stop (at min(open, stop): gaps fill through);
    # protective stops and targets from the thesis are attached as orders when a buy fills.
    def positions(self) -> pd.DataFrame; def orders(self) -> pd.DataFrame; def fills(self) -> pd.DataFrame
    def equity_curve(self, prices_by_ticker) -> pd.Series; def cancel(self, order_id) -> None
    # Persisted via a new ExperimentStore migration (paper_orders, paper_fills tables); every row is labelled simulated.

# postmortem.py — POST-TRADE REVIEW
@dataclass(frozen=True)
class TradeReview:
    order_id: int; ticker: str; entry_date: date; exit_date: date | None; entry: float; exit: float | None
    realised_r: float | None; mae_r: float | None; mfe_r: float | None; holding_days: int | None
    expected_holding_days: int | None; outcome: str   # "target" | "stop" | "manual" | "open" | "invalidated"
    slippage_vs_assumed: float | None; notes: tuple[str, ...]
def review_trade(book: PaperBook, order_id, prices) -> TradeReview
def review_all(book, prices_by_ticker) -> tuple[list[TradeReview], dict]
# summary: hit rate, average R, expectancy, profit factor, R by confidence level (calibration),
# systematic patterns (stops hit then recovered within N days, holding periods vs expected, slippage bias),
# sample-size warnings when fewer than 20 closed trades.
```

### Integration (research report, agent, UI)

* `research_report.py` builds the **10-section report** from the outputs above, labelling each
  line fact / calculated / interpretation, gating on data quality, and rating confidence with a
  documented rubric. A "no qualifying setup" conclusion is a first-class outcome.
* `agent/` exposes the engine as tools with strict JSON schemas; runs a manual tool loop on the
  Claude API (`claude-opus-5-5`, adaptive thinking, server-side refusal fallback); records an audit
  log; and checks the model's output: every number must appear in a tool result or the user's
  input, every URL must come from a tool result, required sections must be present, and no
  guarantee language is allowed. Failing checks are shown prominently; in strict mode the report
  is withheld.
* `pages/9_Research_Agent.py` shows the active mode and its required inputs before running.
