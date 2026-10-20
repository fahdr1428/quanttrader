"""State for the Research Agent page: data loaders, risk limits, holdings, the paper account and the API key.

The API key is read from the environment (``ANTHROPIC_API_KEY``) or Streamlit secrets and is
never displayed, logged or stored. Risk limits are the user's own; until the user confirms
them, the illustrative defaults are used and labelled as such everywhere.
"""

from __future__ import annotations

import hmac
import os
from datetime import date, timedelta
from typing import Any, Callable

import streamlit as st

from quantlab.agent.tools import ToolContext
from quantlab.data import DataProviderError, MarketDataset
from quantlab.fundamentals import SEC_USER_AGENT_ENV, SecEdgarProvider, YahooFundamentalsProvider
from quantlab.holdings import Holding
from quantlab.paper import PaperBook
from quantlab.risk import RiskLimits, illustrative_limits
from quantlab.simulation import SIM_END, SIM_START, parse_demo_ticker
from ui.state import _load_synthetic, _yahoo, current_data, get_store

HISTORY_YEARS = 6


def load_prices(ticker: str) -> MarketDataset:
    """Validated daily prices for the agent: the dataset loaded in the sidebar when it matches, a labelled
    synthetic series for demo tickers, otherwise about six years of Yahoo Finance end-of-day data."""
    symbol = ticker.strip().upper()
    loaded = current_data()
    if loaded is not None and loaded.asset.ticker.upper() == symbol:
        return loaded.asset
    if parse_demo_ticker(symbol):
        return _load_synthetic(symbol, SIM_START.strftime("%Y-%m-%d"), SIM_END.strftime("%Y-%m-%d"))
    end = date.today()
    start = end - timedelta(days=365 * HISTORY_YEARS)
    return _yahoo(symbol, start.isoformat(), end.isoformat(), "total_return")


def _secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        found = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - having no secrets file is normal
        return None
    return str(found) if found else None


def api_key() -> str | None:
    """The Anthropic API key, or None. Callers pass it straight to the client; it is never shown."""
    return _secret("ANTHROPIC_API_KEY")


def agent_passcode() -> str | None:
    """Optional passcode (``QLP_AGENT_PASSCODE``) that visitors must enter before using the API key."""
    return _secret("QLP_AGENT_PASSCODE")


def passcode_ok(entered: str) -> bool:
    expected = agent_passcode()
    return expected is None or hmac.compare_digest(entered.encode(), expected.encode())


def max_runs_per_session() -> int:
    try:
        return max(1, int(_secret("QLP_AGENT_MAX_RUNS") or 10))
    except ValueError:
        return 10


def fundamentals_providers() -> tuple[tuple[Any, ...], list[str]]:
    """SEC EDGAR first (primary source) when a User-Agent is configured, then Yahoo Finance."""
    providers: list[Any] = []
    notes = []
    agent = _secret(SEC_USER_AGENT_ENV)
    if agent:
        providers.append(SecEdgarProvider(user_agent=agent))
    else:
        notes.append(f"SEC EDGAR is off: set {SEC_USER_AGENT_ENV} to 'Your Name your@email.com' to use the primary "
                     "source for US filers' statements.")
    providers.append(YahooFundamentalsProvider())
    return tuple(providers), notes


def risk_limits_form() -> RiskLimits:
    """The user's limits, edited in an expander; illustrative (and labelled) until the user confirms them."""
    ss = st.session_state
    with st.expander("Risk limits", icon=":material/shield:", expanded=False):
        st.toggle("These are my own limits", key="ag_limits_own",
                        help="Until you confirm, sizes use illustrative defaults and every output says so.")
        c1, c2 = st.columns(2)
        equity = c1.number_input("Account equity", min_value=100.0, value=100_000.0, step=1_000.0, key="ag_equity")
        cash = c2.number_input("Available cash", min_value=0.0, value=100_000.0, step=1_000.0, key="ag_cash")
        c3, c4, c5 = st.columns(3)
        per_trade = c3.number_input("Risk per trade %", 0.05, 5.0, 0.5, 0.05, key="ag_risk")
        daily = c4.number_input("Daily loss limit %", 0.1, 25.0, 2.0, 0.1, key="ag_daily")
        drawdown = c5.number_input("Drawdown halt %", 1.0, 100.0, 15.0, 1.0, key="ag_dd")
        c6, c7, c8 = st.columns(3)
        position = c6.number_input("Max position %", 1.0, 100.0, 10.0, 1.0, key="ag_pos")
        sector = c7.number_input("Max sector %", 1.0, 100.0, 30.0, 1.0, key="ag_sector")
        gross = c8.number_input("Max gross exposure %", 1.0, 100.0, 100.0, 1.0, key="ag_gross",
                                help="At most 100%: no leverage.")
        st.caption("Long only, no leverage, no short selling. A failed limit check always means: do not trade.")
    try:
        limits = RiskLimits(account_equity=equity, available_cash=min(cash, equity),
                            max_risk_per_trade_pct=per_trade / 100, max_daily_loss_pct=daily / 100,
                            max_position_pct=position / 100, max_sector_pct=sector / 100,
                            max_gross_exposure_pct=gross / 100, max_drawdown_pct=drawdown / 100,
                            illustrative=not ss.get("ag_limits_own", False))
    except ValueError as exc:
        st.error(f"Invalid risk limits: {exc}. Using illustrative defaults.", icon=":material/error:")
        limits = illustrative_limits(equity)
    return limits


def paper_book() -> PaperBook:
    """The simulated paper account (in the research database: per session on public deployments)."""
    return PaperBook(get_store(), account="default")


def holdings() -> tuple[Holding, ...]:
    return tuple(st.session_state.get("ag_holdings", ()))


def build_context(limits: RiskLimits, benchmark: str, providers: tuple[Any, ...],
                  load: Callable[[str], MarketDataset] = load_prices) -> ToolContext:
    def safe_load(ticker: str) -> MarketDataset:
        try:
            return load(ticker)
        except DataProviderError:
            raise
        except ValueError as exc:  # invalid ticker or date input: report it as a data problem
            raise DataProviderError(str(exc)) from exc

    return ToolContext(load_prices=safe_load, risk_limits=limits, fundamentals=providers,
                       holdings=holdings(), paper_book=paper_book(), benchmark=benchmark)
