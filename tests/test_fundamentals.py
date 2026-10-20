"""Fundamentals: provider mapping, point-in-time SEC statements and hand-checked metrics (offline fakes only)."""

from __future__ import annotations

import json
import urllib.error
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.data import MalformedDataError, ProviderConnectionError, RateLimitError, TickerNotFoundError
from quantlab.fundamentals import (
    DIVIDEND,
    EARNINGS,
    EARNINGS_RELEASE,
    METRIC_KEYS,
    SEC_FACTS_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    SEC_USER_AGENT_ENV,
    SPLIT,
    FinancialStatements,
    NotSupportedError,
    RequestThrottle,
    SecEdgarProvider,
    StatementPeriod,
    YahooFundamentalsProvider,
    fundamental_metrics,
    statements_quality,
)
from quantlab.provenance import Fact, Provenance, Timeliness

NOW = datetime(2024, 3, 1, 21, 0, tzinfo=timezone.utc)
FY23, FY22 = pd.Timestamp("2023-12-31"), pd.Timestamp("2022-12-31")


# --------------------------------------------------------------------------- #
# Yahoo Finance: a fake yfinance Ticker
# --------------------------------------------------------------------------- #


def _yahoo_frame(rows):
    return pd.DataFrame(rows, index=[FY23, FY22]).T


class FakeTicker:
    def __init__(self, **attrs):
        self.info = {"longName": "Acme <b>Corp</b>", "currency": "USD", "financialCurrency": "USD",
                     "sharesOutstanding": 100.0, "sector": "Industrials", "industry": "Widgets",
                     "country": "United States", "fullExchangeName": "NYSE"}
        self.income_stmt = _yahoo_frame({"Total Revenue": [1000.0, np.nan], "Operating Revenue": [990.0, 800.0],
                                         "Gross Profit": [400.0, 300.0], "Net Income": [150.0, 100.0],
                                         "Diluted EPS": [1.5, 1.0], "Unrelated Row": [7.0, 7.0]})
        self.balance_sheet = _yahoo_frame({"Total Assets": [2000.0, 1800.0], "Total Debt": [300.0, 250.0],
                                           "Cash And Cash Equivalents": [100.0, 90.0]})
        self.cashflow = _yahoo_frame({"Operating Cash Flow": [180.0, 140.0], "Capital Expenditure": [-30.0, -20.0]})
        self.news = []
        self.calendar = {}
        self.dividends = pd.Series(dtype=float)
        self.splits = pd.Series(dtype=float)
        self.insider_transactions = pd.DataFrame()
        self.earnings_dates = pd.DataFrame()
        self.__dict__.update(attrs)


def yahoo(**attrs):
    return YahooFundamentalsProvider(ticker_factory=lambda s: FakeTicker(**attrs), sleep=lambda s: None,
                                     clock=lambda: NOW)


def test_yahoo_statements_map_labels_and_never_fill_gaps_with_zero():
    st = yahoo().statements("acme")
    assert st.ticker == "ACME" and st.currency == "USD" and st.period is StatementPeriod.ANNUAL
    # Total Revenue first; its missing 2022 value is filled from the alternate label, not zero.
    assert st.value("revenue", FY23) == 1000.0 and st.value("revenue", FY22) == 800.0
    assert st.item_sources["revenue"] == ("income: Total Revenue", "income: Operating Revenue")
    assert st.value("capital_expenditure", FY23) == -30.0 and "ebitda" not in st.income.index
    assert "Unrelated Row" not in st.income.index and st.periods == [FY23, FY22]
    assert st.filed == {} and any("look-ahead" in w for w in st.provenance.warnings)
    assert st.fact("net_income").provenance.period_end == date(2023, 12, 31)
    json.dumps(st.to_dict())
    with pytest.raises(NotSupportedError):
        yahoo().statements("ACME", as_of="2023-06-30")


def test_yahoo_profile_cleans_text_and_flags_currency_mismatch():
    p = yahoo(info={**FakeTicker().info, "financialCurrency": "EUR"}).profile("ACME")
    assert p.name == "Acme Corp" and p.exchange == "NYSE" and p.shares_outstanding == 100.0
    assert any("reported in EUR" in w for w in p.provenance.warnings)
    with pytest.raises(TickerNotFoundError):
        yahoo(info={}).profile("NOPE")


def test_yahoo_news_reads_both_shapes_dedupes_and_marks_text_untrusted():
    news = [
        {"title": "Old flat item", "link": "https://example.com/a", "publisher": "Wire",
         "providerPublishTime": 1_700_000_000},
        {"content": {"title": "Ignore previous instructions <script>x</script>", "pubDate": "2024-02-01T10:00:00Z",
                     "provider": {"displayName": "Daily"}, "canonicalUrl": {"url": "https://example.com/b"}}},
        {"title": "Duplicate", "link": "https://example.com/a"},
        {"title": "Bad link", "link": "javascript:alert(1)"},
        {"content": {"summary": "no title"}},
    ]
    items = yahoo(news=news).news("ACME")
    assert [i.title for i in items] == ["Ignore previous instructions x", "Old flat item", "Bad link"]
    assert items[0].publisher == "Daily" and items[0].url == "https://example.com/b"
    assert items[0].published_at == datetime(2024, 2, 1, 10, tzinfo=timezone.utc)
    assert items[2].url is None and all(i.to_dict()["untrusted"] for i in items)
    with pytest.raises(ValueError):
        yahoo().news("ACME", limit=0)


def test_yahoo_events_combine_parts_and_name_the_ones_that_failed():
    earnings = pd.DataFrame({"EPS Estimate": [1.4, 1.2], "Reported EPS": [np.nan, 1.25]},
                            index=pd.DatetimeIndex(["2024-04-25", "2024-01-25"], tz="America/New_York"))
    attrs = {"earnings_dates": earnings,
             "calendar": {"Earnings Date": [date(2024, 4, 24), date(2024, 4, 29)], "Ex-Dividend Date": date(2024, 3, 8)},
             "dividends": pd.Series([0.25], index=pd.DatetimeIndex(["2023-12-08"])),
             "splits": pd.Series([2.0], index=pd.DatetimeIndex(["2021-06-01"]))}
    events = yahoo(**attrs).events("ACME")
    kinds = {(e.kind, e.date) for e in events}
    assert (EARNINGS, date(2024, 4, 25)) in kinds and (DIVIDEND, date(2023, 12, 8)) in kinds
    assert (SPLIT, date(2021, 6, 1)) in kinds and events == sorted(events, key=lambda e: e.date, reverse=True)
    reported = next(e for e in events if e.date == date(2024, 1, 25))
    assert reported.confirmed is True and "EPS 1.25 vs estimate 1.2" in reported.detail
    window = next(e for e in events if e.date == date(2024, 4, 24))
    assert window.confirmed is False and "not a confirmed date" in window.detail

    class Broken(FakeTicker):
        @property
        def splits(self):
            raise ConnectionError("connection reset")

        @splits.setter
        def splits(self, value):
            pass

    partial = YahooFundamentalsProvider(ticker_factory=lambda s: Broken(**{k: v for k, v in attrs.items()
                                                                           if k != "splits"}),
                                        retries=0, clock=lambda: NOW).events("ACME")
    assert partial and all("Not retrieved: split history" in e.provenance.warnings[-1] for e in partial)


def test_yahoo_insider_transactions_keep_missing_values_missing():
    frame = pd.DataFrame({"Insider": ["DOE JANE", "ROE RICH"], "Position": ["CEO", "Director"],
                          "Start Date": ["2024-01-10", "2024-02-05"], "Text": ["Sale at price 50.00", None],
                          "Shares": [1000, np.nan], "Value": [50_000.0, np.nan]})
    trades = yahoo(insider_transactions=frame).insider_transactions("ACME")
    assert [t.insider for t in trades] == ["ROE RICH", "DOE JANE"]
    assert trades[0].shares is None and trades[0].value is None and trades[1].value == 50_000.0


def test_yahoo_retries_transient_errors_only():
    calls, sleeps = [], []

    def factory(symbol):
        calls.append(symbol)
        if len(calls) < 3:
            raise Exception("429 Too Many Requests")
        return FakeTicker()

    provider = YahooFundamentalsProvider(ticker_factory=factory, retries=2, backoff_seconds=1.0,
                                         sleep=sleeps.append, clock=lambda: NOW)
    assert provider.profile("ACME").name == "Acme Corp" and sleeps == [1.0, 2.0]
    gone = YahooFundamentalsProvider(ticker_factory=lambda s: (_ for _ in ()).throw(Exception("404 Not Found")),
                                     sleep=sleeps.append, clock=lambda: NOW)
    with pytest.raises(TickerNotFoundError):
        gone.profile("GONE")
    assert sleeps == [1.0, 2.0]   # not retried
    limited = YahooFundamentalsProvider(ticker_factory=lambda s: (_ for _ in ()).throw(Exception("Too Many Requests")),
                                        retries=1, sleep=lambda s: None, clock=lambda: NOW)
    with pytest.raises(RateLimitError):
        limited.profile("ACME")


# --------------------------------------------------------------------------- #
# SEC EDGAR: a fake http_get serving companyfacts / submissions / tickers
# --------------------------------------------------------------------------- #

CIK = "0000001234"


def _fact(val, end, filed, form, fp=None, start=None, accn="0001234-24-000010"):
    entry = {"val": val, "end": end, "filed": filed, "form": form, "fp": fp, "accn": accn}
    if start:
        entry["start"] = start
    return entry


def _fy(val, year, filed, accn="0001234-24-000010"):
    return _fact(val, f"{year}-12-31", filed, "10-K", "FY", f"{year}-01-01", accn)


COMPANYFACTS = {"facts": {
    "us-gaap": {
        "Revenues": {"units": {"USD": [
            _fy(800, 2022, "2023-02-15", "0001234-23-000005"),
            _fy(820, 2022, "2024-02-15"),                       # restated in the next 10-K
            _fy(1000, 2023, "2024-02-15"),
            _fact(240, "2023-03-31", "2023-05-01", "10-Q", "Q1", "2023-01-01"),
            _fact(740, "2023-09-30", "2023-11-01", "10-Q", "Q3", "2023-01-01"),
            _fact(250, "2023-06-30", "2023-08-01", "10-Q", "Q2", "2023-04-01"),
            _fact(250, "2023-09-30", "2023-11-01", "10-Q", "Q3", "2023-07-01"),
        ]}},
        "NetIncomeLoss": {"units": {"USD": [_fy(100, 2022, "2023-02-15"), _fy(150, 2023, "2024-02-15")]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [_fy(1.0, 2022, "2023-02-15"),
                                                             _fy(1.5, 2023, "2024-02-15")]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            _fact(50, "2023-03-31", "2023-05-01", "10-Q", "Q1", "2023-01-01"),
            _fact(120, "2023-06-30", "2023-08-01", "10-Q", "Q2", "2023-01-01"),
            _fact(200, "2023-09-30", "2023-11-01", "10-Q", "Q3", "2023-01-01"),
            _fy(290, 2023, "2024-02-15"), _fy(140, 2022, "2023-02-15"),
        ]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_fy(30, 2023, "2024-02-15")]}},
        "Assets": {"units": {"USD": [_fact(2000, "2023-12-31", "2024-02-15", "10-K", "FY"),
                                     _fact(1800, "2022-12-31", "2023-02-15", "10-K", "FY")]}},
        "LongTermDebt": {"units": {"USD": [_fact(250, "2023-12-31", "2024-02-15", "10-K", "FY")]}},
        "ShortTermBorrowings": {"units": {"USD": [_fact(50, "2023-12-31", "2024-02-15", "10-K", "FY")]}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_fact(100, "2023-12-31", "2024-02-15", "10-K")]}},
        "Liabilities": {"units": {"USD": [{"val": "oops", "end": "2023-12-31", "filed": "2024-02-15"}]}},
    },
    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        _fact(100, "2024-02-01", "2024-02-15", "10-K", "FY"), _fact(95, "2023-02-01", "2023-02-15", "10-K", "FY")]}}},
}}

SUBMISSIONS = {
    "name": "ACME CORP", "tickers": ["ACME"], "exchanges": ["NYSE"], "sicDescription": "Widgets",
    "addresses": {"business": {"isForeignLocation": 0}},
    "filings": {"recent": {
        "accessionNumber": ["0001234-24-000010", "0001234-24-000005", "0001234-23-000020"],
        "filingDate": ["2024-02-15", "2024-01-25", "2023-11-01"],
        "reportDate": ["2023-12-31", "2024-01-25", "2023-09-30"],
        "acceptanceDateTime": ["2024-02-15T16:05:00.000Z", "2024-01-25T21:01:00.000Z", "2023-11-01T12:00:00.000Z"],
        "form": ["10-K", "8-K", "10-Q"],
        "primaryDocument": ["acme-20231231.htm", "ex99.htm", "../evil.htm"],
        "primaryDocDescription": ["10-K", "8-K", "10-Q"],
        "items": ["", "2.02,9.01", ""],
    }, "files": []},
}


class FakeSec:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
        self.routes = {SEC_TICKERS_URL: {"0": {"cik_str": 1234, "ticker": "ACME", "title": "Acme Corp"}},
                       SEC_FACTS_URL.format(cik=CIK): COMPANYFACTS,
                       SEC_SUBMISSIONS_URL.format(cik=CIK): SUBMISSIONS}

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        if self.fail is not None:
            raise self.fail
        return json.dumps(self.routes[url]).encode()


def sec(http=None, **kw):
    kw.setdefault("use_cache", False)
    return SecEdgarProvider(user_agent="Test Runner test@example.com", http_get=http or FakeSec(),
                            throttle=RequestThrottle(1e6), clock=lambda: NOW, sleep=lambda s: None, **kw)


def test_sec_statements_keep_first_filed_values_and_their_filing_times():
    http = FakeSec()
    st = sec(http).statements("acme")
    assert st.currency == "USD" and st.periods == [FY23, FY22]
    assert st.value("revenue", FY22) == 800.0 and st.value("revenue", FY23) == 1000.0   # as originally reported
    assert st.value("capital_expenditure", FY23) == -30.0                               # payments negated
    assert st.value("total_debt", FY23) == 300.0 and st.value("total_assets", FY22) == 1800.0
    assert "total_liabilities" not in st.balance.index                                  # unreadable, not zero
    # Filed 2024-02-15: public by the end of that day in New York (04:59:59 UTC the next day in winter).
    assert st.filed[FY23] == datetime(2024, 2, 16, 4, 59, 59, tzinfo=timezone.utc)
    fact = st.fact("revenue")
    assert fact.provenance.period_start == date(2023, 1, 1) and fact.provenance.published_at == st.filed[FY23]
    warnings = " ".join(st.provenance.warnings)
    assert "differ between filings" in warnings and "1 XBRL observation(s) could not be read" in warnings
    assert st.provenance.source_url == SEC_FACTS_URL.format(cik=CIK)
    assert all(h["User-Agent"] == "Test Runner test@example.com" for _, h in http.calls)
    assert sec().statements("ACME", restated=True).value("revenue", FY22) == 820.0


def test_sec_as_of_rebuilds_what_was_public_on_that_date():
    before = sec().statements("ACME", as_of="2023-06-30")
    assert before.periods == [FY22] and before.value("revenue", FY22) == 800.0
    assert any("only values filed by 2023-06-30" in w for w in before.provenance.warnings)
    after = sec().statements("ACME", as_of=date(2024, 3, 1))
    assert after.value("revenue", FY22) == 820.0                                       # latest known by then
    with pytest.raises(TickerNotFoundError, match="filed by 2020-01-01"):
        sec().statements("ACME", as_of="2020-01-01")
    with pytest.raises(ValueError, match="timezone-aware"):
        sec().statements("ACME", as_of=datetime(2024, 3, 1))


def test_sec_quarterly_statements_derive_quarters_from_year_to_date_totals():
    q = sec().statements("ACME", StatementPeriod.QUARTERLY)
    ocf = q.row("operating_cash_flow")
    assert ocf[pd.Timestamp("2023-03-31")] == 50 and ocf[pd.Timestamp("2023-06-30")] == 70
    assert ocf[pd.Timestamp("2023-09-30")] == 80 and ocf[pd.Timestamp("2023-12-31")] == 90
    assert q.value("revenue", "2023-12-31") == 260.0                                   # Q4 = FY − nine months
    assert q.value("revenue", "2023-06-30") == 250.0                                   # reported directly
    assert any("derived as the difference" in w for w in q.provenance.warnings)


def test_sec_profile_and_filings_come_from_the_submissions_index():
    p = sec().profile("ACME")
    assert (p.name, p.exchange, p.industry, p.country) == ("ACME CORP", "NYSE", "Widgets", "United States")
    assert p.sector is None and p.shares_outstanding == 100 and p.shares_as_of == date(2024, 2, 1)
    filings = sec().filings("ACME")
    assert [f.form for f in filings] == ["10-K", "8-K", "10-Q"]
    assert filings[0].url == "https://www.sec.gov/Archives/edgar/data/1234/000123424000010/acme-20231231.htm"
    assert filings[2].url.endswith("/000123423000020/0001234-23-000020-index.htm")    # unsafe path not used
    assert filings[1].items == ("2.02", "9.01")
    assert filings[0].provenance.published_at == datetime(2024, 2, 15, 16, 5, tzinfo=timezone.utc)
    assert [f.form for f in sec().filings("ACME", forms=["10-q"])] == ["10-Q"]
    events = sec().events("ACME")
    assert [(e.kind, e.date) for e in events] == [(EARNINGS_RELEASE, date(2024, 1, 25))]
    with pytest.raises(NotSupportedError):
        sec().news("ACME")
    with pytest.raises(TickerNotFoundError):
        sec().profile("ZZZZ")


def test_sec_requires_an_identifying_user_agent(monkeypatch):
    monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
    http = FakeSec()
    anonymous = SecEdgarProvider(http_get=http, use_cache=False, throttle=RequestThrottle(1e6))
    with pytest.raises(ProviderConnectionError, match=SEC_USER_AGENT_ENV):
        anonymous.statements("ACME")
    assert http.calls == []                                                            # nothing was sent
    with pytest.raises(ProviderConnectionError):
        SecEdgarProvider(user_agent="a\r\nX-Injected: 1", http_get=http, use_cache=False).profile("ACME")
    monkeypatch.setenv(SEC_USER_AGENT_ENV, "Env User env@example.com")
    SecEdgarProvider(http_get=http, use_cache=False, throttle=RequestThrottle(1e6)).statements("ACME")
    assert http.calls[0][1]["User-Agent"] == "Env User env@example.com"


def test_sec_http_errors_are_typed_and_only_transient_ones_retried():
    def http_error(code):
        return urllib.error.HTTPError("https://data.sec.gov/x", code, "err", {}, None)

    for code, cls in ((404, TickerNotFoundError), (403, ProviderConnectionError), (503, ProviderConnectionError)):
        http = FakeSec(fail=http_error(code))
        with pytest.raises(cls):
            sec(http, retries=1).statements("ACME")
        assert len(http.calls) == (2 if code == 503 else 1)
    with pytest.raises(MalformedDataError, match="other than JSON"):
        sec(lambda url, headers: b"<html>blocked</html>").statements("ACME")


def test_sec_cache_serves_fresh_copies_and_flags_stale_ones(tmp_path):
    first = FakeSec()
    sec(first, use_cache=True, cache_dir=tmp_path).statements("ACME")
    offline = FakeSec(fail=urllib.error.URLError("network down"))
    cached = sec(offline, use_cache=True, cache_dir=tmp_path).statements("ACME")
    assert offline.calls == [] and cached.provenance.cache_status == "cached"
    later = SecEdgarProvider(user_agent="Test Runner test@example.com", http_get=offline, use_cache=True,
                             cache_dir=tmp_path, throttle=RequestThrottle(1e6), sleep=lambda s: None,
                             clock=lambda: NOW + timedelta(days=2), retries=0)
    stale = later.statements("ACME")
    assert stale.provenance.cache_status == "stale_cache"
    assert any("may be out of date" in w for w in stale.provenance.warnings)


# --------------------------------------------------------------------------- #
# Metrics, checked by hand
# --------------------------------------------------------------------------- #


def statements(income, balance=None, cashflow=None, period=StatementPeriod.ANNUAL, ends=(FY23, FY22),
               currency="USD", today=NOW):
    def frame(rows):
        if not rows:
            return pd.DataFrame(index=pd.Index([], dtype=object), columns=pd.DatetimeIndex([]), dtype="float64")
        return pd.DataFrame.from_dict({k: [*v, *[np.nan] * (len(ends) - len(v))] for k, v in rows.items()},
                                      orient="index", columns=pd.DatetimeIndex(list(ends)), dtype="float64")

    prov = Provenance(source="Hand-built statements", retrieved_at=today, timeliness=Timeliness.PERIODIC_FILING,
                      currency=currency)
    return FinancialStatements("ACME", currency, frame(income), frame(balance), frame(cashflow), period, prov, {})


INCOME = {"revenue": [1000, 800], "gross_profit": [400, 300], "operating_income": [200, 150],
          "net_income": [150, 100], "eps_diluted": [1.5, 1.0], "depreciation_amortization": [50, 40]}
BALANCE = {"total_debt": [300], "cash": [100], "total_equity": [900], "current_assets": [600],
           "current_liabilities": [300], "shares_outstanding": [100], "total_assets": [2000],
           "total_liabilities": [1100]}
CASHFLOW = {"operating_cash_flow": [180, 140], "capital_expenditure": [-30, -20]}


def price(value=50.0, currency="USD"):
    prov = Provenance(source="Test close", retrieved_at=NOW, timeliness=Timeliness.END_OF_DAY,
                      period_end=date(2024, 3, 1), currency=currency)
    return Fact("close", value, prov, unit=currency)


def test_metrics_match_hand_calculations():
    m = fundamental_metrics(statements(INCOME, BALANCE, CASHFLOW), price())
    assert list(m) == list(METRIC_KEYS)
    expected = {"revenue_growth_yoy": 0.25, "earnings_growth_yoy": 0.5, "gross_margin": 0.4, "operating_margin": 0.2,
                "net_margin": 0.15, "free_cash_flow": 150, "fcf_margin": 0.15, "market_cap": 5000,
                "enterprise_value": 5200, "pe_trailing": 50 / 1.5, "price_to_sales": 5.0,
                "ev_to_ebitda": 5200 / 250, "fcf_yield": 0.03, "net_debt_to_ebitda": 0.8,
                "debt_to_equity": 300 / 900, "current_ratio": 2.0, "cash_conversion": 1.2}
    for key, value in expected.items():
        assert m[key].value == pytest.approx(value), key
    assert "approximation" in m["ev_to_ebitda"].inputs[1].formula          # EBITDA = OI + |D&A|, labelled
    assert "balance sheet" in m["market_cap"].warnings[0]                   # share count source disclosed
    assert "latest fiscal year" in m["gross_margin"].formula
    json.dumps({k: v.to_dict() for k, v in m.items()})


def test_undefined_ratios_are_none_with_a_reason_not_a_number():
    losing = {**INCOME, "net_income": [-50, 100], "eps_diluted": [-0.5, 1.0]}
    m = fundamental_metrics(statements(losing, BALANCE, CASHFLOW), price())
    assert m["pe_trailing"].value is None and "not positive" in m["pe_trailing"].reason
    assert m["earnings_growth_yoy"].value == pytest.approx(-1.5)
    from_loss = fundamental_metrics(statements({**INCOME, "net_income": [150, -100]}, BALANCE, CASHFLOW), price())
    assert from_loss["earnings_growth_yoy"].value is None and "non-positive base" in from_loss["earnings_growth_yoy"].reason
    no_debt = fundamental_metrics(statements(INCOME, {k: v for k, v in BALANCE.items() if k != "total_debt"},
                                             CASHFLOW), price())
    assert no_debt["enterprise_value"].value is None and "not treated as zero" in no_debt["enterprise_value"].reason
    no_price = fundamental_metrics(statements(INCOME, BALANCE, CASHFLOW), price(value=None))
    assert no_price["market_cap"].value is None and no_price["fcf_yield"].value is None


def test_a_currency_mismatch_blocks_price_based_metrics_but_not_margins():
    m = fundamental_metrics(statements(INCOME, BALANCE, CASHFLOW), price(currency="EUR"))
    for key in ("enterprise_value", "pe_trailing", "price_to_sales", "ev_to_ebitda", "fcf_yield"):
        assert m[key].value is None and "no currency conversion" in m[key].reason, key
    assert m["gross_margin"].value == pytest.approx(0.4) and m["market_cap"].unit == "EUR"


def test_ttm_sums_four_quarters_and_both_sides_of_a_ratio_share_one_basis():
    ends = pd.date_range("2022-03-31", periods=8, freq="QE")[::-1]
    quarterly = statements({"revenue": [260, 250, 250, 240, 210, 200, 200, 190],
                            "net_income": [40, 40, 35, 35, 30, 25, 25, 20]},
                           BALANCE, period=StatementPeriod.QUARTERLY, ends=tuple(ends))
    m = fundamental_metrics(statements(INCOME, BALANCE, CASHFLOW), price(), ttm=quarterly)
    assert m["revenue_growth_yoy"].value == pytest.approx(1000 / 800 - 1) and "TTM" in m["revenue_growth_yoy"].formula
    assert m["net_margin"].value == pytest.approx(150 / 1000) and "TTM" in m["net_margin"].formula
    # gross_profit is not reported quarterly, so the gross margin uses the fiscal year for both parts.
    assert "latest fiscal year" in m["gross_margin"].formula and "TTM not available" in m["gross_margin"].formula
    with pytest.raises(ValueError, match="quarterly"):
        fundamental_metrics(statements(INCOME), price(), ttm=statements(INCOME))


def test_statement_quality_flags_signs_identities_gaps_and_staleness():
    clean = statements_quality(statements(INCOME, BALANCE, CASHFLOW))
    assert not any(w.startswith(("Sign anomaly", "Inconsistent", "Stale", "total_assets")) for w in clean)
    assert any("not reported" in w and "ebitda" in w for w in clean)
    broken = statements({**INCOME, "gross_profit": [1200, 300]},
                        {**BALANCE, "total_liabilities": [500]},
                        {**CASHFLOW, "capital_expenditure": [30, -20]})
    warnings = " | ".join(statements_quality(broken, today=date(2025, 6, 1)))
    assert "gross_profit exceeds revenue" in warnings and "capital_expenditure is positive" in warnings
    assert "total_assets differ" in warnings and "Stale" in warnings
    gap = statements({"revenue": [1000, 800], "net_income": [np.nan, 100]}, ends=(FY23, pd.Timestamp("2021-12-31")))
    gap_warnings = " | ".join(statements_quality(gap))
    assert "no value for the latest period" in gap_warnings and "730 days between" in gap_warnings
