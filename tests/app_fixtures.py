"""Build a ResearchData object from synthetic prices for UI smoke tests.

The dataset is explicitly labelled as synthetic test data; it exists only so
that every page can be exercised without network access.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quantlab.data import CacheStatus, MarketDataset, fingerprint_prices
from quantlab.validation import clean_price_frame
from tests.helpers import random_walk


def synthetic_research_data(n: int = 1400, seed: int = 17, eval_offset: int = 260):
    from ui.state import ResearchData

    raw = random_walk(n=n, seed=seed, start="2019-01-02")
    prices, report = clean_price_frame(raw)
    dataset = MarketDataset(
        ticker="SYNTH",
        prices=prices,
        raw=raw,
        provider_key="csv",
        source_name="Synthetic test data (not real market data)",
        adjustment="as_provided",
        requested_start=prices.index[0],
        requested_end=prices.index[-1],
        fetched_at=datetime.now(timezone.utc),
        cache_status=CacheStatus.UPLOADED,
        quality=report,
        fingerprint=fingerprint_prices(prices),
    )
    fp = dataset.fingerprint
    return ResearchData(
        asset=dataset,
        benchmark_dataset=None,
        benchmark_ticker="SYNTH",
        prices=prices,
        bench_prices=prices,
        dropped_dates=pd.DatetimeIndex([]),
        fingerprint=fp,
        bench_fingerprint=fp,
        eval_start=prices.index[eval_offset],
        eval_end=prices.index[-1],
        requested_start=prices.index[eval_offset].date(),
        requested_end=prices.index[-1].date(),
        warmup_days=400,
    )
