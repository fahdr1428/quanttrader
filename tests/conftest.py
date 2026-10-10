"""Shared synthetic datasets.

All data in the test suite is synthetic and constructed so that expected
outcomes can be computed by hand. Nothing here is real market data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers import random_walk


@pytest.fixture
def walk() -> pd.DataFrame:
    return random_walk()


@pytest.fixture
def long_walk() -> pd.DataFrame:
    return random_walk(n=1500, seed=11)
