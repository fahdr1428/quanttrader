"""Methodology — how results are produced, and the test behind every claim."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantlab.config import METHODOLOGY_VERSION, TRADING_DAYS_PER_YEAR
from quantlab.evidence import THRESHOLDS
from quantlab.traceability import CLAIMS
from ui import components as ui

ui.page_header(
    "Methodology",
    "How every number is produced",
    "The conventions behind the results, and for each methodological claim the code that implements it and "
    "the automated test that fails if it stops being true.",
)

ui.section("Conventions", f"Methodology version {METHODOLOGY_VERSION}. Changing any of these bumps the version.")
ui.kv_table({
    "Timing": "Signal at the close of session t; fill at the open of session t + 1. A final-session signal is "
              "reported, not executed.",
    "Positions": "Long-only, unleveraged target weights between 0% and 100%; partial positions rebalance only "
                 "outside a band.",
    "Costs": "Proportional fee on traded notional plus adverse slippage on every fill; the same signals are "
             "re-run without costs to show the cost drag.",
    "Annualisation": f"{TRADING_DAYS_PER_YEAR} sessions a year; Sharpe-based statistics use returns in excess of "
                     "the risk-free rate.",
    "Data": "Daily exchange sessions only; invalid rows removed and reported, never filled in.",
    "Randomness": "The engine is deterministic; bootstrap, permutation and grid tests and synthetic markets use "
                  "fixed seeds.",
})

ui.section("Evidence thresholds", "Used by the checklist on the Overview. Kept in one place: quantlab/evidence.py.")
ui.show_table(pd.DataFrame(
    [{"Check": key, "Threshold": value, "Why": why} for key, (value, why) in THRESHOLDS.items()]
))

ui.section("Claim → implementation → test",
           "Every row is verified in CI: the implementation must exist and each listed test must exist and pass.")
query = st.text_input("Filter", placeholder="e.g. look-ahead, Sharpe, CSV", key="meth_filter").strip().lower()
rows = [
    {"Area": c.area, "Claim": c.claim, "Implementation": c.implementation,
     "Tests": "\n".join(t.split("::")[1] for t in c.tests)}
    for c in CLAIMS
    if not query or query in f"{c.area} {c.claim} {c.implementation}".lower()
]
if rows:
    ui.show_table(pd.DataFrame(rows))
else:
    st.info("No claim matches that filter.")
st.caption("Tests live in the tests/ folder of the repository; run them with `python -m pytest`.")
ui.disclaimer()
