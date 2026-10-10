#!/usr/bin/env python
"""Controlled experiment: do the models find real structure — and only real structure?

Generates synthetic markets with a *known* data-generating process and reports
how each strategy behaves:

* **noise** — a random walk with no predictable component. A model that appears
  to forecast it is leaking future information or overfitting.
* **momentum** — the expected daily return is +drift after a rising quarter and
  -drift after a falling one (a planted time-series-momentum effect).

Everything is seeded, so the output is reproducible. Results on synthetic data
say nothing about real markets; they test whether the machinery works.

    python scripts/synthetic_model_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from quantlab.analytics import run_strategy  # noqa: E402
from quantlab.backtest import run_buy_and_hold  # noqa: E402
from quantlab.config import BacktestConfig  # noqa: E402
from quantlab.ml import classification_report  # noqa: E402
from quantlab.strategies import create_strategy  # noqa: E402
from tests.helpers import make_prices  # noqa: E402

STRATEGIES = ("buy_and_hold", "ma_crossover", "adaptive_ensemble", "ml_logistic")
SHORT = {"buy_and_hold": "Buy & hold", "ma_crossover": "MA 20/50", "adaptive_ensemble": "Ensemble",
         "ml_logistic": "ML classifier"}


def market(n: int, seed: int, drift: float, noise: float = 0.01, lookback: int = 63):
    rng = np.random.default_rng(seed)
    logp = np.zeros(n)
    for t in range(1, n):
        mu = 0.0 if t <= lookback else (drift if logp[t - 1] > logp[t - 1 - lookback] else -drift)
        logp[t] = logp[t - 1] + mu + rng.normal(0, noise)
    close = 100 * np.exp(logp)
    opens = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.002, n))
    return make_prices(close, opens, start="2010-01-04")


def main() -> None:
    cfg = BacktestConfig()
    print("| Market | Seed | " + " | ".join(f"{SHORT[k]} Sharpe" for k in STRATEGIES) + " | ML AUC | ML hit-rate edge |")
    print("|---|---|" + "---|" * (len(STRATEGIES) + 2))
    for label, drift in (("Noise (no signal)", 0.0), ("Weak momentum", 0.0005), ("Moderate momentum", 0.001)):
        for seed in (1, 2, 3):
            prices = market(3000, seed, drift)
            start = prices.index[700]
            bench = run_buy_and_hold(prices, cfg, start)
            sharpes, auc, edge = [], None, None
            for key in STRATEGIES:
                run = run_strategy(prices, create_strategy(key), cfg, start, benchmark=bench)
                sharpes.append(run.report.value("sharpe"))
                if key == "ml_logistic":
                    rep = classification_report(run.net.signals["prob_up_raw"], prices["close"], 21)
                    auc, edge = rep["auc"], rep["accuracy_edge"]
            cells = " | ".join(f"{s:+.2f}" for s in sharpes)
            print(f"| {label} | {seed} | {cells} | {auc:.3f} | {edge:+.1%} |")


if __name__ == "__main__":
    main()
