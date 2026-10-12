"""Evidence checklist: what has — and has not — been tested for a strategy result.

A backtest's headline numbers say little on their own. This module collects the
checks the app can run into one transparent list, each with an explicit
threshold, so a reader can see in seconds which questions a result survives,
which it fails, and which were never asked. It deliberately produces **no
aggregate score**: passing every check would still not make a strategy
profitable, and a single number would read as a certificate.

The function is pure (no Streamlit). It only reads results that were already
computed; a check whose analysis has not been run is reported as ``not_run`` —
never as passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

Status = Literal["pass", "caution", "fail", "not_run"]

# Every threshold in one place, with the reason it was chosen.
THRESHOLDS: dict[str, tuple[float, str]] = {
    "min_sessions": (756, "About three years of daily data: below this, Sharpe ratios are very noisy."),
    "dsr": (0.95, "Deflated Sharpe at 95%: the result beats the best of the configurations tried by luck."),
    "spa_p": (0.05, "Hansen's SPA at the 5% level: the best grid point beats the benchmark after the search."),
    "pbo": (0.5, "Below 50%: the in-sample winner usually stays in the top half out of sample."),
    "permutation_p": (0.10, "10% level: a lenient bar, because the permutation test has modest power."),
    "wf_retention": (0.5, "Out-of-sample Sharpe at least half the in-sample level."),
}


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    question: str
    status: Status
    value: str
    threshold: str
    explanation: str
    where: str


def _fmt(x: float | None, spec: str) -> str:
    return "n/a" if x is None or pd.isna(x) else format(x, spec)


def evidence_checklist(
    *,
    report: Any,
    accounting_valid: bool,
    data_warnings: int,
    synthetic: bool,
    parameters_prespecified: bool,
    n_trials: int,
    dsr: float | None,
    exposure_matched_cagr: float | None = None,
    permutation_p: float | None = None,
    selection: Any = None,
    walk_forward: dict[str, float] | None = None,
    regimes: pd.DataFrame | None = None,
) -> list[EvidenceItem]:
    """Build the checklist from results that already exist.

    ``walk_forward`` holds ``oos_sharpe``, ``is_mean`` and ``oos_mean`` for a
    walk-forward run of the same strategy on the same sample; ``selection`` is
    an :class:`quantlab.analytics.SelectionTests` for a grid of the same
    strategy on the same sample. Pass None for anything not run.
    """
    items: list[EvidenceItem] = []
    add = items.append

    add(EvidenceItem("accounting", "Does the accounting reconcile?", "pass" if accounting_valid else "fail",
                     "7 of 7 checks" if accounting_valid else "failed", "All seven identities",
                     "Cash, shares, fees, P&L attribution and timing are re-derived from the ledger.", "Backtest"))
    if synthetic:
        add(EvidenceItem("data", "Is this real market data?", "caution", "simulated", "Real prices",
                         "Synthetic demo data: results describe a simulated process, not any security.", "Data"))
    else:
        add(EvidenceItem("data", "Did the data pass every quality check?",
                         "pass" if data_warnings == 0 else "caution", f"{data_warnings} warning(s)", "No warnings",
                         "Warnings mean rows were removed or something looked suspicious; see the report.", "Data"))

    n = int(report.value("observations") or 0)
    min_n, why_n = THRESHOLDS["min_sessions"]
    add(EvidenceItem("sample", "Is the sample long enough?", "pass" if n >= min_n else "caution",
                     f"{n:,} sessions", f"≥ {min_n:,.0f} sessions", why_n, "Overview"))

    sharpe, lo = report.value("sharpe"), report.value("sharpe_ci_low")
    if sharpe is None or sharpe <= 0:
        status: Status = "fail"
    else:
        status = "pass" if lo is not None and lo > 0 else "caution"
    add(EvidenceItem("sharpe_ci", "Is the Sharpe ratio distinguishable from zero?", status,
                     f"{_fmt(sharpe, '.2f')} (95% CI from {_fmt(lo, '.2f')})", "CI lower bound > 0",
                     "The interval allows for sample length, skew and fat tails but not for how many "
                     "configurations were tried.", "Analytics"))

    add(EvidenceItem("prespecified", "Were the parameters fixed before seeing results?",
                     "pass" if parameters_prespecified else "caution",
                     "yes" if parameters_prespecified else "possibly tuned", "Defaults or declared in advance",
                     "Parameters chosen after looking at this sample are in-sample and overstate performance.",
                     "Sidebar"))

    dsr_bar, why_dsr = THRESHOLDS["dsr"]
    if n_trials < 2 or dsr is None:
        add(EvidenceItem("dsr", "Does it survive the number of configurations tried?", "not_run",
                         f"{n_trials} trial(s)", f"Deflated Sharpe ≥ {dsr_bar:.0%}",
                         "Needs at least two configurations tried on this sample.", "Overview"))
    else:
        add(EvidenceItem("dsr", "Does it survive the number of configurations tried?",
                         "pass" if dsr >= dsr_bar else ("caution" if dsr >= 0.5 else "fail"),
                         f"{dsr:.0%} over {n_trials} trials", f"≥ {dsr_bar:.0%}", why_dsr, "Overview"))

    cagr = report.value("cagr")
    if exposure_matched_cagr is None or cagr is None:
        add(EvidenceItem("exposure", "Does it beat a portfolio with the same average exposure?", "not_run", "—",
                         "Higher CAGR", "Separates timing from simply holding less market risk.", "Analytics"))
    else:
        add(EvidenceItem("exposure", "Does it beat a portfolio with the same average exposure?",
                         "pass" if cagr > exposure_matched_cagr else "fail",
                         f"{cagr:+.2%} vs {exposure_matched_cagr:+.2%}", "Higher CAGR",
                         "Lower drawdowns from being in cash are not timing skill.", "Analytics → Robustness"))

    perm_bar, why_perm = THRESHOLDS["permutation_p"]
    add(EvidenceItem("permutation", "Does the timing itself matter?",
                     "not_run" if permutation_p is None else ("pass" if permutation_p <= perm_bar else "fail"),
                     "—" if permutation_p is None else f"p = {permutation_p:.3f}", f"p ≤ {perm_bar:.2f}", why_perm,
                     "Analytics → Robustness"))

    spa_bar, why_spa = THRESHOLDS["spa_p"]
    pbo_bar, why_pbo = THRESHOLDS["pbo"]
    if selection is None:
        for key, question, bar in (("spa", "Does the best grid point survive the search?", f"SPA p ≤ {spa_bar}"),
                                   ("pbo", "Is choosing among grid points informative?", f"PBO < {pbo_bar:.0%}")):
            add(EvidenceItem(key, question, "not_run", "—", bar, "Run a parameter-sensitivity grid for this strategy.",
                             "Strategy Lab → Parameter sensitivity"))
    else:
        p = selection.spa_vs_benchmark.p_value
        add(EvidenceItem("spa", "Does the best grid point survive the search?",
                         "pass" if p <= spa_bar else "fail", f"p = {p:.3f} vs benchmark", f"p ≤ {spa_bar}", why_spa,
                         "Strategy Lab → Parameter sensitivity"))
        pbo = selection.pbo.pbo if selection.pbo is not None else None
        add(EvidenceItem("pbo", "Is choosing among grid points informative?",
                         "not_run" if pbo is None else ("pass" if pbo < pbo_bar else "fail"),
                         _fmt(pbo, ".0%"), f"< {pbo_bar:.0%}", why_pbo, "Strategy Lab → Parameter sensitivity"))

    ret_bar, why_ret = THRESHOLDS["wf_retention"]
    if walk_forward is None:
        add(EvidenceItem("walk_forward", "Does it hold up out of sample?", "not_run", "—",
                         f"OOS Sharpe > 0, ≥ {ret_bar:.0%} of in-sample",
                         "Run a walk-forward analysis of this strategy on this sample.", "Walk-Forward"))
    else:
        oos, is_mean, oos_mean = (walk_forward.get(k) for k in ("oos_sharpe", "is_mean", "oos_mean"))
        retention = (oos_mean / is_mean) if is_mean and is_mean > 0 and oos_mean is not None else None
        if oos is None or oos <= 0:
            status = "fail"
        elif retention is not None and retention >= ret_bar:
            status = "pass"
        else:
            status = "caution"
        add(EvidenceItem("walk_forward", "Does it hold up out of sample?", status,
                         f"OOS Sharpe {_fmt(oos, '.2f')}, retention {_fmt(retention, '.0%')}",
                         f"OOS Sharpe > 0, ≥ {ret_bar:.0%} of in-sample", why_ret, "Walk-Forward"))

    if regimes is None or regimes.empty:
        add(EvidenceItem("regimes", "Does it work in both up- and down-trends?", "not_run", "—",
                         "Positive Sharpe in both", "Needs 200 sessions of benchmark history.", "Analytics → Robustness"))
    else:
        trend = regimes[regimes["Dimension"] == "Trend"].set_index("Regime")["Strategy Sharpe"]
        values = [trend.get(r) for r in ("Up-trend", "Down-trend")]
        ok = all(v is not None and pd.notna(v) and v > 0 for v in values)
        add(EvidenceItem("regimes", "Does it work in both up- and down-trends?", "pass" if ok else "caution",
                         " / ".join(_fmt(v, ".2f") for v in values) + " (up / down)", "Positive Sharpe in both",
                         "An edge confined to one regime depends on that regime recurring.", "Analytics → Robustness"))
    return items
