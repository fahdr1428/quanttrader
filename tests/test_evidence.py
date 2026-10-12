"""Evidence checklist: thresholds, 'not run' handling, and the Overview integration."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quantlab.evidence import THRESHOLD_DISPLAY, THRESHOLDS, evidence_checklist


class _Report:
    def __init__(self, **values):
        self._values = values

    def value(self, key):
        return self._values.get(key)


def _items(**overrides):
    kwargs = dict(report=_Report(observations=1000, sharpe=0.8, sharpe_ci_low=0.1, cagr=0.08), accounting_valid=True,
                  data_warnings=0, synthetic=False, parameters_prespecified=True, n_trials=1, dsr=None)
    kwargs.update(overrides)
    return {i.key: i for i in evidence_checklist(**kwargs)}


def test_unrun_analyses_are_never_reported_as_passed():
    items = _items()
    for key in ("dsr", "exposure", "permutation", "spa", "pbo", "walk_forward", "regimes"):
        assert items[key].status == "not_run", key


def test_thresholds_decide_pass_caution_and_fail():
    items = _items(report=_Report(observations=500, sharpe=0.4, sharpe_ci_low=-0.2, cagr=0.03),
                   accounting_valid=False, data_warnings=2, parameters_prespecified=False,
                   n_trials=12, dsr=0.4, exposure_matched_cagr=0.05, permutation_p=0.30)
    assert items["accounting"].status == "fail"
    assert items["data"].status == "caution"
    assert items["sample"].status == "caution"  # fewer than THRESHOLDS["min_sessions"]
    assert items["sharpe_ci"].status == "caution"  # positive Sharpe, interval includes zero
    assert items["prespecified"].status == "caution"
    assert items["dsr"].status == "fail"
    assert items["exposure"].status == "fail" and items["permutation"].status == "fail"
    good = _items(n_trials=5, dsr=0.97, exposure_matched_cagr=0.02, permutation_p=0.04)
    assert {good[k].status for k in ("sample", "sharpe_ci", "dsr", "exposure", "permutation")} == {"pass"}
    assert THRESHOLDS["min_sessions"][0] == 756


def test_grid_walk_forward_and_regime_checks():
    selection = SimpleNamespace(spa_vs_benchmark=SimpleNamespace(p_value=0.2), pbo=SimpleNamespace(pbo=0.3))
    regimes = pd.DataFrame({"Dimension": ["Trend", "Trend"], "Regime": ["Up-trend", "Down-trend"],
                            "Strategy Sharpe": [1.0, -0.2]})
    items = _items(selection=selection, regimes=regimes,
                   walk_forward={"oos_sharpe": 0.4, "is_mean": 1.0, "oos_mean": 0.3})
    assert items["spa"].status == "fail" and items["pbo"].status == "pass"
    assert items["walk_forward"].status == "caution"  # positive OOS but only 30% retention
    assert items["regimes"].status == "caution"
    items = _items(walk_forward={"oos_sharpe": -0.1, "is_mean": 1.0, "oos_mean": -0.1})
    assert items["walk_forward"].status == "fail"


def test_synthetic_data_is_never_a_pass():
    assert _items(synthetic=True)["data"].status == "caution"


def test_overview_checklist_updates_after_a_walk_forward_run():
    from tests.test_app_smoke import _app

    at = _app()
    html = " ".join(m.value for m in at.markdown)
    assert "Evidence checklist" in html and "NOT RUN" in html
    at.switch_page("pages/5_Walk_Forward.py").run()
    at.button(key="wf_run").click().run()
    at.switch_page("pages/1_Research_Overview.py").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Run a walk-forward analysis of this strategy" not in html  # the row now reports a result


def test_every_threshold_has_a_readable_name_and_comparison():
    assert THRESHOLD_DISPLAY.keys() == THRESHOLDS.keys()
    for key, (value, _) in THRESHOLDS.items():
        name, comparison = THRESHOLD_DISPLAY[key]
        assert name and "_" not in name and comparison.format(value)
