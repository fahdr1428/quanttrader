"""Claim -> implementation -> test traceability.

Every methodological claim the app and README make about how results are
produced is listed here with the code that implements it and the tests that
would fail if it stopped being true. ``tests/test_traceability.py`` checks that
every implementation path still resolves and every referenced test still
exists, so the documentation cannot silently drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    area: str
    claim: str
    implementation: str  # dotted path to a module attribute
    tests: tuple[str, ...]  # "tests/file.py::test_name"


CLAIMS: tuple[Claim, ...] = (
    Claim("Timing", "Signals use closes up to session t; orders fill at the open of session t + 1.",
          "quantlab.backtest.simulate",
          ("tests/test_backtest.py::test_fill_uses_next_open_not_signal_close",
           "tests/test_backtest.py::test_reconciliation_detects_lookahead_fill")),
    Claim("Timing", "Changing future prices cannot change any earlier signal, for every strategy.",
          "quantlab.strategies.Strategy.generate_signals",
          ("tests/test_strategies.py::test_changing_future_prices_cannot_change_earlier_signals",
           "tests/test_backtest.py::test_future_prices_cannot_change_past_decisions")),
    Claim("Timing", "With a strategy's declared history, loading more history changes no signal (rule-based "
          "strategies; the ML refit schedule is anchored to the data start).",
          "quantlab.strategies.Strategy.history_sessions",
          ("tests/test_model_regressions.py::test_results_do_not_depend_on_how_much_history_was_loaded",
           "tests/test_model_regressions.py::test_ml_declares_that_its_results_depend_on_the_data_start")),
    Claim("Timing", "Data after the evaluation end is removed before signals are computed.",
          "quantlab.backtest.run_backtest", ("tests/test_backtest.py::test_data_after_evaluation_end_is_ignored",)),
    Claim("Accounting", "Seven accounting identities are checked after every backtest.",
          "quantlab.backtest.reconciliation_report",
          ("tests/test_backtest.py::test_reconciliation_passes_for_all_strategies",
           "tests/test_backtest.py::test_reconciliation_detects_corrupted_equity",
           "tests/test_engine_invariants.py::test_accounting_invariants_hold_in_random_scenarios")),
    Claim("Accounting", "Fees and slippage are proportional to traded notional and never increase value.",
          "quantlab.execution.ExecutionModel",
          ("tests/test_backtest.py::test_round_trip_with_costs_matches_formula",
           "tests/test_engine_invariants.py::test_costs_never_increase_final_value")),
    Claim("Accounting", "Round-trip returns are measured on the capital committed, counting recycled money once.",
          "quantlab.backtest.build_round_trips",
          ("tests/test_engine_regressions.py::test_rebalancing_does_not_dilute_round_trip_returns",
           "tests/test_fractional.py::test_episode_round_trip_aggregates_partial_fills")),
    Claim("Accounting", "Whole-share sizing never overspends cash and always terminates, even above 2^53 shares.",
          "quantlab.execution.ExecutionModel.buy_quantity",
          ("tests/test_engine_regressions.py::test_whole_share_sizing_terminates_beyond_float_integer_precision",)),
    Claim("Data", "Missing or invalid prices are removed and reported, never filled in.",
          "quantlab.validation.clean_price_frame",
          ("tests/test_validation.py::test_missing_prices_are_removed_and_reported_never_filled",
           "tests/test_validation.py::test_non_positive_prices_are_removed")),
    Claim("Data", "Total-return prices scale open, high, low and close by Adj Close / Close.",
          "quantlab.data.apply_adjustment",
          ("tests/test_data.py::test_total_return_adjustment_scales_ohlc_by_adj_factor",
           "tests/test_data_regressions.py::test_unadjusted_adj_close_with_reported_dividends_is_flagged")),
    Claim("Data", "CSV date formats are detected from the whole column; ambiguous day/month order is refused.",
          "quantlab.data.parse_price_csv_detailed",
          ("tests/test_data.py::test_csv_date_formats_are_detected_from_the_whole_column",
           "tests/test_data.py::test_ambiguous_day_month_order_is_refused")),
    Claim("Data", "Weekly, intraday and seven-day calendars are refused rather than annualised at 252.",
          "quantlab.validation.check_sampling_frequency",
          ("tests/test_data.py::test_non_daily_calendars_are_refused",)),
    Claim("Data", "Today's unfinished bar is dropped while the exchange session is open.",
          "quantlab.data.YahooFinanceProvider",
          ("tests/test_data_regressions.py::test_unfinished_session_is_dropped_during_trading_hours_and_disclosed",)),
    Claim("Metrics", "Sharpe, PSR, DSR and trial Sharpes all use returns in excess of the risk-free rate.",
          "quantlab.research.probabilistic_sharpe_ratio",
          ("tests/test_research.py::test_sharpe_inference_uses_excess_returns",
           "tests/test_metrics.py::test_inference_metrics_are_consistent_with_the_headline_sharpe")),
    Claim("Metrics", "The Sharpe standard error allows for skew and fat tails (Lo 2002; Mertens 2002).",
          "quantlab.inference.sharpe_inference",
          ("tests/test_inference.py::test_sharpe_standard_error_matches_monte_carlo",)),
    Claim("Metrics", "Skew and kurtosis may widen the Sharpe interval but never narrow it, so PSR keeps its size "
          "for rarely invested strategies; few sessions with market exposure are flagged.",
          "quantlab.inference.sharpe_variance_term",
          ("tests/test_inference.py::test_psr_holds_its_size_for_a_rarely_invested_strategy",
           "tests/test_metrics.py::test_sharpe_inference_is_flagged_when_few_sessions_carry_market_risk")),
    Claim("Metrics", "HAC (Newey-West) standard errors account for autocorrelation.",
          "quantlab.inference.newey_west_variance_of_mean",
          ("tests/test_inference.py::test_newey_west_matches_brute_force_and_default_lags",
           "tests/test_inference.py::test_hac_standard_error_exceeds_iid_on_autocorrelated_data")),
    Claim("Multiple testing", "The deflated Sharpe ratio penalises the number of configurations tried.",
          "quantlab.research.deflated_sharpe_ratio",
          ("tests/test_research.py::test_deflated_sharpe_penalises_many_trials",)),
    Claim("Multiple testing", "Sensitivity grids are counted as trials with all of their parameters; a setting "
          "that leaves the returns unchanged is not a new trial.",
          "ui.state.register_trials_bulk",
          ("tests/test_app_smoke.py::test_sensitivity_trials_include_non_grid_parameters",
           "tests/test_app_smoke.py::test_settings_that_leave_returns_unchanged_add_no_grid_or_walk_forward_trials")),
    Claim("Multiple testing", "PBO is about 0.5 on pure noise for any grid size and near 0 for one dominant "
          "configuration.",
          "quantlab.overfitting.probability_of_backtest_overfitting",
          ("tests/test_overfitting.py::test_pbo_of_pure_noise_is_near_one_half",
           "tests/test_overfitting.py::test_no_information_baseline_is_one_half_for_odd_grids",
           "tests/test_overfitting.py::test_dominant_configuration_gives_low_pbo")),
    Claim("Multiple testing", "SPA and the Reality Check hold their size on null grids and detect planted edges.",
          "quantlab.significance.hansen_spa",
          ("tests/test_significance.py::test_size_under_the_null_is_close_to_nominal",
           "tests/test_significance.py::test_power_against_one_planted_edge")),
    Claim("Multiple testing", "Romano-Wolf p-values control family-wise errors across the grid.",
          "quantlab.significance.romano_wolf_stepdown",
          ("tests/test_significance.py::test_romano_wolf_controls_the_nulls_while_finding_two_edges",)),
    Claim("Robustness", "The exposure-matched benchmark carries the strategy's average exposure and reconciles.",
          "quantlab.nulls.exposure_matched_benchmark",
          ("tests/test_nulls.py::test_exposure_matched_benchmark_matches_exposure_and_reconciles",)),
    Claim("Robustness", "The timing permutation test is calibrated under the null.",
          "quantlab.nulls.timing_permutation_test", ("tests/test_nulls.py::test_calibrated_under_the_null",)),
    Claim("Robustness", "Regime labels use only information known before each return.",
          "quantlab.robustness.regime_performance",
          ("tests/test_robustness.py::test_regimes_use_information_known_before_each_return",)),
    Claim("Out of sample", "Walk-forward parameter selection never sees data after the training end.",
          "quantlab.research.walk_forward",
          ("tests/test_research.py::test_selection_ignores_data_after_training_end",
           "tests/test_research.py::test_out_of_sample_equity_does_not_depend_on_later_prices")),
    Claim("Machine learning", "Each refit trains only on outcomes already known (purged labels).",
          "quantlab.ml.walk_forward_probabilities",
          ("tests/test_models.py::test_every_refit_trains_only_on_known_labels",
           "tests/test_models.py::test_predictions_do_not_change_when_unknown_outcomes_change")),
    Claim("Machine learning", "On pure noise the model shows no positive skill, averaged over seeds.",
          "quantlab.ml.classification_report", ("tests/test_models.py::test_no_skill_on_a_random_walk",)),
    Claim("Reproducibility", "A run manifest reproduces every headline number from the exported prices.",
          "quantlab.reporting.reproduce_from_manifest",
          ("tests/test_reporting.py::test_manifest_reproduces_exactly_from_exported_prices",
           "tests/test_reporting.py::test_synthetic_run_reproduces_offline_from_its_manifest")),
    Claim("Honesty", "The evidence checklist never reports an analysis that was not run as passed.",
          "quantlab.evidence.evidence_checklist",
          ("tests/test_evidence.py::test_unrun_analyses_are_never_reported_as_passed",)),
    Claim("Privacy", "Free text that looks like a credential is refused and records are session-private.",
          "quantlab.experiments.ExperimentStore",
          ("tests/test_experiments.py::test_credentials_are_refused",
           "tests/test_experiments.py::test_sessions_are_isolated")),
)
