#!/usr/bin/env python
"""Re-run a backtest from a Quant Lab Pro run manifest and verify the results.

Usage (from the project root, with the virtual environment active):

    # Re-download the data from the provider recorded in the manifest
    python scripts/reproduce_backtest.py --manifest run_manifest.json

    # Exact offline reproduction using the price CSV exported with the run
    python scripts/reproduce_backtest.py --manifest run_manifest.json --prices prices.csv

The script prints whether the data fingerprint, methodology version and each
headline metric match the recorded values. Exit code 0 means everything matched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from quantlab.data import (  # noqa: E402
    AdjustmentMode,
    DataProviderError,
    YahooFinanceProvider,
    fingerprint_prices,
    load_market_data,
)
from quantlab.reporting import reproduce_from_manifest  # noqa: E402
from quantlab.validation import clean_price_frame  # noqa: E402


def _load_prices(manifest: dict, prices_path: str | None) -> tuple[pd.DataFrame, str]:
    data = manifest["data"]
    if prices_path:
        frame = pd.read_csv(prices_path, index_col=0, parse_dates=[0], float_precision="round_trip")
        clean, report = clean_price_frame(frame)
        if report.is_blocking:
            raise SystemExit(f"Price file failed validation: {[i.message for i in report.errors]}")
        return clean, fingerprint_prices(clean)
    if data["provider"] != "yahoo":
        raise SystemExit("This manifest used uploaded data; pass the exported CSV with --prices.")
    dataset = load_market_data(
        data["ticker"],
        data["data_start"],
        manifest["evaluation"]["end"],
        YahooFinanceProvider(),
        adjustment=AdjustmentMode(data["price_adjustment"]) if data["price_adjustment"] in {m.value for m in AdjustmentMode} else AdjustmentMode.TOTAL_RETURN,
    )
    return dataset.prices, dataset.fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to a run manifest JSON file.")
    parser.add_argument("--prices", help="Optional price CSV exported with the run (offline, exact).")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    try:
        prices, fingerprint = _load_prices(manifest, args.prices)
    except DataProviderError as exc:
        print(f"Data unavailable: {exc.user_message}")
        return 2

    outcome = reproduce_from_manifest(manifest, prices, fingerprint)
    print(f"Strategy:            {manifest['strategy']['key']} {manifest['strategy']['params']}")
    print(f"Evaluation period:   {manifest['evaluation']['start']} to {manifest['evaluation']['end']}")
    print(f"Data fingerprint:    {'MATCH' if outcome['fingerprint_match'] else 'DIFFERENT (provider revised the data?)'}")
    print(f"Methodology version: {'MATCH' if outcome['methodology_match'] else 'DIFFERENT'}")
    print(f"Accounting checks:   {'PASSED' if outcome['accounting_valid'] else 'FAILED'}")
    print()
    print(f"{'metric':<16}{'recorded':>18}{'reproduced':>18}  match")
    for row in outcome["metrics"]:
        rec = "n/a" if row["recorded"] is None else f"{row['recorded']:.10g}"
        rep = "n/a" if row["reproduced"] is None else f"{row['reproduced']:.10g}"
        print(f"{row['metric']:<16}{rec:>18}{rep:>18}  {'yes' if row['match'] else 'NO'}")
    ok = outcome["fingerprint_match"] and outcome["all_metrics_match"] and outcome["accounting_valid"]
    print("\nResult:", "REPRODUCED EXACTLY" if ok else "DIFFERENCES FOUND (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
