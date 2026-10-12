"""Documentation must not drift from the code (claims the audit found to be outdated stay fixed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.strategies import STRATEGY_CLASSES

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
DEPLOYMENT = (ROOT / "DEPLOYMENT.md").read_text(encoding="utf-8")
WORDS = {7: "seven"}


def test_strategy_count_in_the_architecture_tree():
    assert f"{WORDS[len(STRATEGY_CLASSES)]} strategies" in README


@pytest.mark.parametrize("phrase", [
    "only random number", "about 15 seconds", "five strategies", "lower bound on true", "AUC 0.58–0.64",
    "used as provided; the app cannot", "shown on every page", "every rebalance band",
])
def test_outdated_readme_claims_do_not_return(phrase):
    assert phrase not in README


def test_deployment_does_not_understate_runtimes():
    assert "take several seconds" not in DEPLOYMENT and "bounded caches," not in DEPLOYMENT


def test_readme_test_count_matches_the_suite():
    import re
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         capture_output=True, text=True, cwd=ROOT, timeout=300).stdout
    collected = int(re.search(r"(\d+) tests? collected", out).group(1))
    claimed = {int(n) for n in re.findall(r"(\d+) (?:automated )?tests", README)}
    assert claimed == {collected}, (claimed, collected)
