"""The Methodology page's claim -> implementation -> test table must stay true.

Every claim must point at code that exists and at tests that exist, so a
renamed function or a deleted test fails CI instead of leaving the page wrong.
That the referenced tests *pass* is checked by the rest of the suite.
"""

from __future__ import annotations

import ast
import importlib
from functools import cache
from pathlib import Path

import pytest

from quantlab.traceability import CLAIMS

ROOT = Path(__file__).resolve().parent.parent


def _resolve(dotted: str):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ModuleNotFoundError(dotted)


@cache
def _test_functions(path: str) -> frozenset[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return frozenset(node.name for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_"))


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.implementation)
def test_implementation_exists(claim):
    assert callable(_resolve(claim.implementation)), claim.implementation


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.implementation)
def test_every_referenced_test_exists(claim):
    assert claim.tests, "a claim without a test is not traceable"
    for ref in claim.tests:
        path, _, name = ref.partition("::")
        assert path.startswith("tests/") and (ROOT / path).is_file(), ref
        assert name in _test_functions(path), f"{ref} does not exist"


def test_claims_are_unique_and_described():
    texts = [c.claim for c in CLAIMS]
    assert len(texts) == len(set(texts))
    assert all(c.area and c.claim.endswith(".") for c in CLAIMS)
