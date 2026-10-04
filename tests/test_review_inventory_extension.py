"""New lessons require explicit scope while every original review remains required."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "docs/review-tools/check_review_round.py"
SPEC = importlib.util.spec_from_file_location("review_round_extension", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extension_keeps_original_inventory():
    baseline = {"baseline_revision": "old", "records": {"B.4": {}}}
    additions = {
        "baseline_revision": "old",
        "sections": {"B.5": {"source": "course/chapters/0B.md#B.5", "reason": "Tool choice"}},
    }
    assert MODULE.expected_inventory(baseline, additions) == {"B.4", "B.5"}


@pytest.mark.parametrize(
    "additions",
    [
        {"baseline_revision": "different", "sections": {}},
        {"baseline_revision": "old", "sections": {"B.4": {"source": "x#B.4", "reason": "Replace"}}},
        {"baseline_revision": "old", "sections": {"B.5": {"source": "x#B.6", "reason": "Mismatch"}}},
        {"baseline_revision": "old", "sections": {"B.5": {"source": "x#B.5", "reason": ""}}},
    ],
)
def test_extension_rejects_replaced_or_ambiguous_scope(additions):
    with pytest.raises(ValueError):
        MODULE.expected_inventory({"baseline_revision": "old", "records": {"B.4": {}}}, additions)
