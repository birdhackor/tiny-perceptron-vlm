"""Version-integrity fixtures only; these are not fabricated reader-pass reports."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "readability_audit", Path(__file__).resolve().parents[1] / "docs/review-tools/audit_readability_round.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize("tamper", ["unit", "introduction", "figure"])
def test_same_checkpoint_count_cannot_hide_changed_reading_material(tamper):
    # The old section-hash/count-only validation missed these three changes.
    body, intro, units = b"fixture body\n", b"fixture introduction\n", ["intro unit", "body unit"]
    figures = {"fixture.svg": "a" * 64}
    header = {
        "reviewer_task": "/root/test_fixture",
        "source": "fixture.md#1.1",
        "source_sha256": AUDIT.digest(body),
        "intro_sha256": AUDIT.digest(intro),
        "figure_sha256": figures.copy(),
    }
    notes = [
        {"unit_index": i, "unit_sha256": AUDIT.digest(unit.encode()), "understanding": "fixture note"}
        for i, unit in enumerate(units)
    ]
    arguments = {
        "task": header["reviewer_task"],
        "source": header["source"],
        "body": body,
        "intro": intro,
        "figures": figures,
        "units": units,
        "fields": ("understanding",),
    }
    assert AUDIT.trace_errors(header, notes, **arguments) == []
    altered_header, altered_notes = deepcopy(header), deepcopy(notes)
    if tamper == "unit":
        altered_notes[1]["unit_sha256"] = "b" * 64
    elif tamper == "introduction":
        altered_header["intro_sha256"] = "b" * 64
    else:
        altered_header["figure_sha256"]["fixture.svg"] = "b" * 64
    errors = AUDIT.trace_errors(altered_header, altered_notes, **arguments)
    assert errors, "Equal section hash and checkpoint count must not conceal stale material"
