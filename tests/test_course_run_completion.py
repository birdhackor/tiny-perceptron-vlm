"""提前停止的正式工作不得取代完整教材證據；縮步通路仍明示為 smoke。"""

import json

import pytest

from scripts import ingest_course_result
from scripts.course_experiments import run, text


@pytest.mark.parametrize("scale,expected", [(1.0, "incomplete_run"), (0.1, "interface_smoke_only")])
def test_partial_schedule_cannot_replace_published_complete_evidence(tmp_path, monkeypatch, scale, expected):
    monkeypatch.setattr(
        text,
        "run_text_foundation",
        lambda _ctx: {
            "runs": [
                {"training": {"requested_steps": 200, "steps": 17, "budget_exhausted": True}},
                {"training": {"requested_steps": 200, "steps": 200, "budget_exhausted": False}},
            ]
        },
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text('{"assets": []}')
    output = tmp_path / "run"
    report = run.execute("text_foundation", "cpu", output, tmp_path, assets, step_scale=scale)
    assert report["evidence_status"] == expected
    assert report["unfinished_schedules"] == [{"path": "results.runs[0].training", "requested_steps": 200, "steps": 17}]
    monkeypatch.setattr(ingest_course_result, "ROOT", tmp_path)
    destination = tmp_path / "docs/course-experiments/results/text_foundation.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("existing complete evidence")
    with pytest.raises(ValueError, match="complete formal run"):
        ingest_course_result.ingest(output / "result.json")
    assert destination.read_text() == "existing complete evidence"
    assert json.loads((output / "result.json").read_text())["evidence_status"] == expected


def test_completed_schedule_keeps_full_evidence_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        text,
        "run_text_foundation",
        lambda _ctx: {"training": {"requested_steps": 200, "steps": 200, "budget_exhausted": False}},
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text('{"assets": []}')
    report = run.execute("text_foundation", "cpu", tmp_path / "run", tmp_path, assets)
    assert report["evidence_status"] == "complete_run"
    assert report["unfinished_schedules"] == []
