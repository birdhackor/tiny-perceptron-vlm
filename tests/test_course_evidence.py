"""Public evidence must not include private sample text or mistake a smoke run for training."""

import json

import pytest

from scripts import ingest_course_result


def test_private_policy_redaction_keeps_only_aggregate_metrics():
    public = ingest_course_result.redact(
        {
            "toy": {"samples": [{"prompt": "visible", "generated": "visible answer"}]},
            "pilot": {
                "private_only": True,
                "evaluation": {
                    "test": {"nll": 1.25, "samples": [{"prompt": "private prompt", "generated": "private answer"}]}
                },
                "data": {"train": {"records": 12}},
            },
        }
    )
    assert public["toy"]["samples"][0]["prompt"] == "visible"
    assert public["pilot"]["evaluation"]["test"] == {"nll": 1.25}
    assert public["pilot"]["data"]["train"]["records"] == 12
    assert "private prompt" not in json.dumps(public)
    assert "private answer" not in json.dumps(public)


def test_smoke_cannot_replace_completed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_course_result, "ROOT", tmp_path)
    existing = tmp_path / "docs/course-experiments/results/sft.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"evidence_status":"complete_run"}')
    smoke = tmp_path / "smoke.json"
    smoke.write_text('{"experiment_id":"sft","evidence_status":"interface_smoke_only"}')
    with pytest.raises(ValueError, match="complete formal run"):
        ingest_course_result.ingest(smoke)
    assert json.loads(existing.read_text()) == {"evidence_status": "complete_run"}


def test_older_backup_flag_keeps_observation_and_qualifies_resume_scope():
    result = {"experiment_id": "simple_models", "hf": {"private_full_training_state": True}}
    corrected = ingest_course_result.clarify_backup_scope(result)
    assert corrected["hf"]["private_full_training_state"] is True
    assert "weights/vocabulary only" in corrected["hf"]["legacy_backup_flag_scope"]
    assert "not as a per-format resume check" in corrected["hf"]["legacy_backup_flag_scope"]
    current = {"hf": {"private_full_experiment_output": True}}
    assert ingest_course_result.clarify_backup_scope(current) == current
