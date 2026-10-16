"""用虛構預測核對Auto分母與凍結，不執行任何正式成品test。"""

import json

import pytest
import torch

from tiny_perceptron.lesson_products.evaluation import auto_quality, c_report, chosen_meanings, frozen_guard


def test_C_rejecting_difficult_complete_case_is_an_end_to_end_failure():
    rows = [
        {"task": "sum", "state": "三數加法;x=0;y=1;z=2", "gold": 3},
        {"task": "sum", "state": "三數加法;x=1;y=2;z=3", "gold": 6},
    ]
    predictions = {1: torch.tensor([3, 0]), 3: torch.tensor([3, 6]), 5: torch.tensor([3, 6])}
    result = auto_quality(rows, predictions, ["sum", "yes", "sum", "no"], {"version": 1, "sum_depth": 3})
    assert result["accuracy_end_to_end"] == 0.5
    assert result["conditional_accuracy"] == 1
    assert result["coverage"] == 0.5
    assert result["count_complete_eligible"] == 2
    assert result["mean_actual_b_loops_all_complete"] == 1.5
    assert result["mean_b_loops_accepted"] == 3
    assert result["c_calls_per_input"] == 2


def test_semantic_ties_and_per_question_calibration_metrics_ignore_padding():
    rows = [
        {"question": "choice", "candidates": ["none", "card", "app"], "gold": "app"},
        {"question": "info", "candidates": ["no", "yes"], "gold": "yes"},
    ]
    logits = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, float("-inf")]])
    assert chosen_meanings(logits, rows) == ["app", "yes"]
    report = c_report(logits, rows, {"choice": 1, "info": 0.5, "rubric": 1})
    assert report["choice"]["after"]["accuracy"] == 1
    assert report["info"]["after"]["cross_entropy"] < report["info"]["before"]["cross_entropy"]
    assert report["info"]["after"]["brier"] < report["info"]["before"]["brier"]


def test_final_guard_rejects_data_drift_before_loading_any_model(tmp_path):
    (tmp_path / "manifest.json").write_text('{"fictional":true}')
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"test_outcomes_observed": False, "data_manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="相同資料"):
        frozen_guard(freeze, tmp_path, tmp_path, tmp_path, {})
