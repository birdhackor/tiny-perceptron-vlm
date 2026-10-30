"""評測管線反例；全部是人工fixture，沒有訓練模型或預測真正final。"""

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.omni_revision import core_evaluation as measured
from scripts.omni_revision import core_final as final
from scripts.omni_revision import core_report as report
from tiny_perceptron.omni_products.data import av_pair, split_a
from tiny_perceptron.omni_products.inference import SOUND_MAP
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file

ROOT = Path(__file__).resolve().parents[1]


def test_final_source_watchlist_includes_real_old_A_transitive_modules():
    sources = final.evaluation_sources()
    for module in ("model", "modern", "multimodal", "data", "attention", "__init__"):
        assert f"tiny_perceptron/{module}.py" in sources
    for name in ("scripts/omni_revision/formal_local.py", "scripts/omni_revision/__init__.py"):
        assert name in sources
    assert len(sources) == 57
    assert all(file_sha(ROOT / name) == digest for name, digest in sources.items())


def prediction(question):
    ids = [row["id"] for row in question["candidates"]] + (["none"] if question["kind"] == "choice" else [])
    return {
        "ids": ids,
        "logits": [8.0 if x == question["gold"] else 0.0 for x in ids],
        "answer": question["gold"],
        "gold": question["gold"],
    }


@pytest.fixture(scope="module")
def complete_fixture():
    # 只借用公開的家族ID／gold檢查schema；以下答案是人工填入，絕非模型成績。
    data = json.loads((ROOT / "docs/omni-products/v1/data-definition.json").read_bytes())
    metrics = json.loads((ROOT / "docs/omni-products/v1/metrics.json").read_bytes())
    av_rows = [
        {
            "family": row["family"],
            "panel": row["panel"],
            "answers": row["labels"],
            "gold": row["labels"],
            "paired_correct": True,
            "remove_image_answer": 2,
            "remove_audio_answer": 2,
            "waveform_swap_answers": [0, 1],
            "waveform_swap_gold": [0, 1],
        }
        for row in data["AV"]["final"]
    ]
    av = {
        "primary": measured.proportion(200, 200),
        "panels": {str(i): measured.proportion(50, 50) for i in range(4)},
        "remove_image": measured.proportion(200, 200),
        "remove_audio": measured.proportion(200, 200),
        "same_metadata_waveform_swap": {
            "paired": measured.proportion(200, 200),
            "observations": 400,
            "correct_observations": 400,
        },
        "records": av_rows,
    }
    models = {f"av-{kind}-s{seed}": copy.deepcopy(av) for kind in ("additive", "time-rope") for seed in (11, 29, 47)}
    sound = {
        "families": 200,
        "mean_active_reference_snr_db": 12.0,
        "invalid_snr_families": 0,
        "gold_program": measured.proportion(200, 200),
        "legal_output": measured.proportion(200, 200),
        "chunk_controls": [
            {"family_index": i, "chunks_bit_exact": True, "future_code_prefix_bit_exact": True} for i in range(10)
        ],
        "rows": [
            {
                "family_index": i,
                "program": program,
                "active_reference_snr_db": 12.0,
                "generation": {"program_success": True, "legal": True},
            }
            for i, program in enumerate(data["sound"]["final"])
        ],
    }
    chain_keys = {
        "primary": "chain_correct",
        "A_semantic": "semantic_correct",
        "mapping": "mapping_correct",
        "actual_condition_Talker": "actual_condition_correct",
        "gold_semantic_condition_Talker": "gold_semantic_condition_correct",
        "legal": "legal",
    }
    chain = {name: measured.proportion(200, 200) for name in chain_keys}
    chain.update(
        mapping_attempts=200,
        distinct_semantic_sound_programs=2,
        all_failures_remain_in_denominator=True,
        rows=[
            {"family": row["family"], "mapping_attempted": True, **{key: True for key in chain_keys.values()}}
            for row in data["AV"]["final"]
        ],
    )
    families = [
        {
            **{
                key: row[key]
                for key in ("family", "source", "panel", "world_supported", "choice_information_sufficient")
            },
            "tasks": {kind: prediction(row["questions"][kind]) for kind in ("choice", "info", "rubric")},
            "choice_info": prediction(row["questions"]["choice_info"]),
        }
        for row in data["C"]["panels"]["final"]
    ]
    diagnostics = []
    for row in data["C"]["panels"]["final"]:
        for item in row["candidate_count_diagnostics"]:
            total = item["total_choices_including_none"]
            q = {
                "kind": "choice",
                "gold": item["gold"],
                "candidates": data["C"]["diagnostic_candidate_pool"][: total - 1],
            }
            diagnostics.append({"family": row["family"], "total_choices_including_none": total, **prediction(q)})
    invariant_keys = (
        "shared_recompute_logits_max_abs",
        "shared_recompute_hidden_max_abs",
        "shared_recompute_memory_max_abs",
        "Q1_Q2_Q1_logits_max_abs",
        "permutation_probability_max_abs",
    )
    invariants = {
        "development_families": 8,
        "atol": 1e-5,
        "all_passed": True,
        "rows": [
            {"family": row["family"], "permutation_answer_same": True, **dict.fromkeys(invariant_keys, 0.0)}
            for row in data["C"]["panels"]["dev"][:8]
        ],
    }
    failed = {"error": {"error_type": "ExplicitFixture", "message": "artificial failure, not a model run"}}
    old_rows = [
        {
            "task": row["task"],
            "family": row["family"],
            "source_record": row["record"],
            "old_correct": False,
            "wrapper_correct": False,
            "old": failed,
            "wrapper": failed,
        }
        for row in data["old_A_retention"]
    ]
    old = {
        "observations": 258,
        "parent_families": 180,
        "rows": old_rows,
        "tasks": {
            task: {"families": n, "old_correct": 0, "wrapper_correct": 0, "difference_percentage_points": 0.0}
            for task, n in Counter(row["task"] for row in old_rows).items()
        },
        "counterfactual_diagnostics": {
            "observations": 24,
            "correct": 0,
            "actual_executor_used": False,
            "rows": [
                {"family": row["family"], "record": row["record"], "correct": False, **failed}
                for row in data["counterfactual_tool_diagnostics"]
            ],
        },
    }
    import itertools

    b = {
        "population_families": 84,
        "frozen_depth": 1,
        "old_correct": 0,
        "wrapper_correct": 0,
        "difference_percentage_points": 0.0,
        "rows": [
            {"parent": numbers, "old_correct": False, "wrapper_correct": False, "old": failed, "wrapper": failed}
            for numbers in data["B_retention"]["values"]
        ],
        "ordered_diagnostic": {
            "observations": 343,
            "parent_families": 84,
            "correct": 0,
            "not_independent343": True,
            "rows": [
                {"numbers": list(numbers), "correct": False, **failed}
                for numbers in itertools.product(range(7), repeat=3)
            ],
        },
    }
    auto = {
        "families": 100,
        "counts": metrics["Auto"]["counts"],
        **{
            key: measured.proportion(0, 100)
            for key in ("route_accuracy", "gold_route_end_to_end", "real_route_end_to_end")
        },
        "rows": [
            {
                "family": row["family"],
                "gold_route": row["gold_route"],
                "gold_route_end_to_end_correct": False,
                "real_route_end_to_end_correct": False,
                "route_correct": False,
                "gold_route_error": failed,
                "real_route_error": failed,
            }
            for row in data["Auto"]["families"]
        ],
    }
    results = {
        "av": {"models": models, "paired_family_comparison": measured.paired_AV_comparison(models)},
        "sound": {
            "codec_and_gold_program": sound,
            "actual_A_chain": chain,
            "actual_playable_timing": [{"family_index": i, **failed} for i in range(10)],
        },
        "choice": {
            "primary": measured.evaluate_c(families, 1.0, 0.5),
            "data_control_raw_only": measured.evaluate_c(families, 1.0, None),
            "candidate_count_secondary": {
                "parent_families": 60,
                "secondary_observations": 420,
                "independent_n_increase": False,
                "rows": diagnostics,
            },
            "development_invariants": invariants,
        },
        "integration": {"old_A_retention": old, "B_retention": b, "auto": auto},
    }
    selection = {"selected": {"av": "av-time-rope-s29"}, "temperature": 1.0, "threshold": 0.5}
    return results, selection, metrics, data


def test_complete_failure_rows_are_measurements_not_automatic_performance_pass(complete_fixture):
    results, selection, metrics, data = copy.deepcopy(complete_fixture)
    sound = results["sound"]["codec_and_gold_program"]
    sound["rows"][0]["generation"]["program_success"] = False
    sound["gold_program"] = measured.proportion(199, 200)
    sound["chunk_controls"][0]["chunks_bit_exact"] = False
    result = report.verdict(results, selection, metrics, data)
    assert result["research_measurements_complete"]
    assert not result["global_targets_all_met"]
    assert not result["promote_finite_sound_module"]
    assert not result["promote_A_AV_sound_extension"]


@pytest.mark.parametrize(
    "missing",
    [
        "old_tasks",
        "raw_metrics",
        "candidate_row",
        "timing_attempt",
        "swap",
        "bootstrap",
        "auto_row",
        "sound_row",
        "chain_stage",
        "control_rubric",
    ],
)
def test_partial_aggregate_cannot_be_declared_complete(complete_fixture, missing):
    values = copy.deepcopy(complete_fixture)
    results = values[0]
    if missing == "old_tasks":
        results["integration"]["old_A_retention"]["tasks"] = {}
    elif missing == "raw_metrics":
        results["choice"]["primary"]["types"]["info"].pop("raw")
    elif missing == "candidate_row":
        results["choice"]["candidate_count_secondary"]["rows"].pop()
    elif missing == "timing_attempt":
        results["sound"]["actual_playable_timing"].pop()
    elif missing == "swap":
        next(iter(results["av"]["models"].values())).pop("same_metadata_waveform_swap")
    elif missing == "bootstrap":
        results["av"].pop("paired_family_comparison")
    elif missing == "auto_row":
        results["integration"]["auto"]["rows"].pop()
    elif missing == "sound_row":
        results["sound"]["codec_and_gold_program"]["rows"].pop()
    elif missing == "chain_stage":
        results["sound"]["actual_A_chain"].pop("actual_condition_Talker")
    elif missing == "control_rubric":
        results["choice"]["data_control_raw_only"]["types"].pop("rubric")
    with pytest.raises((ValueError, KeyError)):
        report.verdict(*values)


def test_bootstrap_uses_same_families_and_reports_each_seed(complete_fixture):
    models = copy.deepcopy(complete_fixture[0]["av"]["models"])
    for seed in (11, 29, 47):
        for row in models[f"av-additive-s{seed}"]["records"]:
            row["paired_correct"] = False
    result = measured.paired_AV_comparison(models)
    assert result["replicates"] == 1000 and result["resampling_seed"] == 20261029
    assert result["families"] == 200
    assert result["three_seed_mean"]["paired_family_bootstrap95"] == [1.0, 1.0]
    assert result["paired_accuracy_three_seed_mean"] == {"additive": 0.0, "time_rope": 1.0}
    models["av-additive-s11"]["records"].append(models["av-additive-s11"]["records"][0])
    with pytest.raises(ValueError):
        measured.paired_AV_comparison(models)


@pytest.mark.parametrize("answer", ["結果是18。", "答案為18", "計算結果：18", "18", "  +18！ "])
def test_auto_calculator_scores_numeric_meaning_and_live_execution(answer):
    record = {"gold_route": "A", "family": "auto-A-calculator-fixture", "parent": ["A", "add", 7, 11], "gold": 18}
    trace = {"executed": True, "tool_call": {"tool": "calculator", "operation": "add", "a": 7, "b": 11}}
    assert measured.score_auto_result(record, {"answer": answer, "tool_trace": trace})["correct"]
    assert not measured.score_auto_result(record, {"answer": answer, "tool_trace": {**trace, "executed": False}})[
        "correct"
    ]
    assert not measured.score_auto_result(record, {"answer": "結果是19。", "tool_trace": trace})["correct"]
    assert not measured.score_auto_result(record, {"answer": "18或19", "tool_trace": trace})["correct"]


def test_mapping_is_scored_against_actual_semantic_not_gold(tmp_path, monkeypatch):
    toy = av_pair(split_a()["dev"][0])[0]
    gold = ("circle", "square")[toy[1]]
    actual = "square" if gold == "circle" else "circle"
    assistant = SimpleNamespace(
        codec=None,
        talker=None,
        video=lambda *args, **kwargs: {
            "answer": actual,
            "sound": {"program": SOUND_MAP[actual], "waveform": torch.zeros(80), "eos": True, "capped": False},
        },
    )
    monkeypatch.setattr(measured, "av_pair", lambda *args: [toy, toy])
    monkeypatch.setattr(
        measured,
        "program_success",
        lambda wave, program: {"success": program == SOUND_MAP[actual], "measured": {"legal": True}},
    )
    monkeypatch.setattr(measured, "waveform_receipt", lambda *args: {"explicit_fixture": True})
    monkeypatch.setattr(measured, "generated_sound", lambda *args: {"program_success": True})
    result = measured.evaluate_chain(
        assistant, [{"family": 1, "variant": 0, "panel": 0, "predeclared_chain_pair_index": 0}], tmp_path
    )
    assert result["mapping"]["correct"] == result["actual_condition_Talker"]["correct"] == 1
    assert result["A_semantic"]["correct"] == result["primary"]["correct"] == 0
    assert result["gold_semantic_condition_Talker"]["correct"] == 1


def test_incomplete_selection_is_rejected_before_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "FROZEN", tmp_path)
    json_file(tmp_path / "model-selection.json", {"all_formal_jobs": {}, "model_files": {}})
    json_file(tmp_path / "formal-recipes.json", {"jobs": {}})
    with pytest.raises(ValueError, match="完整schema"):
        final.checked_selection(tmp_path, file_sha(tmp_path / "model-selection.json"))
    assert not (tmp_path / "final-opening.json").exists()


@pytest.mark.parametrize("field", [None, "origin_part", "completion_part", "opened", "selection", "data", "time"])
def test_part_origin_must_match_opening(tmp_path, field):
    opening = {
        "created_at": "2026-10-30T01:00:00+00:00",
        "model_selection_sha256": "selection",
        "data_definition_sha256": "data",
        "metrics_sha256": "metrics",
        "evaluation_sources_sha256": {"fixture": "sha"},
    }
    origin = {
        "part": "av",
        "created_at": "2026-10-30T01:00:01+00:00",
        "final_opened": True,
        "final_opening_sha256": "opening",
        **{key: value for key, value in opening.items() if key != "created_at"},
    }
    receipt = {
        "part": "av",
        "created_at": "2026-10-30T01:00:02+00:00",
        "measurement_completed": True,
        "final_opening_sha256": "opening",
        "model_selection_sha256": "selection",
    }
    if field == "origin_part":
        origin["part"] = "sound"
    elif field == "completion_part":
        receipt["part"] = "sound"
    elif field == "opened":
        origin["final_opened"] = False
    elif field == "selection":
        origin["model_selection_sha256"] = "other"
    elif field == "data":
        origin["data_definition_sha256"] = "other"
    elif field == "time":
        receipt["created_at"] = "2026-10-30T00:00:02+00:00"
    json_file(tmp_path / "results.json", {"explicit_fixture": True})
    json_file(tmp_path / "part-origin.json", origin)
    receipt.update(
        results_sha256=file_sha(tmp_path / "results.json"), part_origin_sha256=file_sha(tmp_path / "part-origin.json")
    )
    json_file(tmp_path / "part-completion.json", receipt)
    if field is None:
        report.validate_part("av", tmp_path, opening, "opening")
    else:
        with pytest.raises(ValueError):
            report.validate_part("av", tmp_path, opening, "opening")
