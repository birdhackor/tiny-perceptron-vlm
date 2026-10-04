"""Compare explicit T.10 printed numbers with the saved raw reports."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = "fact_finish_t_10"
text = json.loads((HERE / f"{P}-snapshot-docs_course-experiments_results_distillation.json").read_text())
modal = json.loads((HERE / f"{P}-snapshot-docs_course-experiments_results_multimodal_distillation.json").read_text())
weights = json.loads((HERE / f"{P}-weights.json").read_text())
tables = []
attributes = text["results"]["tasks"]["attributes"]
for key, correct, nll, tensor_bytes in [
    ("teacher", 5, 0.5058, 566272),
    ("w16_ce", 1, 1.1134, 54976),
    ("w16_teacher_hard", 1, 1.1134, 54976),
    ("w16_ce_kl", 1, 1.2407, 54976),
    ("w32_ce", 4, 0.4393, 134528),
    ("w32_teacher_hard", 4, 0.4393, 134528),
    ("w32_ce_kl", 4, 0.5037, 134528),
    ("w32_ce_kl_packed4", 3, 0.5960, 64160),
]:
    score = attributes["teacher_test"] if key == "teacher" else attributes["runs"][key]["test"]
    storage = attributes["teacher_storage"] if key == "teacher" else attributes["runs"][key]["storage"]
    assert score["correct"] == correct and score["examples"] == score["eos_count"] == 10
    assert score["supervised_tokens"] == 69
    assert abs(score["answer_nll"] - nll) <= 0.00005
    assert storage["tensor_bytes"] == tensor_bytes
    tables.append(
        {
            "row": key,
            "correct": correct,
            "examples": 10,
            "effective_tokens": 69,
            "nll_raw": score["answer_nll"],
            "nll_printed": nll,
            "tensor_bytes": tensor_bytes,
        }
    )
assert round(text["elapsed_seconds"], 2) == 90.14
assert round(attributes["teacher_cache"]["seconds"], 3) == 0.049
assert attributes["teacher_cache"]["file_bytes"] == 345101
assert round(attributes["hard_target_generation"]["seconds"], 3) == 0.827
assert round(attributes["runs"]["w32_ce"]["training"]["seconds"], 3) == 2.859
assert round(attributes["runs"]["w32_ce_kl"]["training"]["seconds"], 3) == 3.345
style = text["results"]["tasks"]["style_transfer"]
assert style["hard_target_generation"]["wrong"] == 5
wrong_dates = [row for row in style["hard_target_generation"]["audit"] if not row["teacher_correct"]]
assert all(row["question"].startswith("task=date;") for row in wrong_dates)
for report in [style["teacher_style"]] + [
    style["runs"][key]["style"] for key in ["w32_ce", "w32_teacher_hard", "w32_ce_kl"]
]:
    assert report["content_correct"] == 3 and report["examples"] == 27
    assert report["json_valid"] == report["json_examples"] == 7
    arithmetic = [value for name, value in report["rubric"].items() if name != "clarification"]
    assert sum(group["content_correct"] for group in arithmetic) == 0
    assert sum(group["records"] for group in arithmetic) == 21
    assert report["rubric"]["clarification"]["content_correct"] == 3
    assert report["rubric"]["clarification"]["records"] == 6
moe = text["results"]["tasks"]["moe_to_dense"]
for report, nll in [
    (moe["teacher_test"], 2.3017),
    (moe["runs"]["w32_ce"]["test"], 2.4601),
    (moe["runs"]["w32_ce_kl"]["test"], 2.3890),
]:
    assert report["supervised_tokens"] == 41914 == 41862 + 52
    assert report["nll_sequence_chunks"] == 352
    assert report["correct"] == report["eos_count"] == 0 and report["examples"] == 52
    assert abs(report["answer_nll"] - nll) <= 0.00005
gsm = attributes["out_of_domain_gsm8k"]
for report in [gsm["teacher"]] + [run["out_of_domain_gsm8k"] for run in attributes["runs"].values()]:
    assert report["examples"] == report["eos_count"] == 2 and report["correct"] == 0
assert len(attributes["runs"]) == 8
for name, teacher_correct, ce_correct, kl_correct, target_count, train_count in [
    ("vqa", 9, 9, 8, 72, 8391),
    ("joint", 12, 8, 6, 138, 16104),
]:
    task = modal["results"]["tasks"][name]
    assert task["teacher_test"]["correct"] == teacher_correct
    for key, expected in [("ce", ce_correct), ("ce_kl", kl_correct)]:
        report = task["runs"][key]["test"]
        assert report["examples"] == 12 and report["correct"] == expected
        assert report["effective_tokens"] == target_count
        assert report["eos_rate"] == 1.0 and report["invalid_special_tokens"] == 0
        assert task["runs"][key]["training"]["effective_answer_tokens"] == train_count
assert modal["results"]["tasks"]["joint"]["teacher_validation"]["correct"] == 8
joint = modal["results"]["tasks"]["joint"]["runs"]["ce_kl"]
assert joint["blank_audio_test"]["correct"] == 3
assert all(row["generated"].split(",")[0] == "circle" for row in joint["test"]["samples"])
assert [row["generated_ids"] for row in joint["test"]["samples"]] == [
    row["generated_ids"] for row in joint["blank_image_test"]["samples"]
]
for weight in weights:
    if weight["public_id"] == "multimodal_distillation":
        teacher = "teacher" in weight["file"]
        assert weight["tensor_bytes"] == (582656 if teacher else 144384)
        assert sum(row["elements"] for row in weight["tensors"]) == (145664 if teacher else 36096)
        assert weight["config"]["width"] == (64 if teacher else 32)
        assert weight["config"]["layers"] == (2 if teacher else 1)
        assert (weight["modal_config"]["image_size"] // weight["modal_config"]["patch_size"]) ** 2 == (
            16 if teacher else 4
        )
assert math.isclose((120 / 8) * 20 / 60, 5)
assert math.isclose((3 * 400) / 20, 60)
result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_t_10-numbers.py",
    "environment": {
        "device": "CPU record arithmetic",
        "python": "3.13.5",
        "torch": "2.14.1+cpu (weight inspection upstream)",
    },
    "attribute_table": tables,
    "wrong_teacher_dates": wrong_dates,
    "timing_scope": "Original L4 records only; CPU cannot remeasure the reported GPU times.",
    "gsm8k_derivation": [
        "120 pages / 8 pages * 20 minutes / 60 = 5 hours",
        "3 books * 400 pages / 20 pages per day = 60 days",
    ],
    "result": "Every explicit T.10 score, token count, storage value, timing rounding and modality comparison checked; no contradictions.",
}
(HERE / f"{P}-numbers-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
