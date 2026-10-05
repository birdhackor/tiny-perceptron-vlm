"""This same factual reviewer verifies the corrected T.6 against original proof.

No training or model/data download. Earlier raw proof and issue are retained.
"""
from pathlib import Path
import ast
from datetime import datetime, UTC
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
IDENTITY = "/root/phase4_factual_coordinator/factual_t_6"
EXPECTED_SOURCE = "bc90e92e0da1c91de6d1fdec61e0e5b355ed3e708fb55875c956b83466a70492"
OLD_REPORT_SHA = "c4be2ca669182297f8ad56f4f2772f8796c50d0f8b66a26ee06dc05761254d64"
RAW_PROOF_SHA = "af10f5de7c9800c4cb6e948f130bbf28fcf47d80ec85bc78655485e8804bb49b"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


old_raw = (OUT / "round1-report.json").read_bytes()
assert sha(old_raw) == OLD_REPORT_SHA
report = json.loads(old_raw)
assert report["reviewer_task"] == IDENTITY and report["reviewer_context"] == "fresh"
assert report["verdict"] == "revise" and report["issues"][0]["status"] == "open"
spec = importlib.util.spec_from_file_location("facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
current, whole, first_line = facts.original_section(ROOT / "course/training.md", "T.6")
assert sha(current) == EXPECTED_SOURCE
old = (OUT / "section.md").read_bytes()
old_phrase = "視覺留出兩張新位置圖，"
new_phrase = "視覺留出兩張向右偏移的藍色圖，方形、圓形各一張；同位置的紅、綠圖仍用於訓練，留出的是未見過的顏色與位置組合。"
assert old.count(old_phrase.encode()) == 1
assert current == old.replace(old_phrase.encode(), new_phrase.encode(), 1)
old_fences = facts.fences(old, first_line)
new_fences = facts.fences(current, first_line)
assert len(old_fences) == len(new_fences) == 4
assert all(a["raw"] == b["raw"] and a["language"] == b["language"] for a, b in zip(old_fences, new_fences))
assert b"![" not in current and b".svg" not in current
current_source_checks = {}
for source in report["sources"]:
    if source["kind"] == "repository_code":
        observed = sha((ROOT / source["path"]).read_bytes())
        assert observed == source["sha256"]
        current_source_checks[source["id"]] = {"path": source["path"], "sha256": observed, "unchanged": True, "personally_read_locator": source["inspection_note"]}
for artifact in report["artifacts"]:
    assert sha((ROOT / artifact["path"]).read_bytes()) == artifact["sha256"]
context_checks = {}
for path, lesson in [("course/training.md", "T.4"), ("course/chapters/11.md", "11.7"), ("course/chapters/05.md", "5.7"), ("course/chapters/12.md", "12.1"), ("course/chapters/11.md", "11.13")]:
    body, _, line = facts.original_section(ROOT / path, lesson)
    unchanged = body == (OUT / f"previous-{lesson}.md").read_bytes()
    context_checks[lesson] = {"path": path, "raw_section_sha256": sha(body), "first_line": line, "unchanged": unchanged}
    assert unchanged
proof_raw = (OUT / "diagnostic-results.json").read_bytes()
assert sha(proof_raw) == RAW_PROOF_SHA
proof = json.loads(proof_raw)
checks = {}


def record(identifier, pointers, result):
    assert result
    checks[identifier] = {"status": "verified", "raw_proof_pointers": pointers, "support_scope_rechecked": next(c["scope"] for c in report["claims"] if c["id"] == identifier)}


record("local-pretrain", ["/pretrain_runs"], len(proof["pretrain_runs"]) == 4 and all(
    r["report"]["mode"] == ("train" if r["updating"] else "dry-run-no-weight-update")
    and bool(r["changed_encoder_parameters"]) == r["updating"]
    and bool(r["save_calls"]) == r["updating"]
    and len(r["report"]["loss"]) == 1 for r in proof["pretrain_runs"]))
partition = proof["pretrain_partition"]
record("holdout-counts", ["/pretrain_partition", "/pretrain_runs/*/report"],
    partition["vision"]["holdout_examples"] == partition["audio"]["holdout_examples"] == 2
    and partition["vision"]["holdout_labels"] == partition["audio"]["holdout_labels"] == [0, 1]
    and partition["vision"]["train_examples"] == 16 and partition["audio"]["train_examples"] == 4
    and partition["audio"]["holdout_raw_frequencies_hz"] == [180, 1000]
    and partition["accuracy_arithmetic"] == {"one_of_two": 1 / 2, "two_of_two": 2 / 2})
tree = ast.parse((ROOT / "scripts/pretrain_encoders.py").read_text())
branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "args.modality == 'vision'")
assignment = next(n for n in branch.body if isinstance(n, ast.Assign))
assert ast.unparse(assignment) == partition["vision"]["original_assignment"]
# Pure structural recomputation of the personally read original comprehension.
rows = [(c, s, o, int(s == "circle"), c == "blue" and o == 1) for c in ("red", "green", "blue") for s in ("square", "circle") for o in (-1, 0, 1)]
held = [r for r in rows if r[-1]]
training = [r for r in rows if not r[-1]]
assert [list(r[:4]) for r in held] == partition["vision"]["holdout_raw_attributes"]
assert [list(r[:4]) for r in training if r[2] == 1] == partition["vision"]["training_at_holdout_offset"]
assert {(r[0], r[2]) for r in held}.isdisjoint({(r[0], r[2]) for r in training})
assert {(r[0], r[2]) for r in training if r[0] == "blue"} == {("blue", -1), ("blue", 0)}
assert {r[0] for r in training if r[2] == 1} == {"red", "green"}
assert all(r[2] == 1 for r in held)
record("vision-new-position", ["/pretrain_partition/vision/{original_assignment,holdout_raw_attributes,training_offsets,training_at_holdout_offset}"], True)
checks["vision-new-position"]["support_scope_rechecked"] = "The held combination is blue+horizontal offset1; training contains blue at offsets -1,0 and red/green at offset1. Same position is shared across splits."
checks["vision-new-position"]["corrected_quote"] = new_phrase
checks["vision-new-position"]["necessary_original_method_locators"] = ["scripts/pretrain_encoders.py main31-41", "tiny_perceptron/multimodal.py scene177-190 (center=size//2+offset on x axis)"]
record("modal-freeze", ["/train_freeze_runs"], len(proof["train_freeze_runs"]) == 9 and all(
    r["changed_parameters"] and set(r["changed_parameters"]) <= set(r["trainable_parameters"])
    and not set(r["unused_open_parameters"]) & (set(r["changed_parameters"]) | set(r["parameters_with_gradient"]))
    and (r["freeze"] != "projector" or all("projector" in k for k in r["trainable_parameters"]))
    and (r["freeze"] != "partial" or all("projector" in k or k.startswith(("language.blocks.0.", "language.blocks.2.")) for k in r["trainable_parameters"]))
    for r in proof["train_freeze_runs"]))
record("inference-targets", ["/synthetic_answer_targets", "/inference_contract"], [r["decoded_supervision"] for r in proof["synthetic_answer_targets"]] == ["circle", "high", "square,low"] and all(r["last_label_is_eos"] for r in proof["synthetic_answer_targets"]) and len(proof["inference_contract"]) == 3 and all(r["expectations_are_task_targets_not_scores"] for r in proof["inference_contract"]))
record("language-member", ["/language_extraction"], proof["language_extraction"]["same_text_input_logits_equal"] and proof["language_extraction"]["class"] == "TinyLM" and proof["language_extraction"]["no_text_ability_claim"])
resume = proof["resume"]
record("resume-contract", ["/resume"], resume["all_weight_tensors_exactly_equal"] and resume["torch_python_rng_equal"] and resume["continuous_steps"] == resume["resumed_final_steps"] == 2 and resume["format_version"] == "multimodal-v1" and len([k for k in resume if k.endswith("rejected")]) == 6 and {"optimizer", "step", "torch_rng", "python_rng", "trainable_parameters"} <= set(resume["payload_keys"]))
record("fixed-recipes", ["/fixed_vqa_partial", "/experiment_dependencies"], all(k.startswith(("image_projector.", "language.blocks.2.")) for k in proof["fixed_vqa_partial"]) and proof["experiment_dependencies"]["projector"]["dependencies"] == ["sft", "encoders"] and proof["experiment_dependencies"]["vqa"]["dependencies"] == ["sft", "encoders", "projector"])
files = proof["file_contract"]
record("file-contract", ["/file_contract", "/ocr_renderer"], files["image_shape_CHW"] == [3, 16, 16] and files["jsonl_relative_path_checked"] and files["decoded_target"] == "12" and [r["accepted"] for r in files["audio_cases"]] == [True, False, False, False] and [r["case"] for r in files["audio_cases"]] == ["valid", "wrong_rate", "stereo", "empty"] and proof["ocr_renderer"]["size_WH"] == [16, 16])
record("sampling-units", ["/file_contract"], files["samples_per_second"] == 1600 / 0.1 == 16000 and files["tone_220_count"] == files["tone_880_count"] == 1600 and files["frequency_not_sample_rate"])
scores = proof["ablation_scoring"]["results"]
for ablation, value in scores.items():
    assert value["examples"] == len(value["samples"]) == 3
    assert value["exact_match"] == sum(s["exact_match"] for s in value["samples"]) / 3
    assert [s["target"] for s in value["samples"]] == ["0", "1", "2"]
    if ablation == "shuffle":
        assert all(s["row"] != s["donor_row"] and s["generated"] == str(s["donor_row"]) and not s["exact_match"] for s in value["samples"])
record("ocr-ablation", ["/ablation_scoring", "/ocr_renderer"], scores["none"]["exact_match"] == 1 and scores["blank"]["exact_match"] == scores["shuffle"]["exact_match"] == 0)
assert set(checks) == {c["id"] for c in report["claims"]}
assert proof["temporary_weights_removed"]
import torch, numpy, PIL, soundfile
environment = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "cuda_build": str(torch.version.cuda), "numpy": numpy.__version__, "pillow": PIL.__version__, "soundfile": soundfile.__version__, "device": "CPU read/structural inspection only; no training run"}
for key in ("python", "python_executable", "torch", "torch_git", "cuda_build", "numpy", "pillow", "soundfile"):
    assert environment[key] == proof["environment"][key]
evidence = {"reviewer_task": IDENTITY, "checked_at_utc": datetime.now(UTC).isoformat(), "source": "course/training.md#T.6", "source_sha256": sha(current), "initial_source_sha256": sha(old), "initial_report_sha256": OLD_REPORT_SHA,
    "retained_raw_cpu_proof_sha256": RAW_PROOF_SHA, "current_section_first_line": first_line, "corrected_quote": new_phrase, "change_scope": "Exactly one held-split sentence changed. Four bash fences, all original methods, proof bytes and necessary contextual sections remain unchanged.", "current_method_hash_checks": current_source_checks, "necessary_context_hash_checks": context_checks,
    "claims_personally_rechecked": checks, "environment": environment, "training_rerun": False, "figure_applicability": "T.6 has no embedded figure; no figure/version changed or visual claim needs reinspection.", "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-t6-independent-20261005/reinspection.py", "result": "All 11 claim supports/limits and versions verified; original issue now resolved; no new unresolved claim.", "exit_code": 0}
(OUT / "current-section.md").write_bytes(current)
(OUT / "reinspection-results.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"personal_reinspection_complete": True, "reviewer_task": IDENTITY, "source_sha256": sha(current), "claim_count": len(checks), "only_one_sentence_changed": True, "training_rerun": False, "reinspection_results_sha256": sha((OUT / "reinspection-results.json").read_bytes()), "checked_at_utc": evidence["checked_at_utc"]}))
