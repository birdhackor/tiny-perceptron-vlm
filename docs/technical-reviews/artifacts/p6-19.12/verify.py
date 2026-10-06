"""Fresh 19.12 audit: inspect existing raw bytes; no training or model generation."""
from pathlib import Path
import collections
import gzip
import hashlib
import importlib.util
import json
import platform
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def save(name, value):
    (ART / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


spec = importlib.util.spec_from_file_location("original_evaluator", ROOT / "scripts/selftrained/evaluate.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
raw_chapter = (ROOT / "course/chapters/19.md").read_bytes()
start = re.search(rb"^## 19\.12 .+$", raw_chapter, re.M).start()
next_heading = re.search(rb"^## ", raw_chapter[start + 1:], re.M)
end = start + 1 + next_heading.start() if next_heading else len(raw_chapter)
section = raw_chapter[start:end]
(ART / "19.12-initial-source.md").write_bytes(section)
if sha(raw_chapter) == "e6a4b4d0ce69ce82f1c810eb2e9c414aa0e5ea47cf4657be6bf91e5a2b463c5b":
    (ART / "19-initial-frozen-input.md").write_bytes(raw_chapter)

expected = {
    "text": (464, 429, 560), "ocr": (252, 281, 324),
    "vision_clothing": (357, 356, 360), "vision_relation": (1427, 1426, 1440),
    "voice_qa": (42, 37, 90), "voice_topic_continuation": (26, 24, 60),
    "tool_call": (0, 7, 276), "tool_reply": (551, 468, 552),
    "tool_concept": (6, 3, 8), "tool_missing": (5, 8, 8),
    "tool_unavailable": (29, 24, 48), "tool_unsupported": (4, 4, 8),
}
summary = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
           "section_sha256": sha(section), "chapter_frozen_input_sha256": sha(raw_chapter),
           "method": "Reaggregate existing original output rows and execute original scoring only; no model loading or generation.",
           "architectures": {}}

for arch_i, arch in enumerate(("moe", "dense")):
    test = read_json(f"docs/selftrained/results/public-raw/{arch}/test/metrics.json")
    frozen = read_json(f"docs/selftrained/results/public-raw/{arch}/freeze/frozen.json")
    receipt = read_json(f"docs/selftrained/results/public-raw/{arch}/test/receipt.json")
    journal = read_json(f"docs/selftrained/results/public-raw/{arch}/test/evaluation-receipt.json")
    source_index = read_json(f"docs/selftrained/results/v2-{arch}-public-final-test-stage-index.json")
    paths = [Path(k) for k in source_index["evidence"] if k.endswith("/raw/outputs.jsonl")]
    assert len(paths) == 1
    original = paths[0].read_bytes()
    assert sha(original) == journal["outputs_sha256"]
    compressed = gzip.compress(original, compresslevel=9, mtime=0)
    raw_name = f"{arch}-original-test-outputs.jsonl.gz"
    (ART / raw_name).write_bytes(compressed)
    assert gzip.decompress((ART / raw_name).read_bytes()) == original
    rows = [json.loads(line) for line in original.splitlines()]
    assert len(rows) == len({row["record"]["id"] for row in rows}) == 3734
    assert all(row["record"]["split"] == "test" for row in rows)
    grouped = collections.defaultdict(list)
    rescored = 0
    history_matches = 0
    for row in rows:
        grouped[row["record"]["task"]].append(row)
        calculated = evaluator.score_reply(row["record"], row["trace"]["final_output"], row["trace"])
        assert calculated == row["score"], row["record"]["id"]
        rescored += 1
        if row["record"]["task"] == "voice_topic_continuation":
            cfg = row["record"]["evaluation"]
            prompt = row["record"]["messages"][:cfg["first_prompt_message_count"]]
            actual = row["trace"]["continuation_messages"]
            assert actual == prompt + [{"role": "assistant", "content": row["trace"]["initial_output"]}, row["record"]["messages"][cfg["continuation_user_message_index"]]]
            assert row["end_to_end_success"] == bool(row["first_reply_score"]["semantic"] and row["first_reply_score"]["format"] is not False and row["score"]["semantic"] and row["score"]["format"] is not False)
            history_matches += 1
    results = {}
    for task, exp in expected.items():
        items = grouped[task]
        if task == "voice_topic_continuation":
            numerator = sum(row["end_to_end_success"] for row in items)
        elif task == "tool_call":
            numerator = sum(row["score"]["tool_roundtrip"] for row in items)
        else:
            numerator = sum(row["score"]["semantic"] for row in items)
        assert (numerator, len(items)) == (exp[arch_i], exp[2])
        results[task] = {"numerator": numerator, "denominator": len(items), "rate": numerator / len(items)}
    ocr = [p for r in grouped["ocr"] for p in r["perception"] if p["kind"] == "ocr"]
    ctc_correct = sum(p["prediction"] == r["record"]["supervision"]["ocr_text"] for r in grouped["ocr"] for p in r["perception"] if p["kind"] == "ocr")
    assert ctc_correct == len(ocr) == 324
    keys = ["checkpoint_sha256", "safe_weights_sha256", "inference_manifest_sha256", "selected_checkpoint_sha256", "thresholds"]
    assert all(test[k] == frozen[k] for k in keys)
    assert frozen["created_at"] < receipt["started_at"]
    assert frozen["selection"] == "validation only" and frozen["test_once"] is True
    assert journal["expected_count"] == journal["completed_count"] == test["count"] == 3734
    assert journal["resumed_from"] is None and test["resumed_completed_count"] == 0
    assert receipt["status"] == "completed" and journal["status"] == "complete" and receipt["returncode"] == 0
    assert receipt["public_export"]["public_export"]["revision"] == "979cdfacc588ad0536f1c64fff96f264571cf054"
    manifest = json.loads((ART / f"{arch}-public-inference-manifest.json").read_bytes())
    validation = [json.loads(line) for line in (ART / f"{arch}-original-training-validation-metrics.jsonl").read_bytes().splitlines()]
    best = min(validation, key=lambda row: row["validation_loss"])
    assert best["step"] == manifest["selected_step"] == 1000
    assert manifest["selected_checkpoint_sha256"] == frozen["selected_checkpoint_sha256"]
    assert manifest["selection"] == "validation_loss" and manifest["origin"]["kind"] == "all-neural-weights-random"
    code_matches = {}
    for filename in ["scripts/selftrained/evaluate.py", "scripts/selftrained/train.py", "tiny_perceptron/selftrained/dataset.py", "tiny_perceptron/selftrained/tools.py", "tiny_perceptron/selftrained/model.py", "tiny_perceptron/selftrained/inference.py"]:
        code_matches[filename] = sha((ROOT / filename).read_bytes()) == frozen["code_sha256"][filename]
    assert all(code_matches.values())
    threshold_failures = {}
    for key, value in test["stratified_final_reply"].items():
        if key.startswith("text/intent=") and value["semantic_rate"] < frozen["thresholds"]["text_semantic_per_intent"]:
            threshold_failures[key] = (value["semantic_numerator"], value["count"], .8)
        if key.startswith(("voice_qa/intent=", "voice_topic_continuation/intent=")) and value["semantic_rate"] < frozen["thresholds"]["voice_per_intent"]:
            threshold_failures[key] = (value["semantic_numerator"], value["count"], .8)
    assert results["tool_call"]["rate"] < frozen["thresholds"]["tool_roundtrip"]
    # Independently reconstruct the archived counting method; no prior results read.
    derived_failures = dict(threshold_failures)
    thresholds = frozen["thresholds"]
    for task in ["tool_missing", "tool_unsupported", "tool_unavailable"]:
        rate = sum(r["score"]["semantic"] for r in grouped[task]) / len(grouped[task])
        if rate < thresholds["unsupported_clarification"]:
            derived_failures[task] = (sum(r["score"]["semantic"] for r in grouped[task]), len(grouped[task]), thresholds["unsupported_clarification"])
    for task, items in grouped.items():
        formats = [r["score"]["format"] for r in items if r["score"].get("format") is not None]
        if formats and sum(formats) / len(formats) < thresholds["format"]:
            derived_failures[task + "/format"] = (sum(formats), len(formats), thresholds["format"])
    derived_failures["tool_call/true_roundtrip"] = (results["tool_call"]["numerator"], 276, thresholds["tool_roundtrip"])
    ocr_items = grouped["ocr"]
    ocr_exact = sum(r["score"]["exact"] for r in ocr_items) / len(ocr_items)
    ocr_cer = sum(r["score"]["ocr_errors"] for r in ocr_items) / sum(r["score"]["ocr_characters"] for r in ocr_items)
    if ocr_exact < thresholds["ocr_exact"]:
        derived_failures["ocr/exact"] = (sum(r["score"]["exact"] for r in ocr_items), len(ocr_items), thresholds["ocr_exact"])
    if ocr_cer > thresholds["ocr_cer_max"]:
        derived_failures["ocr/CER"] = (ocr_cer, thresholds["ocr_cer_max"])
    assert len(derived_failures) == (15 if arch == "moe" else 12)
    summary["architectures"][arch] = {
        "table_and_boundary_counts": results, "row_rescoring_matches": rescored,
        "raw_output_original_path": str(paths[0]), "raw_output_bytes": len(original), "raw_output_sha256": sha(original),
        "permanent_lossless_gzip": raw_name, "gzip_sha256": sha(compressed),
        "voice_actual_first_output_history_matches": history_matches,
        "ctc_exact": [ctc_correct, len(ocr)],
        "tool_chain_segments": {key: test["per_task_final_reply"]["tool_call"][key] for key in ["tool_parameters", "tool_valid", "tool_executed", "tool_final", "tool_roundtrip"]},
        "best_validation_step": best["step"], "best_validation_loss": best["validation_loss"],
        "public_manifest_selected_step": manifest["selected_step"], "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"],
        "freeze_before_test": True, "frozen_test_identity_matches": True, "completed_once_receipt": True,
        "code_matches_frozen": code_matches, "original_thresholds": frozen["thresholds"],
        "direct_text_voice_threshold_failures": threshold_failures,
        "derived_threshold_count": len(derived_failures), "derived_threshold_failures": derived_failures,
        "count_scope": "Archived derivative method applies unsupported_clarification to tool_missing, tool_unsupported, tool_unavailable. Frozen JSON binds numeric thresholds but does not enumerate this full mapping.",
    }

manifest = read_json("docs/selftrained/v2-manifest.json")
archive = ROOT / manifest["package"]["path"]
archive_raw = archive.read_bytes()
assert sha(archive_raw) == manifest["package"]["sha256"]
assert len(archive_raw) == manifest["package"]["bytes"]
data = {}
with tarfile.open(archive) as tar:
    for entry in manifest["records"]:
        raw = tar.extractfile(entry["path"]).read()
        assert len(raw) == entry["bytes"] and sha(raw) == entry["sha256"]
        data[entry["path"]] = [json.loads(line) for line in raw.splitlines()]
        assert sha(raw) == read_json("docs/selftrained/results/public-raw/moe/freeze/frozen.json")["data_sha256"][entry["path"]]
ocr_scope = {}
for split in ["train", "validation", "test"]:
    rows = data[f"ocr-{split}.jsonl"]
    ocr_scope[split] = {"records": len(rows), "characters": "".join(sorted(set("".join(r["supervision"]["ocr_text"] for r in rows)))),
                        "font_families": sorted({r["supervision"]["font_family"] for r in rows}),
                        "target_lengths": sorted({len(r["supervision"]["ocr_text"]) for r in rows}),
                        "all_roi_provided": all("roi" in r for r in rows),
                        "all_target_glyphs_within_roi": all(all(r["roi"][0] <= b[0] <= b[2] <= r["roi"][2] and r["roi"][1] <= b[1] <= b[3] <= r["roi"][3] for b in r["supervision"]["glyph_boxes"]) for r in rows)}
assert all(len(s["characters"]) == 12 and s["all_roi_provided"] and s["all_target_glyphs_within_roi"] for s in ocr_scope.values())
assert ocr_scope["train"]["font_families"] == ocr_scope["test"]["font_families"] == ["Sans", "Serif"]
assert ocr_scope["test"]["target_lengths"] == [1, 2, 3, 4]
test_records = [r for name, rows in data.items() if name.endswith("-test.jsonl") for r in rows]
assert len(test_records) == len({r["id"] for r in test_records}) == 3734
voice = data["voice-test.jsonl"]
voice_scope = {"records": len(voice), "tasks": dict(collections.Counter(r["task"] for r in voice)),
               "recording_assets": len({r["audio"] for r in voice}), "groups": len({r["group_id"] for r in voice}),
               "intents": sorted({r["supervision"]["intent"] for r in voice}),
               "recording_assets_by_split": {s: len({r["audio"] for r in data[f"voice-{s}.jsonl"]}) for s in ["train", "validation", "test"]},
               "recording_asset_overlap_train_test": len({r["audio"] for r in data["voice-train.jsonl"]} & {r["audio"] for r in voice})}
assert voice_scope["recording_assets"] == voice_scope["groups"] == 30
vision = data["vision-test.jsonl"]
vision_scope = {"tasks": dict(collections.Counter(r["task"] for r in vision)),
                "label_ids": sorted({x for r in vision for x in r["supervision"]["vision_labels"] if x >= 0}),
                "layout_slot_counts": sorted({len(r["image_layout"]["slots"]) for r in vision}),
                "all_layout_provided": all("image_layout" in r for r in vision)}
summary["data_scope"] = {"archive_sha256": sha(archive_raw), "archive_bytes": len(archive_raw), "records_verified": len(data), "test_question_count": len(test_records), "ocr": ocr_scope, "voice": voice_scope, "vision": vision_scope}
training = []
for p in sorted((ROOT / "docs/selftrained/results/training-raw").glob("*/raw/train-receipt.json")):
    d = json.loads(p.read_bytes())
    assert d["test_used_for_selection"] is False and d["completed_requested_steps"] is True and d["origin"]["kind"] == "all-neural-weights-random"
    training.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p.read_bytes()), "stage": d["stage"], "steps": d["steps"], "selection": d["selection"], "test_used_for_selection": d["test_used_for_selection"], "completed_requested_steps": d["completed_requested_steps"], "origin_kind": d["origin"]["kind"]})
assert len(training) == 15
summary["training_receipts"] = training
namespace = {}
exec("by_question = (12 + 0) / (12 + 6)\nby_task = (1.0 + 0.0) / 2", {}, namespace)
assert namespace["by_question"] == 2 / 3 and namespace["by_task"] == .5
summary["mean_example"] = namespace
save("verification-output.json", summary)
print(json.dumps({"section_sha256": summary["section_sha256"], "raw_rows_rescored": {a:d["row_rescoring_matches"] for a,d in summary["architectures"].items()}, "table_and_boundary_counts": {a:d["table_and_boundary_counts"] for a,d in summary["architectures"].items()}, "data_scope": summary["data_scope"], "training_receipts": len(training), "mean_example": namespace}, ensure_ascii=False, indent=2))
