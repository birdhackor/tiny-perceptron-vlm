"""Recheck storage identities only: no evaluation, model load, download or training."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ARCHIVE = "docs/technical-reviews/history/phase4-2_5-own-initial-pass-b4f2685b62da833f3723d8c705f61d4874e0cfa33eb9f05fc6159e534adbdcdb.json"
OLD_SHA = "b4f2685b62da833f3723d8c705f61d4874e0cfa33eb9f05fc6159e534adbdcdb"
TASK = "/root/phase4_factual_coordinator/factual_2_5"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def identity(path):
    target = ROOT / path
    raw = target.read_bytes()
    return {"path": path, "sha256": sha(raw), "bytes": len(raw)}
def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

assert sha((ROOT / ARCHIVE).read_bytes()) == OLD_SHA
raw_report = (ROOT / "docs/technical-reviews/2.5.json").read_bytes()
assert sha(raw_report) == OLD_SHA
report = json.loads(raw_report)
assert report["reviewer_task"] == TASK and report["reviewer_context"] == "fresh"
artifacts = {item["id"]: item for item in report["artifacts"]}
external_inputs = [item for item in report["artifacts"] if item["path"].startswith("outputs/")]
assert {item["id"] for item in external_inputs} == {"checkpoint-mlp1", "checkpoint-mlp3", "checkpoint-mlp5"}
for item in report["artifacts"]:
    if item["id"].startswith("checkpoint-"):
        continue
    assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"], item["id"]
for source in report["sources"]:
    if source["kind"] == "repository_code":
        assert not source["path"].startswith("outputs/")
        assert sha((ROOT / source["path"]).read_bytes()) == source["sha256"], source["id"]
# Current manuscript/figure identity is checked without reinterpreting old conclusions.
spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
body, whole, first_line = module.original_section(ROOT / "course/chapters/02.md", "2.5")
assert sha(body) == report["source_sha256"]
for path, expected in report["figure_sha256"].items():
    assert sha((ROOT / path).read_bytes()) == expected
probe = json.loads((OUT / "probe-result.json").read_bytes())
local = json.loads((OUT / "historical/existing-local-cpu-result.json").read_bytes())
published = json.loads((OUT / "historical/simple_models.json").read_bytes())
assert local["revision"] == "5d60e35d58a9e085640a957b2ed1808cd4883f53"
assert published["revision"] == "26f34ebb5d1e237611567697d2b3ea4d64669331"
local_hashes = {item["path"]: item["sha256"] for item in local["artifacts"]}
published_hashes = {item["path"]: item["sha256"] for item in published["artifacts"]}
inputs, optional_current_identity_checks = [], []
for name in ("mlp1", "mlp3", "mlp5"):
    observed = probe["evaluations"][name]
    original_path = observed["checkpoint_original_path"]
    expected = observed["checkpoint_sha256"]
    assert expected == local_hashes[name + ".pt"] == artifacts["checkpoint-" + name]["sha256"]
    assert original_path == artifacts["checkpoint-" + name]["path"]
    assert observed["context"] == local["results"]["runs"][name]["context"]
    assert observed["steps"] == local["results"]["runs"][name]["steps"] == 200
    assert observed["parameters"] == local["results"]["runs"][name]["parameters"]
    check = {"path": original_path, "identity_only": True, "model_loaded": False,
             "required_for_future_report_validation": False}
    if (ROOT / original_path).is_file():
        check.update(available_now=True, sha256=sha((ROOT / original_path).read_bytes()))
        assert check["sha256"] == expected
    else:
        check["available_now"] = False
    optional_current_identity_checks.append(check)
    inputs.append({
        "id": name,
        "input_role": "weights evaluated during the original reviewer CPU probe, not this storage recheck",
        "original_local_path": original_path,
        "original_sha256": expected,
        "run": probe["existing_local_version"],
        "model": {"kind": "mlp", "class": "ContextMLP", "format": "simple-v1",
                  "context": observed["context"], "width": 16, "vocabulary_size": 17,
                  "parameters": observed["parameters"],
                  "checkpoint_state_contract": "format_version, model state_dict, vocabulary, context, width, kind; inferred from original run source and config, not loaded again"},
        "training_config_at_source_run": {"seed": 42, "steps": 200, "full_batch_train_targets": 103,
                  "optimizer": "AdamW", "lr": 0.01, "gradient_clip_norm": 1.0,
                  "source": "existing-local-code/scripts/course_experiments/text.py:run_simple_models"},
        "availability_scope": "The original local checkpoint identifies the then-read input only. It is not a report artifact, repository_code source, permanent snapshot, or fresh Git/CI prerequisite.",
        "public_url": None,
        "public_url_reason": "No verified public URL for this specific existing local CPU checkpoint version was used; the older published run's checkpoint metadata is not substituted.",
        "original_recorded_per_split_values": observed["metrics"],
    })
data = []
for split in ("train", "validation", "test"):
    name = f"data/{split}.jsonl"
    permanent = f"{OUT.relative_to(ROOT).as_posix()}/historical/{name}"
    item = identity(permanent)
    assert item["sha256"] == local_hashes[name] == published_hashes[name]
    rows = [json.loads(line) for line in (ROOT / permanent).read_text().splitlines()]
    actual_targets = sum(len(row["text"]) + 1 for row in rows)
    assert len(rows) == probe["denominators"][split]["documents"]
    assert actual_targets == probe["denominators"][split]["effective_next_character_targets"]
    data.append({"split": split, "permanent_raw_data": item,
                 "original_local_path": f"outputs/course-experiments/course-v1/simple_models/{name}",
                 "shared_between_source_run_versions": True,
                 "denominators_from_original_record": probe["denominators"][split]})
code = []
for prefix, receipt in (("historical-code", published), ("existing-local-code", local)):
    for name in ("tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py"):
        path = f"{OUT.relative_to(ROOT).as_posix()}/{prefix}/{name}"
        item = identity(path)
        assert item["sha256"] == receipt["code_sha256"][name]
        code.append({"run_revision": receipt["revision"], "original_repo_path": name, "permanent_source": item})
base = OUT.relative_to(ROOT).as_posix()
numeric_records = [identity(f"{base}/{name}") for name in
                   ("probe-result.json", "probe-stdout.txt", "probe-stderr.txt", "probe.py",
                    "original/fence-1.py", "original/simple_examples.py", "execution-receipts.json",
                    "execution-environment.json", "historical/simple_models.json",
                    "historical/existing-local-cpu-result.json", "historical/vocabulary.json")]
provenance = {
    "schema_version": 1,
    "kind": "permanent_input_identification_and_existing_numeric_evidence",
    "lesson_id": "2.5",
    "reviewer_task": TASK,
    "recorded_on": "2026-10-05",
    "archived_initial_report": {"path": ARCHIVE, "sha256": OLD_SHA},
    "original_execution_scope": "Original successful probe evaluated already-existing local5d60e35 CPU weights; no training, model download or GPU. Published26f34eb receipt read separately, not its original weights evaluated.",
    "this_recheck_scope": "Read and hash-check permanent raw JSON/data/code and optional existing input identities only. No model load, forward/evaluation, source download or training. Recording provenance is not another model execution.",
    "original_input_checkpoints": inputs,
    "original_published_result_version": probe["published_version"],
    "data_versions": data,
    "model_and_example_code_versions": code,
    "permanent_numeric_and_execution_records": numeric_records,
    "original_execution_environment": probe["environment"],
    "original_numeric_tolerances": probe["tolerances"],
    "optional_current_checkpoint_identity_checks": optional_current_identity_checks,
    "fresh_git_ci_requirements": "Only permanent registered report artifacts and source snapshots are required; no outputs path or checkpoint file is required by the repaired report. Replaying the historical forward evaluation would require obtaining the identified original weights independently and is not claimed here.",
}
write_json(OUT / "input-provenance.json", provenance)
observed = {"result": "pass", "kind": "storage_and_identity_recheck_only", "model_execution": False,
            "training_performed": False, "checkpoint_downloaded_or_copied": False,
            "source_sha256": sha(body), "initial_report_sha256": OLD_SHA,
            "permanent_input_provenance": identity(f"{base}/input-provenance.json"),
            "preserved_claim_count": len(report["claims"]),
            "denominators": probe["denominators"],
            "required_outputs_artifacts_found_in_initial_report": len(external_inputs),
            "optional_checkpoint_identity_checks": optional_current_identity_checks,
            "recheck_environment": {"python": sys.version, "device": "cpu", "torch_imported": False}}
write_json(OUT / "permanent-input-provenance-recheck-result.json", observed)
print(json.dumps(observed, ensure_ascii=False, indent=2, allow_nan=False))
