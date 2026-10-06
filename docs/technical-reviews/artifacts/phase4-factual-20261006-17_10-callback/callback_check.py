"""Actual same-owner callback fingerprint/scope checks; no ML code is rerun."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
TASK = "/root/phase4_factual_coordinator/factual_17_10"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):
    return path.relative_to(ROOT).as_posix()

prior_path = BASE / "prior-report-opaque.json"
assert sha(prior_path) == "075349c8949e843eb0dbb9264fab0b7ea377af9d8f3637d43fca5f50afa589a3"
prior = json.loads(prior_path.read_text())
assert prior["reviewer_task"] == TASK
old_dir = ROOT / "docs/technical-reviews/artifacts/phase4-factual-20261005-17_10"
old_section = (old_dir / "section.md").read_bytes()
current_section = (BASE / "current-section.md").read_bytes()
assert hashlib.sha256(old_section).hexdigest() == prior["source_sha256"]
assert hashlib.sha256(current_section).hexdigest() == "fcdcc50d25552967d35002618f420f1bf8e6b424db5a9f84b2d5b3ccdc52a35c"
assert old_section.decode().count("文件") == 2
assert old_section.decode().replace("文件", "檔案").encode() == current_section
assert (old_dir / "original-fence.py").read_bytes() == (BASE / "current-original-fence.py").read_bytes()

artifact_hashes = []
for item in prior["artifacts"]:
    path = ROOT / item["path"]
    actual = sha(path)
    assert actual == item["sha256"], item["id"]
    artifact_hashes.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
source_hashes = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        actual = sha(ROOT / source["path"])
        assert actual == source["sha256"], source["id"]
        source_hashes.append({"id": source["id"], "path": source["path"], "sha256": actual, "unchanged": True})
assert sha(ROOT / "tiny_perceptron/quantization.py") == sha(old_dir / "quantization.py")
assert sha(ROOT / "docs/course-experiments/results/quantization.json") == sha(old_dir / "original-l4-quantization.json")
metadata = json.loads((BASE / "current-extraction.json").read_text())
assert metadata["source_sha256"] == hashlib.sha256(current_section).hexdigest()
assert len(metadata["python_fences"]) == 1 and metadata["other_fences"] == []
assert metadata["figure_sha256"] == {} and metadata["svg_references"] == []

receipt = {
    "kind": "same_owner_callback_execution_receipt",
    "reviewer_task": TASK,
    "checked_on": "2026-10-06",
    "command": ".venv/bin/python " + relative(BASE / "callback_check.py") + " > " + relative(BASE / "callback-fingerprint-stdout.txt") + " 2> " + relative(BASE / "callback-fingerprint-stderr.txt"),
    "exit_code": 0,
    "environment": {"python": sys.version, "python_executable": sys.executable,
                    "platform": platform.platform(), "torch_installed": importlib.metadata.version("torch"),
                    "device": "CPU file/byte/hash verification; no PyTorch/model execution"},
    "prior_complete_report": {"path": relative(prior_path), "sha256": sha(prior_path)},
    "prior_source_sha256": hashlib.sha256(old_section).hexdigest(),
    "current_source_sha256": hashlib.sha256(current_section).hexdigest(),
    "current_section_lines": [metadata["section_first_line"], metadata["section_first_line"] + len(current_section.splitlines()) - 1],
    "actual_changes": ["heading: 文件變小 -> 檔案變小", "cost paragraph: 文件大小 -> 檔案大小"],
    "changed_substantive_claims": [], "new_or_changed_commands": [],
    "fence_byte_equality": True, "fence_sha256": sha(BASE / "current-original-fence.py"),
    "prior_artifacts_checked": artifact_hashes, "prior_repository_sources_checked": source_hashes,
    "current_original_quantization_source_sha256": sha(ROOT / "tiny_perceptron/quantization.py"),
    "current_original_raw_l4_sha256": sha(ROOT / "docs/course-experiments/results/quantization.json"),
    "reuse_support_scope": "Unchanged nine original claims, exact byte counts/shape/dtype, original bounded CPU proof, L4 aggregate units/divisors/allocator scope, original fixed-version concept sources. Only terminology changed; no new operation/API contract to execute.",
    "limitations": "No new CPU ML, GPU, training, latency/model-score reevaluation, data/model download, full paper re-fetch, or screenshot. No diagram/visual claim is changed or referenced. Original aggregated timing data have no individual samples; prior limitation remains.",
    "result": "All current delta, fence-byte equality, 31 retained artifact hashes and 4 retained repository source hashes checked; unchanged evidence supports current wording.",
}
(BASE / "callback-execution-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reviewer_task": TASK, "current_source_sha256": receipt["current_source_sha256"],
    "changes": receipt["actual_changes"], "new_or_changed_commands": [], "fence_unchanged": True,
    "artifact_hashes_verified": len(artifact_hashes), "repository_source_hashes_verified": len(source_hashes),
    "current_original_code_and_raw_json_unchanged": True, "result": receipt["result"]}, ensure_ascii=False))
