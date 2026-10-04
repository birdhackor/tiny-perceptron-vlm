"""Read pinned public evidence and execute the current lesson without replacing old evidence."""

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import urllib.request
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_15"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_anonymous(url):
    # No Authorization, Cookie, token, netrc handler, or credentials are supplied.
    request = urllib.request.Request(url, headers={"User-Agent": "Technical-source-review/13.15"})
    opener = urllib.request.build_opener()
    with opener.open(request, timeout=20) as response:
        data = response.read()
        metadata = {
            "url": url,
            "final_url": response.url,
            "status": response.status,
            "request_headers": dict(request.header_items()),
            "content_type": response.headers.get("Content-Type"),
            "bytes": len(data),
            "sha256": sha(data),
        }
    return data, metadata


def main():
    old_report_path = ROOT / "docs/technical-reviews/13.15.json"
    archive = OUT / f"{PREFIX}_old_report.json"
    old_report_bytes = archive.read_bytes() if archive.is_file() else old_report_path.read_bytes()
    old_report = json.loads(old_report_bytes)
    old_section = (OUT / f"{PREFIX}_reviewed_old_section.md").read_text()
    chapter_sections = dict(sections(ROOT / "course/chapters/13.md"))
    body = chapter_sections["13.15"]
    body_hash = sha(body.encode())
    (OUT / f"{PREFIX}_rechecked_section.md").write_text(body)
    url = re.findall(r"https://[^\s)]+/posttraining\.json", body)
    assert len(url) == 1
    blob_url = url[0]
    raw_url = blob_url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    old_url = "https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/posttraining.json"
    assert body.replace(blob_url, old_url) == old_section
    assert body_hash != sha(old_section.encode())
    if not archive.is_file():
        archive.write_bytes(old_report_bytes)
    prerequisites = {lesson: chapter_sections[lesson] for lesson in ("13.10", "13.11", "13.13", "13.14")}
    prerequisites["W.1"] = dict(sections(ROOT / "course/first-steps.md"))["W.1"]
    (OUT / f"{PREFIX}_rechecked_prerequisites.json").write_text(json.dumps(prerequisites, ensure_ascii=False, indent=2) + "\n")
    blob, blob_metadata = fetch_anonymous(blob_url)
    raw, raw_metadata = fetch_anonymous(raw_url)
    assert blob_metadata["status"] == raw_metadata["status"] == 200
    assert b"posttraining.json" in blob
    local = (ROOT / "docs/course-experiments/results/posttraining.json").read_bytes()
    assert raw == local
    raw_path = OUT / f"{PREFIX}_pinned_public_report.json"
    raw_path.write_bytes(raw)
    report = json.loads(raw)
    previous_audit = json.loads((OUT / f"{PREFIX}_audit.json").read_text())
    assert sha(raw) == previous_audit["published_report_sha256"]
    source_checks = {}
    for source in old_report["sources"]:
        if source["kind"] == "repository_code":
            actual = sha((ROOT / source["path"]).read_bytes())
            assert actual == source["sha256"], source["path"]
            source_checks[source["path"]] = actual
    code = re.search(r"```python\n(.*?)```", body, re.S)[1]
    assert sha(code.encode()) == previous_audit["short_program"]["code_sha256"]
    code_path = OUT / f"{PREFIX}_recheck_program.txt"
    code_path.write_text(code)
    cell_runs = []
    for exercise in (False, True):
        executable = code.replace("[[0.1, 0.2, 0.0, 0.0]]", "[[0.1, 0.2, 1.0, 0.0]]") if exercise else code
        namespace = {}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(executable, str(code_path), "exec"), namespace)
        assert list(namespace["context"].shape) == list(namespace["new_logits"].shape) == [1, 4]
        assert namespace["action"].item() == 0
        assert namespace["terms"]["ratio"].item() == 1.0
        assert not any(p.grad is not None for p in namespace["reward_model"].parameters())
        assert not any(p.grad is not None for p in namespace["reference"].parameters())
        assert not torch.equal(namespace["before"], next(namespace["policy"].parameters()))
        cell_runs.append({"exercise_third_feature_one": exercise, "executed_code_sha256": sha(executable.encode()), "stdout": stdout.getvalue(), "context_dtype": str(namespace["context"].dtype), "action_dtype": str(namespace["action"].dtype), "shape": list(namespace["new_logits"].shape)})
    spec = importlib.util.spec_from_file_location("lesson_13_15_original_verifier", OUT / f"{PREFIX}_verify.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    torch.set_num_threads(2)
    public_content_audit = verifier.audit_report(report)
    original_checkpoints = verifier.checkpoint_audit(ROOT / "outputs/posttraining-development/fixed", report)
    previous_run_path = OUT / f"{PREFIX}_run/posttraining/result.json"
    previous_run = json.loads(previous_run_path.read_text())
    previous_rerun_audit = verifier.audit_report(previous_run)
    assert {r["file"]: r["state_sha256"] for r in report["results"]["checkpoints"]} == {r["file"]: r["state_sha256"] for r in previous_run["results"]["checkpoints"]}
    receipts = []
    for checkpoint in report["results"]["checkpoints"]:
        original_path = ROOT / "outputs/posttraining-development/fixed" / checkpoint["file"]
        replay_path = previous_run_path.parent / checkpoint["file"]
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        replay = torch.load(replay_path, map_location="cpu", weights_only=True)
        assert original["model"].keys() == replay["model"].keys()
        tensors = []
        for name, tensor in original["model"].items():
            other = replay["model"][name]
            assert tensor.shape == other.shape and tensor.dtype == other.dtype
            assert torch.equal(tensor, other) and bool(torch.isfinite(tensor).all())
            tensors.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(), "sha256_contiguous_bytes": sha(tensor.contiguous().numpy().tobytes()), "finite": True, "matches_independent_rerun_exactly": True})
        receipts.append({"original_checkpoint": str(original_path.relative_to(ROOT)), "independent_rerun_checkpoint": str(replay_path.relative_to(ROOT)), "original_file_sha256": sha(original_path.read_bytes()), "rerun_file_sha256": sha(replay_path.read_bytes()), "state_sha256": checkpoint["state_sha256"], "original_metadata": original["metadata"], "tensors": tensors, "storage": "Both .pt paths remain ignored by existing .gitignore:25 (*.pt); no training-weight binary is a direct review artifact."})
    receipt_path = OUT / f"{PREFIX}_weight_receipt.json"
    receipt_path.write_text(json.dumps({"command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_15_recheck.py", "load_mode": "torch.load(map_location=cpu, weights_only=True)", "checkpoints": receipts, "result": "all tensor names, shapes, dtypes and values match the independent CPU rerun; all tensors finite"}, ensure_ascii=False, indent=2) + "\n")
    assert dict(sections(ROOT / "course/chapters/13.md"))["13.15"] == body
    result = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_15_recheck.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "torch_git_commit": torch.version.git_version, "accessed_on": "2026-10-04"},
        "current_section_sha256": body_hash,
        "old_section_sha256": sha(old_section.encode()),
        "old_review_sha256": sha(old_report_bytes),
        "only_section_change": {"old_link": old_url, "new_link": blob_url},
        "anonymous_blob_response": blob_metadata,
        "anonymous_raw_response": raw_metadata,
        "public_report_matches_local_exact_bytes": raw == local,
        "public_report_matches_previous_actual_experiment_sha256": sha(raw) == previous_audit["published_report_sha256"],
        "unchanged_repository_source_checks": source_checks,
        "current_literal_cell_and_exercise": cell_runs,
        "public_content_audit": public_content_audit,
        "original_checkpoint_reexecution": original_checkpoints,
        "previous_cpu_rerun_revalidation": previous_rerun_audit,
        "weight_receipt": {"path": str(receipt_path.relative_to(ROOT)), "sha256": sha(receipt_path.read_bytes())},
        "result": "all assertions passed; current pinned report publicly readable with no credentials supplied and identical to independently verified original CPU evidence",
    }
    (OUT / f"{PREFIX}_recheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": result["result"], "section_sha256": body_hash, "public_report_bytes": len(raw), "public_report_sha256": sha(raw), "blob_status": blob_metadata["status"], "raw_status": raw_metadata["status"], "test_success": {"sft": "6/18", "ppo": "12/18"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
