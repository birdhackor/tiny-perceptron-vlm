"""Preserve original manifest bytes and recheck lesson 8.9 evidence publication closure."""

import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from scripts.check_technical_reviews import sections

ROOT = Path.cwd()
ART = "docs/technical-reviews/artifacts/"
PREFIX = "fact_v2_08_09"
REPORT = ROOT / "docs/technical-reviews/8.9.json"
EXPECTED_OLD_REPORT_SHA = "e741cf3023ddcff777999d50f07cfc4dc19f4b12c20c5c5a82d3f6f708ce9724"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def retain(relative, raw):
    path = ROOT / relative
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Refusing to overwrite distinct preserved evidence: {relative}")
    path.write_bytes(raw)
    assert path.read_bytes() == raw


old_bytes = REPORT.read_bytes()
assert sha(old_bytes) == EXPECTED_OLD_REPORT_SHA
old = json.loads(old_bytes)
assert old["reviewer_task"] == "/root/integration_technical_coordinator/fact_v2_08_09"
body = dict(sections(ROOT / "course/chapters/08.md"))["8.9"]
assert sha(body.encode("utf-8")) == old["source_sha256"]
old_path = ART + PREFIX + "_pre-closure-report.json"
retain(old_path, old_bytes)
report = copy.deepcopy(old)
mapping = {
    f"checkpoints/course/{model}/{kind}-manifest.json": ART + PREFIX + f"_{model}-{kind}-manifest.json"
    for model in ("style", "lora")
    for kind in ("download", "export")
}
equivalence = []
for artifact in report["artifacts"]:
    original = artifact["path"]
    if original not in mapping:
        continue
    raw = (ROOT / original).read_bytes()
    parsed = json.loads(raw)
    assert isinstance(parsed, dict) and parsed["schema_version"] == 1
    assert sha(raw) == artifact["sha256"]
    retained = mapping[original]
    retain(retained, raw)
    assert sha((ROOT / retained).read_bytes()) == artifact["sha256"]
    artifact["path"] = retained
    artifact["original_path"] = original
    artifact["description"] += " 原始bytes逐份複製至可提交快照；SHA與原審查依賴完全相同。"
    equivalence.append({"artifact_id": artifact["id"], "original_path": original, "snapshot_path": retained, "expected_sha256": artifact["sha256"], "original_sha256": sha(raw), "snapshot_sha256": sha((ROOT / retained).read_bytes()), "byte_length": len(raw), "byte_identical": (ROOT / retained).read_bytes() == raw})
assert len(equivalence) == 4
source_paths_updated = []
for source in report["sources"]:
    if source.get("path") in mapping:
        original = source["path"]
        source["path"] = mapping[original]
        assert sha((ROOT / source["path"]).read_bytes()) == source["sha256"]
        source_paths_updated.append({"id": source["id"], "original": original, "snapshot": source["path"]})
assert report["claims"] == old["claims"]
assert report["checks"] == old["checks"]
assert report["sources"] == old["sources"]
assert report["verdict"] == old["verdict"] == "pass"
new_code_path = ART + PREFIX + "_closure.py"
history_path = ART + PREFIX + "_closure-recheck.json"
report["artifacts"].extend([
    {"id": "a_closure_old_report", "kind": "source_snapshot", "path": old_path, "sha256": sha(old_bytes), "description": "本人原始8.9 pass報告完整原bytes，保留原四份ignored manifest依賴位置供核查歷史。"},
    {"id": "a_closure_code", "kind": "code", "path": new_code_path, "sha256": sha((ROOT / new_code_path).read_bytes()), "description": "本人補齊可提交證據：保留舊report與四份原bytes、逐SHA/byte等價核對與Git可提交性核對的實際程式。"},
])
direct = {item["path"]: item["sha256"] for item in report["artifacts"]}
direct.update({source["path"]: source["sha256"] for source in report["sources"] if source["kind"] == "repository_code"})
for path, expected in direct.items():
    assert sha((ROOT / path).read_bytes()) == expected, path
to_submit = ["docs/technical-reviews/8.9.json", *direct, history_path]
ignored = subprocess.run(["git", "check-ignore", "--no-index", *to_submit], capture_output=True, text=True, check=False)
assert ignored.returncode == 1 and not ignored.stdout and not ignored.stderr
tracked = subprocess.run(["git", "ls-files", "--", *to_submit], capture_output=True, text=True, check=True)
untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *to_submit], capture_output=True, text=True, check=True)
tracked_paths = set(tracked.stdout.splitlines())
untracked_paths = set(untracked.stdout.splitlines())
assert set(to_submit) - {history_path} <= tracked_paths | untracked_paths
environment = {"python": platform.python_version(), "torch": importlib.metadata.version("torch"), "device": "CPU; file bytes and offline Git checks", "platform": platform.platform()}
history = {
    "reviewer_task": old["reviewer_task"],
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_08_09_closure.py",
    "result": "Four original manifest snapshots byte-identical and SHA-identical; every direct artifact and repository source hash matches; no direct evidence is ignored; original semantic verdict remains pass.",
    "environment": environment,
    "old_report_path": old_path,
    "old_report_sha256": sha(old_bytes),
    "source_sha256": old["source_sha256"],
    "equivalence": equivalence,
    "repository_sources_relocated": source_paths_updated,
    "semantic_equality": {"claims": True, "sources": True, "checks": True, "source_sha256": True, "verdict": "pass"},
    "direct_dependency_hashes": direct,
    "git_checks": {"check_ignore": {"command": "git check-ignore --no-index [current report and every direct dependency]", "returncode": ignored.returncode, "stdout": ignored.stdout, "stderr": ignored.stderr}, "tracked": sorted(tracked_paths), "addable_untracked": sorted(untracked_paths), "pending_history_path": history_path, "publication_state": "Tracked files or ordinary nonignored untracked evidence; coordinator can stage and commit. This check does not claim a commit was created."},
    "scope": "Evidence path relocation only; original manifests and audit/checker history retained. No lesson/data/environment changes, network, training, paid action or reinterpretation of numerical results.",
}
history_raw = (json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
retain(history_path, history_raw)
report["artifacts"].append({"id": "a_closure_recheck", "kind": "execution", "path": history_path, "sha256": sha(history_raw), "description": "本人原bytes等價性與Git可提交性recheck history；維持pass的依據，不是重新訓練。", "command": history["command"], "result": history["result"], "environment": environment})
report["review_rechecks"] = [{"reason": "保存四份ignored manifest為Git可提交原bytes快照", "previous_report_artifact_id": "a_closure_old_report", "execution_artifact_id": "a_closure_recheck", "reviewer_task": old["reviewer_task"], "verdict": "pass", "details": "四份artifact保留原id和完整SHA，仅把直接path指向相同原bytes快照；無repository source需要改路徑。15項主張、五項checks、來源與教材SHA皆未改。"}]
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"report": "docs/technical-reviews/8.9.json", "report_sha256": sha(REPORT.read_bytes()), "verdict": report["verdict"], "snapshots": equivalence, "direct_ignored_dependencies": 0, "old_report_preserved": old_path}, ensure_ascii=False, indent=2))
