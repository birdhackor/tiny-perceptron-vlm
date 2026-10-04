"""Integrity validation of my report/evidence, not an automatic factual verdict."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
report=json.loads((OUT/"report.json").read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert "lesson_id" not in report
assert report["reviewer_task"]=="/root/v4_review_coordinator/factual_guide_student"
assert report["reviewer_context"]=="fresh"
for path,digest in report["current_document_sha256"].items():assert sha(ROOT/path)==digest
for path,digest in report["figure_sha256"].items():assert sha(ROOT/path)==digest
assert (ROOT/report["source"]).read_bytes()==(OUT/"STUDENT.initial-fullfile.md").read_bytes()
source_ids={s["id"] for s in report["sources"]}
artifact_ids={a["id"] for a in report["artifacts"]}
for claim in report["claims"]:
    assert claim["status"]=="verified"
    for evidence in claim["evidence"]:assert evidence["source_id"] in source_ids
    for artifact in claim["artifact_ids"]:assert artifact in artifact_ids
    if claim["kind"]=="empirical":assert isinstance(claim["verification"]["denominators"],dict) and claim["verification"]["denominators"]
for artifact in report["artifacts"]:
    assert sha(ROOT/artifact["path"])==artifact["sha256"],artifact["path"]
    if artifact["kind"]=="execution":assert artifact["command"] and artifact["result"] and artifact["environment"]
for source in report["sources"]:
    assert source["verified"]
    if source["kind"]=="repository_code":assert sha(ROOT/source["path"])==source["sha256"]
    if source["kind"] in {"paper","official_docs","official_source"}:
        assert source["checked_original"] and source["version"] and source["inspection_note"] and source["url"].startswith("https://")
        assert sha(ROOT/source["ignored_original_path"])==source["retrieval_sha256"]
assert set(report["checks"])=={"factual_accuracy","numeric_verification","figure_consistency","source_verification","limitations"}
assert all(c["status"]=="pass" for c in report["checks"].values())
assert not report["issues"] and report["verdict"]=="pass"
assert sha(OUT/"report.initial.json")==report["revision_history"][0]["sha256"]
print(json.dumps({"status":"passed","scope":"Report references/hashes/identity/denominators only; factual verdict is independent personal judgment","report_sha256":sha(OUT/"report.json"),"current_document_sha256":report["current_document_sha256"],"figure_sha256":report["figure_sha256"],"claims":len(report["claims"]),"sources":len(report["sources"]),"artifacts":len(report["artifacts"]),"initial_report_preserved":True},indent=2))
