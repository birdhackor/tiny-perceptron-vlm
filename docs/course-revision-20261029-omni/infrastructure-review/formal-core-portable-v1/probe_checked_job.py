"""Actual checked_job gate using a labeled audit fixture in a private review copy."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / "original-source"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "original"
sys.path.insert(0, str(SOURCE))
from scripts.omni_revision import formal_local
from tiny_perceptron.omni_products import training_runtime as trainer

probe = BASE / "probes" / (LABEL + "-checked-job")
probe.mkdir(parents=True, exist_ok=False)
root = probe / "fixture-root"
shutil.copytree(SOURCE, root)
frozen = root / "docs/omni-products/v1"
audit_path = root / "docs/course-revision-20261029-omni/infrastructure/formal-core-audit-v1.json"
recipe_raw = (frozen / "formal-recipes.json").read_bytes()
recipe = json.loads(recipe_raw)
required = {str((frozen / name).relative_to(root)): trainer.file_sha(frozen / name)
            for name in ("data-definition.json", "metrics.json", "formal-recipes.json")}
required.update(recipe["execution_sources_sha256"])
required.update({f"tiny_perceptron/omni_products/{n}": d for n, d in recipe["implementation_sha256"].items()})
audit = {"status": "pass_for_reviewed_scope", "reviewer_task": "/root/review_test_fixture_only",
         "reviewed_files_sha256": required, "test_fixture": True,
         "scope": "Private review gate fixture; not a real audit or permission to train."}
audit_path.parent.mkdir(parents=True, exist_ok=True)
audit_path.write_text(json.dumps(audit))
cases = []
with patch.object(formal_local, "ROOT", root), patch.object(formal_local, "AUDIT", audit_path):
    for job in recipe["job_order"]:
        formal_local.checked_job(frozen, job)
    cases.append({"case": "all_ten_exact_jobs", "accepted": True})
    for mode in ("missing_audit_binding", "self_reviewer", "data_drift", "metrics_drift", "execution_drift"):
        changed = None
        saved = None
        if mode in {"missing_audit_binding", "self_reviewer"}:
            value = json.loads(json.dumps(audit))
            if mode == "missing_audit_binding": value["reviewed_files_sha256"].pop("tiny_perceptron/omni_products/av.py")
            else: value["reviewer_task"] = "/root"
            audit_path.write_text(json.dumps(value))
        else:
            changed = frozen / ("data-definition.json" if mode == "data_drift" else "metrics.json") if mode != "execution_drift" else root / "scripts/omni_revision/formal_local.py"
            saved = changed.read_bytes()
            changed.write_bytes(saved + b"\n")
        try:
            formal_local.checked_job(frozen, recipe["job_order"][0])
            cases.append({"case": mode, "accepted": True})
        except Exception as error:
            cases.append({"case": mode, "accepted": False, "error_type": type(error).__name__})
        if changed is not None: changed.write_bytes(saved)
        audit_path.write_text(json.dumps(audit))
    output = probe / "missing-alarm-output"
    train_calls = []
    with patch.object(formal_local, "signal", SimpleNamespace(signal=lambda *_: None)), \
         patch.object(formal_local, "train", side_effect=lambda *args: train_calls.append(args)):
        try:
            formal_local.execute(recipe["job_order"][0], frozen, output)
        except Exception as error:
            alarm_error = {"type": type(error).__name__, "reason": str(error)}
        else:
            alarm_error = None
    alarm = {"actual_checked_job_used": True, "exception": alarm_error, "train_calls": len(train_calls),
             "lock_left_behind": (output / ".formal-running").exists(),
             "job_origin_written": (output / recipe["job_order"][0] / "job-origin.json").exists(),
             "job_result_written": (output / recipe["job_order"][0] / "job-result.json").exists()}
value = {"scope": "Private labeled audit fixture only; no model training or formal output-root used.",
         "cases": cases, "missing_alarm": alarm, "formal_success_claimed": False}
assert all(not x["accepted"] for x in cases[1:])
(BASE / (LABEL + "-checked-job-result.json")).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(value,ensure_ascii=False,indent=2))
