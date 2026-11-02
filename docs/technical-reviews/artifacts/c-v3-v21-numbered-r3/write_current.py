"""本人同owner bounded current：只核對既有來源/證據身分，不重跑導航probe。"""

import copy
import hashlib
import importlib.util
import json
import platform
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-revision-20261101-c-v3/technical-v21/numbered_r3/original-v1.json"
RECEIPT = ROOT / "docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/numbered-r3-original-v1.json"
CURRENT = ORIGINAL.with_name("current-v1.json")
ORIGINAL_SHA = "6a43e38c8a1c1616844c965eb6adaf80fc7585577cd6c54fba459c0e5c9726b4"
RECEIPT_SHA = "11623a89f478068ae7623d08c8cd7f938f585a9571a98fe71ffedc1f900f5c11"
TASK = "/root/c_v3_technical_coord/numbered_r3"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv "
    "PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 "
    "uv run --extra cpu --no-sync python "
    "docs/technical-reviews/artifacts/c-v3-v21-numbered-r3/write_current.py"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    if CURRENT.exists():
        raise SystemExit("current-v1已存在，不覆寫")
    assert sha(ORIGINAL) == ORIGINAL_SHA
    assert sha(RECEIPT) == RECEIPT_SHA
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["native_validation"] == "check_original"
    assert receipt["report"] == {"path": relative(ORIGINAL), "sha256": ORIGINAL_SHA}
    assert receipt["reviewer_task"] == TASK and receipt["lesson_id"] == "R.3"
    assert original["reviewer_task"] == TASK and original["reviewer_context"] == "fresh"
    assert original["issues"] == [] and original["verdict"] == "pass"
    assert all(claim["status"] == "verified" for claim in original["claims"])
    execution = json.loads((ROOT / original["artifacts"][0]["path"]).read_text(encoding="utf-8"))
    expected = {relative(ORIGINAL): ORIGINAL_SHA, relative(RECEIPT): RECEIPT_SHA}
    expected[execution["source_path"]] = original["source_sha256"]
    for artifact in original["artifacts"]:
        expected[artifact["path"]] = artifact["sha256"]
    for source in original["sources"]:
        if source["kind"] == "repository_code":
            expected[source["path"]] = source["sha256"]
    for entry in execution["links"]:
        target = (ROOT / "course" / entry["destination"]).resolve()
        expected[relative(target)] = entry["sha256"]
    for entry in execution["chapters"].values():
        expected[entry["path"]] = entry["sha256"]
    expected["course/lessons.md"] = execution["index_sha256"]
    freeze = receipt["source_freeze"]
    expected[freeze["path"]] = freeze["sha256"]
    observations = []
    for path, wanted in sorted(expected.items()):
        observed = sha(ROOT / path)
        observations.append({"path": path, "expected_sha256": wanted,
                             "observed_sha256": observed, "same_identity": observed == wanted})
    readme = (ROOT / "course/README.md").read_text(encoding="utf-8")
    section = re.search(r"^## R\.3 .+?(?=^## |\Z)", readme, re.M | re.S)
    actual_source_sha = hashlib.sha256(section[0].encode()).hexdigest()
    assert actual_source_sha == original["source_sha256"]
    assert all(entry["same_identity"] for entry in observations)
    missing_notebooks = []
    notebook_count = 0
    for chapter, entry in execution["chapters"].items():
        for row in entry["index_notebook_coverage"]:
            notebook_count += 1
            path = ROOT / "notebooks" / chapter / f"{row['lesson']}.ipynb"
            if not path.is_file():
                missing_notebooks.append(relative(path))
    assert notebook_count == execution["indexed_chapter_sections"] and not missing_notebooks
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "device": "cpu", "platform": platform.platform()}
    identity_path = BASE / "bounded-current-identity.json"
    identity = {
        "created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
        "command": COMMAND, "environment": environment,
        "source": "course/README.md#R.3", "source_sha256": actual_source_sha,
        "dependency_identities": observations,
        "notebook_paths_still_present": notebook_count,
        "missing_notebooks": missing_notebooks,
        "original_issues": original["issues"],
        "receipt_received_at": receipt["received_at"],
        "result": "pass",
        "scope": "同owner bounded current只重核真source21、原判及收件、直接連結原文、目錄、執行程式及既有執行artifact的位元組身分；保留原判27項導覽主張與已完成真執行證據。沒有重跑導航probe，不測Notebook內容或成品。Notebook僅重查既有332個路徑仍存在，未聲稱其內容身分或可執行性。原判與receipt僅作所有權/收件lineage，不代替原始技術證據。",
    }
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = copy.deepcopy(original)
    report["created_at"] = datetime.now(UTC).isoformat()
    report["original_issue_resolutions"] = []
    report["bounded_recheck"] = {
        "original_report": {"path": relative(ORIGINAL), "sha256": ORIGINAL_SHA},
        "coordinator_receipt": {"path": relative(RECEIPT), "sha256": RECEIPT_SHA},
        "identity_artifact_id": "a_current_identity",
        "scope": identity["scope"],
        "original_claim_statuses_retained": True,
    }
    report["artifacts"].append({
        "id": "a_current_identity", "kind": "execution", "path": relative(identity_path),
        "sha256": sha(identity_path), "description": identity["scope"],
        "command": COMMAND,
        "result": f"真source21與目前R.3一致；{len(observations)}個來源/依賴檔案完整SHA均與本人原判及協調者receipt一致；332個Notebook路徑仍存在；原判issues=[]，沒有新必要問題。沒有重跑導航probe。",
        "environment": environment,
    })
    report["sources"].extend([
        {"id": "s_current_identity", "kind": "execution", "verified": True,
         "title": "同owner bounded current的真來源/依賴身分重核", "artifact_id": "a_current_identity"},
        {"id": "s_current_inspector", "kind": "repository_code", "verified": True,
         "title": "R.3本人bounded current身分檢查程式", "path": relative(Path(__file__).resolve()),
         "sha256": sha(Path(__file__).resolve()), "version": "2026-11-01本人新建的bounded current核對",
         "inspection_note": "本人核對本程式只讀取原判、收件、真source21與其已列依賴，逐項比SHA及既有Notebook路徑存在性；保留所有原判語義判斷，沒有重跑導航probe、讀舊結論或修改原判。"},
    ])
    for claim in report["claims"]:
        claim["evidence"].append({
            "source_id": "s_current_identity",
            "locator": "source_sha256、dependency_identities、original_issues、scope",
            "supports": "只支持本次同owner可保留原判所用真來源及真執行證據的身分；技術主張仍由原判所列原始執行收據及本人原文核對支持，不以收件當技術正確證明。",
        })
        claim["artifact_ids"].append("a_current_identity")
    report["checks"]["source_verification"]["details"] += " 本次同owner current再次核對真source21與所有已列原始依賴、程式及執行收據SHA未改；原判與協調者收件只作lineage。"
    report["checks"]["limitations"]["details"] += " 本次為相同identity的bounded重核，保留原判完成的證據範圍，沒有重做原先導航probe；332個Notebook只重核路徑存在。"
    spec = importlib.util.spec_from_file_location("r3_current_native", ROOT / "scripts/check_technical_reviews.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    errors = checker._validate(ROOT, ROOT / "course/README.md", "R.3", section[0], report,
                               defaultdict(list, {TASK: ["R.3"]}), set())
    assert not errors, errors
    CURRENT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert sha(ORIGINAL) == ORIGINAL_SHA
    print(json.dumps({"path": relative(CURRENT), "sha256": sha(CURRENT), "verdict": report["verdict"],
                      "original_sha256_unchanged": sha(ORIGINAL), "dependency_files": len(observations),
                      "native_object_errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
