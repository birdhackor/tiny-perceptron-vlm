"""歷史callback的真whole checker與bounded API；不建主處置、不偽造seal。"""

from pathlib import Path
from datetime import UTC, datetime
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[5]
BASE = "docs/course-revision-20261029-omni/reviews/technical/technical_chapter_19"
OWN = BASE + "/technical-original-v1.json"
APPENDIX = BASE + "/technical-historical-appendix-rv1019-technical-chapter-19-v1.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
TASK = "/root/omni_technical_coordinator/technical_chapter_19"
OUTPUT = ROOT / BASE / "historical-callback-self-check-v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {"path": path, "sha256": sha(ROOT / path)}


def load(path):
    return json.loads((ROOT / path).read_bytes())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


started = datetime.now(UTC).isoformat()
receipt = {
    "reviewer_task": TASK,
    "started_at": started,
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-chapter-19-v1/historical-rv1019-v1/check.py",
    "script": ref(str(Path(__file__).relative_to(ROOT))),
    "original_report": ref(OWN), "appendix": ref(APPENDIX),
    "current_validation_source_freeze": ref(FREEZE),
    "resume_disclosure": {
        "coordination_interruption_reported_by_parent": True,
        "exact_interruption_start_end_independently_observed": False,
        "resumed_self_check_started_at": started,
        "previously_completed_work": "本人原判15:18 UTC已封存收件；歷史來源實讀與完整42題重跑18:26 UTC、appendix18:28 UTC已保存。續接保留上述先前實讀，沒有冒稱18:41才重新fresh閱讀。",
        "no_metadata_version_rewrite": "appendix仍綁實際讀過的source14；本人original仍綁實際fix13。現在以source15驗相同頁面與未漂移來源，不覆寫兩原件。",
    },
    "bounds": {
        "full_gate_executed": False,
        "current_main_dispositions_preservation_verified": False,
        "stage_seal_verified": False,
        "closed_item_decisions_context": "authenticated accepted prior dispositions only; per-identity contract self-check. Coordinator confirms no sealed/current new main dispositions yet.",
        "not_run": ["Rechecks.__init__/group/collection", "new assignment/collection chronology", "new main dispositions preservation", "stage seals", "full all gate"],
    },
    "whole_checks": [],
}

original, appendix = load(OWN), load(APPENDIX)
assert receipt["original_report"]["sha256"] == "1778278b379fa457f6b7eacacf87f15af0afa63a4b63f17502b23309f0a5d905"
assert receipt["appendix"]["sha256"] == "c991a8a5cdb4de0a052dc934effc3b58866892faecf632991d87beb1d6461016"
assert original["created_at"] < appendix["created_at"] < started
assert appendix["full_review"] == appendix["original_report"] == ref(OWN)
assert appendix["original_source_freeze"] == ref("docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json")
page = next(p for p in load(FREEZE)["pages"] if p["page_id"] == "chapter-19")
previous_page = next(p for p in load(appendix["source_freeze"]["path"])["pages"] if p["page_id"] == "chapter-19")
assert page == previous_page
body = (ROOT / page["source"]).read_bytes()
body = body[: re.search(rb"^## ", body, re.M).start()]
assert body == (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(body).hexdigest() == original["source_sha256"] == appendix["source_sha256"] == page["source_sha256"]
drift = [s["path"] for s in original["sources"] if s["kind"] == "repository_code" and sha(ROOT / s["path"]) != s["sha256"]]
assert not drift
receipt["unchanged_scope"] = {"source15_page_equals_personally_read_source14_page": True, "source_sha256": page["source_sha256"], "original_repository_code_sources_drift": drift, "original_sources_count": len(original["sources"]), "figure_sha256": {}}

for report in [OWN, APPENDIX]:
    cmd = ["uv", "run", "--extra", "cpu", "python", "docs/course-review-20261019/tools/check_technical_report.py", "chapter-19", report, "--freeze", FREEZE, "--reader-task", "/root/omni_reader_products"]
    at = datetime.now(UTC).isoformat()
    run = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    receipt["whole_checks"].append({"command": cmd, "started_at": at, "completed_at": datetime.now(UTC).isoformat(), "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})

try:
    api = module("chapter19_callback_incremental_api", "docs/review-tools/incremental_revision.py")
    historical = module("chapter19_callback_historical_api_v2", "docs/review-tools/historical_fix_rechecks_v2.py")
    request_group = load(REQUEST)["groups"][5]
    checker = historical.Rechecks.__new__(historical.Rechecks)
    checker.root, checker.api = ROOT, api
    checker.entries, checker.used, checker.reports = {}, set(), []
    checker.checked_readers = set()
    checker.tasks = {TASK: "technical", "/root/omni_reader_products": "reader"}
    checker.live, checker.frozen = {"chapter-19": page}, {"chapter-19": page}

    # 真實已接受pointer/gate/freeze/assignments/dispositions鏈；不製造本輪主表。
    checker.prior(request_group["prior_active_pointer"])
    checker.manifest = copy.deepcopy(checker.prior_manifest)
    checker.decisions = {api_key: row for row in checker.old_dispositions["items"] for api_key in [historical._legacy.issue_key(row)]}
    checker.validate_page("chapter-19", TASK, original, appendix)
    appendix_at = api.at(appendix["created_at"])
    checker.unavailable(request_group["unavailability"], request_group["old_owner"], appendix_at)
    rows = appendix["historical_issue_rechecks"]
    assert len(rows) == len(request_group["items"]) == 2
    checked = []
    for expected, row in zip(request_group["items"], rows, strict=True):
        for name in ["disposition_ref", "original_report", "json_pointer", "historical_fix_verification", "issue_id", "necessity", "affected_pages"]:
            assert row[name] == expected[name]
        old_report = api.document(ROOT, expected["original_report"])
        assert row["original_issue"] == api.node(old_report, expected["json_pointer"])
        assert row["source_quote"] in body.decode()
        assert row["original_source_quote"] in api.artifact(ROOT, row["original_source_snapshot"]).decode()
        checker.closed_item({"kind": "page"}, row, original, appendix, TASK, "chapter-19", appendix_at)
        decision = api.node(checker.old_dispositions, expected["disposition_ref"]["json_pointer"])
        assert checker.page(decision, expected["historical_fix_verification"], "chapter-19")
        checked.append({"issue_id": row["issue_id"], "original_report": row["original_report"], "json_pointer": row["json_pointer"], "original_issue_dict_exact": True, "necessity": row["necessity"], "original_issue_status": row["original_issue"]["status"], "current_status": row["status"], "bounded_closed_item": "passed", "new_main_disposition_preservation_verified": False})
    checker.finish()
    receipt["bounded_historical_api"] = {
        "status": "passed_bounded_metadata_contract_checks",
        "api_files": [ref("docs/review-tools/incremental_revision.py"), ref("docs/review-tools/historical_fix_rechecks.py"), ref("docs/review-tools/historical_fix_rechecks_v2.py")],
        "actual_methods": ["v2.Rechecks.prior (inherited v1)", "v2.Rechecks.validate_page (full + appendix _validate)", "v2.Rechecks.unavailable", "v2.Rechecks.closed_item x2", "v2.Rechecks.page x2", "v2.Rechecks.finish"],
        "accepted_prior_active_pointer": request_group["prior_active_pointer"],
        "accepted_dispositions": checker.old_disposition_ref,
        "entries": checked,
        "entry_count": len(checker.entries), "used_count": len(checker.used),
        "scope_warning": "closed_item.decisions索引來源是經prior認證的accepted原處置，只驗本兩筆identity契約。未給新主表、未執行group/collection，也沒有stage seal或全gate結論。",
    }
except Exception as error:
    receipt["bounded_historical_api"] = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}

receipt["completed_at"] = datetime.now(UTC).isoformat()
receipt["status"] = "passed_bounded_self_checks_only" if all(r["exit_code"] == 0 for r in receipt["whole_checks"]) and receipt["bounded_historical_api"]["status"] == "passed_bounded_metadata_contract_checks" else "failed"
with OUTPUT.open("x") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"status": receipt["status"], "output": str(OUTPUT.relative_to(ROOT)), "sha256": sha(OUTPUT), "whole_checks": [{"report": r["command"][7], "exit_code": r["exit_code"], "stdout": r["stdout"]} for r in receipt["whole_checks"]], "bounded_api": receipt["bounded_historical_api"]}, ensure_ascii=False, indent=2))
raise SystemExit(0 if receipt["status"] == "passed_bounded_self_checks_only" else 1)
