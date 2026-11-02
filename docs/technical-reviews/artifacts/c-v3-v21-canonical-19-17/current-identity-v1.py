"""同一 owner 的有界 current 指紋重核；不重跑原實驗或推論。"""

import copy
import hashlib
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = "docs/technical-reviews/artifacts/c-v3-v21-canonical-19-17/"
ORIGINAL = "docs/course-revision-20261101-c-v3/technical-v21/canonical_19_17/original-v1.json"
ORIGINAL_SHA = "380e44c93dc4011b86a520dc877c18dc0b26a39398cdb2302c479429f2d5f68d"
INTAKE = "docs/course-revision-20261101-c-v3/technical-coordination-v21/receipts/canonical-19-17-original-v1.json"
INTAKE_SHA = "dbacd49a3b164f96e41e020c572d7011fd1d92b31de7ac44fd623534314884d1"
CURRENT = "docs/course-revision-20261101-c-v3/technical-v21/canonical_19_17/current-v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write_new(path, value):
    with (ROOT / path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


assert sha(ROOT / ORIGINAL) == ORIGINAL_SHA
assert sha(ROOT / INTAKE) == INTAKE_SHA
original = read(ORIGINAL)
intake = read(INTAKE)
assert intake["report"] == {"path": ORIGINAL, "sha256": ORIGINAL_SHA}
assert intake["reviewer_task"] == original["reviewer_task"]
assert intake["native_validation"] == "check_original"
assert original["issues"] == [] and len(original["claims"]) == 13

checked = {}


def check(name, expected):
    actual = sha(ROOT / name)
    assert actual == expected, (name, expected, actual)
    checked[name] = actual


probe = read(BASE + "cpu-probe-v1.json")
for name, expected in probe["input_sha256"].items():
    check(name, expected)
for source in original["sources"]:
    if source["kind"] == "repository_code":
        check(source["path"], source["sha256"])
for artifact in original["artifacts"]:
    check(artifact["path"], artifact["sha256"])
for reference in original["retention"].values():
    if isinstance(reference, dict) and "path" in reference and "sha256" in reference:
        check(reference["path"], reference["sha256"])
official = {}
for name, record in probe["official_installed_probability_source"].items():
    assert sha(record["path"]) == record["sha256"]
    official[name] = {"path": record["path"], "sha256": record["sha256"]}
local_root = Path(probe["local_product_identity"]["root"])
local = {}
for name, expected in probe["local_product_identity"]["files"].items():
    path = local_root / name
    assert sha(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]
    local[name] = expected
for kind in ("b", "c"):
    expected = sha(ROOT / f"docs/lesson-products/results/{kind}-artifact-manifest.json")
    assert sha(local_root / kind / "artifact-manifest.json") == expected
    local[kind + "/artifact-manifest.json"] = {"sha256": expected}

chapter = (ROOT / "course/chapters/19.md").read_text()
start = chapter.index("## 19.17 ")
end = chapter.index("\n## ", start + 1)
body = chapter[start : end + 1]
assert hashlib.sha256(body.encode()).hexdigest() == original["source_sha256"]
assert body == (ROOT / "docs/course-revision-20261101-c-v3/source-freeze-v20/sources/19.17.md").read_text()
assert original["figure_sha256"] == {}

identity = {
    "created_at": datetime.now(UTC).isoformat(),
    "owner": original["reviewer_task"],
    "original": {"path": ORIGINAL, "sha256": ORIGINAL_SHA},
    "actual_original_intake": {"path": INTAKE, "sha256": INTAKE_SHA, "received_at": intake["received_at"]},
    "source_identity": {"lesson_id": "19.17", "source_sha256": original["source_sha256"], "figure_sha256": {}},
    "checked_repository_files": checked,
    "checked_installed_official_sources": official,
    "checked_existing_local_products": local,
    "scope": "All original sources/artifacts and actually read input identities match. Retain the original13claims and original measurements. No NN smoke/snippet/policy arithmetic rerun, training, final, network or publication.",
    "measurement_scope": "2026-10-16 fixed experiment remains retention. 84 saved B loops are work counts, not an elapsed-seconds guarantee. Original six CPU smoke calls remain the original execution, not fresh current runs.",
    "issues": [],
    "assertions": "all identity checks passed",
    "environment": {"python": platform.python_version(), "device": "CPU; stdlib identity audit only"},
}
identity_path = BASE + "current-identity-v1.json"
write_new(identity_path, identity)
current = copy.deepcopy(original)
current["created_at"] = datetime.now(UTC).isoformat()
current["original_report"] = {"path": ORIGINAL, "sha256": ORIGINAL_SHA}
current["original_intake_receipt"] = {"path": INTAKE, "sha256": INTAKE_SHA}
current["original_issue_resolutions"] = []
current["current_revalidation"] = {
    "path": identity_path,
    "sha256": sha(ROOT / identity_path),
    "scope": identity["scope"],
    "measurement_scope": identity["measurement_scope"],
}
command = (
    "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv "
    "PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 "
    "uv run --extra cpu --no-sync python " + BASE + "current-identity-v1.py"
)
current["artifacts"].extend([
    {"id": "a_current_identity", "kind": "execution", "path": identity_path,
     "sha256": sha(ROOT / identity_path),
     "description": "同一owner於original真收後，重核本節source21、全部已讀原始來源、原artifacts及既有產品原bytes；沒有重跑推論或實驗。",
     "command": command, "result": "所有identity assertions通過；原13claims與原測量保留，未新增品質或秒數保證。",
     "environment": identity["environment"]},
    {"id": "a_current_code", "kind": "code", "path": BASE + "current-identity-v1.py",
     "sha256": sha(Path(__file__)), "description": "本次有界current identity核對與原生格式驗證的完整程式。"},
])
current["checks"]["source_verification"]["details"] += " original真收後，同owner current再逐檔核全部實讀來源與artifacts，皆為同一bytes；原13claims完整保留。"
assert current["claims"] == original["claims"]
validator = module("canonical1917_current_validator", "scripts/check_technical_reviews.py")
errors = validator._validate(
    ROOT, ROOT / "course/chapters/19.md", "19.17", body, current,
    {original["reviewer_task"]: ["19.17"]}, set(),
)
assert errors == [], errors
assert sha(ROOT / ORIGINAL) == ORIGINAL_SHA
write_new(CURRENT, current)
print(json.dumps({"path": CURRENT, "sha256": sha(ROOT / CURRENT), "verdict": current["verdict"],
                  "created_at": current["created_at"], "claims": len(current["claims"]),
                  "original_issue_resolutions": [], "native_validate_errors": errors,
                  "original_unchanged": sha(ROOT / ORIGINAL) == ORIGINAL_SHA}, ensure_ascii=False, indent=2))
