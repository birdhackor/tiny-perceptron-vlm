"""核對新版文字、沿用證據指紋與事前門檻；唯讀，不重新訓練。"""

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path("docs/technical-reviews/artifacts/review-20261019-chapter-19")
ORIGINAL_REPORT = Path("docs/course-review-20261019/reviews/rv1019_technical_chapter-19/technical-chapter-19-original-01.json")
OLD_SOURCE = Path("docs/course-review-20261019/freeze-01/sources/chapter-19.md")
NEW_SOURCE = Path("docs/course-review-20261019/freeze-05/sources/chapter-19.md")
NEW_FREEZE = Path("docs/course-review-20261019/freeze-05.json")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return (ROOT / path).read_bytes()


def load(path):
    return json.loads(read(path))


original_raw = read(ORIGINAL_REPORT)
assert digest(original_raw) == "8f5a69b66e1ff7f58845556d13c151a4fe15339accb1e223dc611114eac10976", "原報告指紋不符"
original = json.loads(original_raw)
old_output_path = DIRECTORY / "probe-output.json"
old_output = load(old_output_path)
old_source = read(OLD_SOURCE).decode()
new_source = read(NEW_SOURCE).decode()
assert digest(new_source.encode()) == "f61bcc5de4ce485a252fc7d1320d5c9635a5227965936cb33dc4574c98b39ecc", "新版原文指紋不符"
assert digest(read(NEW_FREEZE)) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba", "新版凍結檔指紋不符"
freeze = load(NEW_FREEZE)
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-19")
chapter = read(Path("course/chapters/19.md")).decode()
assert chapter.split("## ", 1)[0] == new_source, "現行章首與凍結原文不同"
assert page["source_sha256"] == digest(new_source.encode()) and page["figures_sha256"] == {}, "凍結頁面索引不符"
before = "讀固定語法的事件或加法題，算出答案"
after = "讀固定語法的事件或加法題，輸出答案預測（事件的狀態追蹤已通過預先訂的門檻，加法還沒達標）"
assert old_source.count(before) == 1 and old_source.replace(before, after, 1) == new_source, "除了預期字句還有其他變更，需另核對"
diff_command = ["git", "diff", "--no-index", str(OLD_SOURCE), str(NEW_SOURCE)]
diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 1 and not diff.stderr, "原文差異命令執行異常"

reused_artifacts = []
for artifact in original["artifacts"]:
    actual = digest(read(Path(artifact["path"])))
    assert actual == artifact["sha256"], "沿用 artifact 指紋已變"
    reused_artifacts.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
code_sources = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = digest(read(Path(source["path"])))
    assert actual == source["sha256"], "沿用程式來源已變，需重新核對"
    code_sources.append({"path": source["path"], "sha256": actual, "unchanged": True})
all_previous_inputs = []
for name, expected in old_output["observed_files_sha256"].items():
    actual = digest(read(Path(name)))
    document = name.startswith("course/")
    if not document:
        assert actual == expected, "原始訓練、資料或測試證據已變"
    all_previous_inputs.append({"path": name, "previous_sha256": expected, "current_sha256": actual, "unchanged": actual == expected, "use": "重新核對導覽" if document else "沿用實質證據"})
links = []
for target in re.findall(r"\]\(([^)]+)\)", new_source):
    name, _, anchor = target.partition("#")
    path = (ROOT / "course/chapters" / name).resolve()
    text = path.read_text()
    headings = re.findall(r"^## (.+)$", text, re.M)
    matches = [h for h in headings if not anchor or h.startswith(anchor + " ")]
    assert not anchor or matches, "導覽定位不存在"
    links.append({"target": target, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path.read_bytes()), "anchor_found": not anchor or bool(matches), "matched_heading": matches[:1] if anchor else []})

records = Path("docs/lesson-products/results")
settings_path = records / "settings-freeze-v1.json"
settings = load(settings_path)
pretest = load(records / "pretest-freeze-record-v1.json")
started = load(records / "settings-freeze-v1.test-started.json")
final_path = records / "final-report-v1.json"
final = load(final_path)
settings_hash = digest(read(settings_path))
assert all(d["settings_freeze_sha256"] == settings_hash for d in (pretest, started, final)), "測試凍結綁定不符"
assert settings["recorded_at"] < pretest["recorded_at"] < started["recorded_at"] < final["recorded_at"], "設定與測試時間先後不符"
assert settings["test_outcomes_observed"] is False and pretest["test_outcomes_observed"] is False, "事前設定紀錄不符"
assert settings["protocol"]["b_primary"] == "boxes, high=5 loops, accuracy>=0.90", "主門檻與核對假設不同"
assert settings["protocol"]["b_secondary"] == "sum accuracy target0.85; any failure is reported", "加法目標與核對假設不同"
thresholds = {"boxes": 0.90, "sum": 0.85}
assessments = {}
for task, threshold in thresholds.items():
    measurement = final["b"]["5"][task]
    exact = measurement["correct"] / measurement["count"]
    gate = final["gates"]["b_primary" if task == "boxes" else "b_secondary_target"]
    assert abs(exact - measurement["accuracy"]) < 1e-7, "原始正確率與整數計數不符"
    assert (exact >= threshold) == gate, "原始門檻結果與計數不符"
    assessments[task] = {"depth": 5, "correct": measurement["correct"], "count": measurement["count"], "families": measurement["families"], "accuracy_recalculated": exact, "recorded_float32_accuracy": measurement["accuracy"], "threshold": threshold, "gate_passed": gate}
replay = old_output["replayed_heldout_arithmetic"]
recount = {str(depth): sum(p["predicted"][str(depth)] == p["gold"] for p in replay["predictions"]) for depth in (1, 3, 5)}
assert len(replay["predictions"]) == 42 and recount == {"1": 0, "3": 1, "5": 2}, "原始完整加法重跑證據不符"

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(DIRECTORY / "probe-current-01.py"),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "environment": {"python": platform.python_version(), "device": "cpu", "platform": platform.platform(), "execution": "standard-library evidence verification; model inference reused from immutable prior execution"},
    "page": {"snapshot": str(NEW_SOURCE), "source_sha256": digest(new_source.encode()), "freeze": str(NEW_FREEZE), "freeze_sha256": digest(read(NEW_FREEZE)), "current_intro_matches": True, "figures": {}},
    "original_report": {"path": str(ORIGINAL_REPORT), "sha256": digest(original_raw)},
    "diff": {"command": " ".join(diff_command), "returncode": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr, "only_replacement": {"before": before, "after": after}},
    "reused_artifacts": reused_artifacts,
    "unchanged_repository_code": code_sources,
    "previous_evidence_inputs": all_previous_inputs,
    "current_navigation": links,
    "threshold_evidence": {"settings_path": str(settings_path), "settings_sha256": settings_hash, "protocol": settings["protocol"], "settings_time": settings["recorded_at"], "pretest_time": pretest["recorded_at"], "test_start_time": started["recorded_at"], "final_time": final["recorded_at"], "all_freeze_bindings_match": True, "final_path": str(final_path), "final_sha256": digest(read(final_path)), "test_outcomes_observed_at_freeze": settings["test_outcomes_observed"], "assessments": assessments},
    "previous_executed_arithmetic": {"source_artifact": str(old_output_path), "sha256": digest(read(old_output_path)), "count": 42, "families": replay["families"], "recounted_correct": recount, "scope": "僅重算先前已執行的完整公開權重推論結果；不是本次重新跑模型。"},
    "changed_claim_assessment": {"outputs_prediction": "verified; Product.solve returns neural argmax, not a guaranteed exact calculator", "tracking_gate": "verified; 2928/2956 >= 0.90 under frozen high=5 finite grammar", "arithmetic_below_target": "verified; 2/42 < 0.85, matching prior complete executed replay"},
    "limits": "沒有新增程式區塊或圖。本次核對文字、目前來源、事前門檻、原始評測與舊 artifact 完整性；沿用的模型實跑由原始 artifact 證明，未重啟訓練或全量評測。",
}
with (ROOT / DIRECTORY / "probe-output-current-01.json").open("x") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
