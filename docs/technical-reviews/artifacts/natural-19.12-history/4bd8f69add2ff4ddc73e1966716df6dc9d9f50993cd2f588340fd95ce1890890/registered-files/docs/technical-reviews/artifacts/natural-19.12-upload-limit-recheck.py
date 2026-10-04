"""Narrow CPU dependency recheck after two chapter20.8 unit clarifications.

Reads only this task's report, registered original evidence, saved source,
current chapter/source; no inference, downloads, Git or environment changes.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import platform
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("docs/technical-reviews/artifacts")
PRIOR_SHA = "5e63203f4e69983c035b6b44dce56c5c3a6a82a64cfa1b6f5207da819c2fbf94"
OLD20_SHA = "30821b8f8a0b01d8e787af0846d827b9c15ff970427081975e95fec2ff5b9dd6"
NEW20_SHA = "0ee9e3536881a84760a62b7fa06125a7bf2ab7b07abd37740a67d35f90ead597"
NEW208_SHA = "3733ef68cd74b299286ad7c19ee6603ab71c6fc3add030b15931e238e458fa91"
SECTION_SHA = "734623a7813f3b202f6f81eee51ada41733cd5fc2060e8f8f33d76c8cce22483"
HISTORY = OUT / "natural-19.12-history" / PRIOR_SHA


def raw(path):
    return (ROOT / path).read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == data, str(path)
    else:
        target.write_bytes(data)


def section(data, lesson):
    pattern = rb"(?m)^## " + re.escape(lesson.encode()) + rb" [^\n]*\n"
    start = re.search(pattern, data).start()
    following = re.search(rb"(?m)^## ", data[start + 1:])
    return data[start:start + 1 + following.start()] if following else data[start:]


prior_bytes = raw(HISTORY / "report.json") if (ROOT / HISTORY / "report.json").exists() else raw("docs/technical-reviews/19.12.json")
assert sha(prior_bytes) == PRIOR_SHA
prior = json.loads(prior_bytes)
assert prior["reviewer_task"] == "/root/natural_factual_19_12" and prior["verdict"] == "pass"
assert len(prior["claims"]) == 86 and all(claim["status"] == "verified" for claim in prior["claims"])
old20 = raw(OUT / "natural-19.12-dependency-current20-30821b8f8a0b.md")
new20 = raw("course/chapters/20.md")
assert sha(old20) == OLD20_SHA and sha(new20) == NEW20_SHA
changes = [
    ("檔案最多8 MB。問「照片裡有哪些東西？」", "檔案上限8 MiB，約8.39 MB（介面標成8 MB）。問「照片裡有哪些東西？」"),
    ("先準備最長30秒、最多8 MB的中文錄音檔，格式使用WAV", "先準備最長30秒、檔案同樣上限8 MiB的中文錄音，格式使用WAV"),
]
expected = old20
for before, after in changes:
    assert expected.count(before.encode()) == 1
    expected = expected.replace(before.encode(), after.encode())
assert expected == new20, "Any unexpected whole-chapter change requires new owner review"
new208 = section(new20, "20.8")
old208 = section(old20, "20.8")
assert sha(new208) == NEW208_SHA
assert sha(section(raw("course/chapters/19.md"), "19.12")) == SECTION_SHA
unchanged_sections = {f"20.{i}": sha(section(new20, f"20.{i}")) for i in range(1, 8)}
assert all(section(new20, lesson) == section(old20, lesson) for lesson in unchanged_sections)
blocks = rb"(?ms)^```[^\n]*\n.*?^```[ \t]*$"
assert re.findall(blocks, new208) == re.findall(blocks, old208)

# Preserve the exact passed report and every registered source/evidence byte.
# The prior complete20 source is recovered from the actual saved full snapshot.
save(HISTORY / "report.json", prior_bytes)
entries = {}
for item in prior["sources"] + prior["artifacts"]:
    if "path" not in item:
        continue
    path = item["path"]
    data = old20 if path == "course/chapters/20.md" else raw(path)
    assert sha(data) == item["sha256"], path
    archive_path = HISTORY / "registered-files" / path
    save(archive_path, data)
    entry = entries.setdefault(path, {"path": path, "sha256": sha(data), "archive_path": str(archive_path), "registrations": []})
    entry["registrations"].append(item["id"])
    entry["recovered_from"] = str(OUT / "natural-19.12-dependency-current20-30821b8f8a0b.md") if path == "course/chapters/20.md" else "exact current bytes matching prior registration"
for path, expected_sha in prior["figure_sha256"].items():
    data = raw(path)
    assert sha(data) == expected_sha, path
    archive_path = HISTORY / "registered-files" / path
    save(archive_path, data)
    entries[path] = {"path": path, "sha256": sha(data), "archive_path": str(archive_path), "registrations": ["figure_sha256"]}
manifest = {"reviewer_task": prior["reviewer_task"], "prior_report_sha256": PRIOR_SHA,
            "prior_complete20_snapshot_available": True, "prior_complete20_sha256": OLD20_SHA,
            "files": list(entries.values()), "initial_b821_whole_source_still_unavailable": True}
save(HISTORY / "archive-manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
save(OUT / "natural-19.12-dependency-current20-0ee9e3536881.md", new20)
save(OUT / "natural-19.12-upload-limit-current20.8.md", new208)
diff = "".join(difflib.unified_diff(old20.decode().splitlines(keepends=True), new20.decode().splitlines(keepends=True),
    fromfile="saved complete20 30821b8f8a0b", tofile="current complete20 0ee9e3536881"))
save(OUT / "natural-19.12-upload-limit-source-diff.patch", diff.encode())

# Read the actual upload constant and enforceable raw-byte comparison without
# importing UI code or loading either model. Calculate the stated units on CPU.
ui = raw("tiny_perceptron/natural_ui.py")
tree = ast.parse(ui.decode())
expr = next(node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(name, ast.Name) and name.id == "MAX_UPLOAD_BYTES" for name in node.targets))


def multiply(expression):
    if isinstance(expression, ast.Constant) and isinstance(expression.value, int):
        return expression.value
    assert isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult)
    return multiply(expression.left) * multiply(expression.right)


limit = multiply(expr)
assert limit == 8_388_608 == 8 * 1024**2
assert "if not 0 < len(content) <= MAX_UPLOAD_BYTES:" in ui.decode()
assert "file.size>8*1024*1024" in ui.decode()
assert round(limit / 1_000_000, 2) == 8.39
student = raw("docs/natural-assistant/STUDENT.md")
assert "檔案上限 8 MiB（約 8.39 MB；介面標為 8 MB）" in student.decode()
assert "檔案同樣上限 8 MiB" in student.decode()
prior_proof = json.loads(raw(OUT / "natural-19.12-dependency-recheck.json"))
assert prior_proof["capability_boundaries"]["scene_total"] == [17, 36]
assert prior_proof["capability_boundaries"]["open_scene_total"] == [0, 12]
assert prior_proof["capability_boundaries"]["synthetic_ocr_exact"] == [18, 18]
assert prior_proof["capability_boundaries"]["external_ocr_raw_exact"] == [1, 10]
assert prior_proof["capability_boundaries"]["speech_chat"] == [1, 12]
result = {
    "reviewer_task": prior["reviewer_task"], "command": ".venv/bin/python docs/technical-reviews/artifacts/natural-19.12-upload-limit-recheck.py",
    "prior_report_sha256": PRIOR_SHA, "prior_report_archive": str(HISTORY / "report.json"),
    "prior_complete_dependency_sha256": OLD20_SHA, "current_complete_dependency_sha256": NEW20_SHA,
    "current20.8_raw_sha256": NEW208_SHA, "current19.12_raw_sha256": SECTION_SHA,
    "actual_read_scope": "Owner reread full current20.8 and19.12; whole current20 compared against true saved complete30821 snapshot",
    "exact_changes": [{"before": a, "after": b} for a, b in changes],
    "whole_chapter_diff": str(OUT / "natural-19.12-upload-limit-source-diff.patch"),
    "20.1_to_20.7_unchanged": unchanged_sections, "20.8_all_code_blocks_unchanged": True,
    "registered_source_evidence_and_figures_exact_match_count": len(entries), "archive_manifest": str(HISTORY / "archive-manifest.json"),
    "upload_units": {"source": "tiny_perceptron/natural_ui.py:26,50,56,409", "source_sha256": sha(ui),
        "limit_bytes": limit, "MiB": limit / 1024**2, "decimal_MB": limit / 1_000_000,
        "rounded_decimal_MB": round(limit / 1_000_000, 2), "same_image_and_audio_limit": True,
        "student_doc_sha256": sha(student)},
    "claim_impact": [{"id": c["id"], "statement_sha256": sha(c["statement"].encode()),
        "status": c["status"], "finding": "No change to supporting architecture/results or19.12 raw text; upload unit clarification has no bearing on this claim"} for c in prior["claims"]],
    "retained_route_scope": "Upstream dense2B+course1,605,632 LoRA updates, separate ASR; not continued328,128 MoE. Existing completed/public CPU record and failed natural tasks retain exact original evidence hashes.",
    "retained_capability_counts": {key: prior_proof["capability_boundaries"][key] for key in ("scene_total", "open_scene_total", "synthetic_ocr_exact", "external_ocr_raw_exact", "speech_chat")},
    "environment": {"python": platform.python_version(), "device": "cpu", "libraries": "Python standard library AST,hashlib,difflib,JSON; no model load or inference"},
    "result": "Passed all exact source-diff, unchanged section/command/evidence/figure,86 claim identity and upload arithmetic assertions",
}
(ROOT / OUT / "natural-19.12-upload-limit-recheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("result", "current_complete_dependency_sha256", "current20.8_raw_sha256", "current19.12_raw_sha256", "registered_source_evidence_and_figures_exact_match_count", "upload_units")}, ensure_ascii=False, indent=2))
