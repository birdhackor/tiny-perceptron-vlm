"""Prose-only recheck: personally reinspect originals; verify immutable reused evidence."""
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
EXTRACT = ROOT / "outputs/reviewer-tools/phase4-8_5-independent-recheck1"
INITIAL_SHA = "7c2aacfff2e628efb36c5c5af0687fc865b00f641bc1e562548b6034f787e3f7"
CURRENT_SHA = "f3948d42d8961345088f0acf6068b0c506080620b1aaee613819ea5a975d8455"
HISTORY = ROOT / "docs/technical-reviews/history" / ("phase4-8_5-own-initial-revise-" + INITIAL_SHA + ".json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(HISTORY) == INITIAL_SHA
initial = json.loads(HISTORY.read_bytes())
assert initial["reviewer_task"] == "/root/phase4_factual_coordinator/factual_8_5"
assert initial["verdict"] == "revise"
assert initial["issues"][0]["status"] == "unresolved"
assert sha(EXTRACT / "section.md") == CURRENT_SHA

new_section = (EXTRACT / "section.md").read_text()
print("FULL CURRENT SECTION (personally reread)")
print(new_section)
assert "本例多出「答案是：」的散文前綴時，它會報錯" in new_section
assert "這裡「格式」只檢查Python預設解析成功、且字典只有answer欄位" in new_section
assert "預設解析器還接受`NaN`等JSON標準外的寫法，因此嚴格JSON規則須另作檢查" in new_section
assert "格式破損或多出散文時" not in new_section

for name in ("section.md", "fence-1.py", "extraction.json"):
    shutil.copyfile(EXTRACT / name, HERE / name)
old_fence, new_fence = BASE / "original-fence/fence-1.py", EXTRACT / "fence-1.py"
assert old_fence.read_bytes() == new_fence.read_bytes()

reused = []
for artifact in initial["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"], artifact["id"]
    reused.append({"id": artifact["id"], "path": artifact["path"], "sha256": artifact["sha256"]})
for fetch in json.loads((BASE / "source-fetch-receipt.json").read_bytes()):
    assert sha(ROOT / fetch["path"]) == fetch["sha256"]

ranges = {
    "rfc8259.txt": [(362, 378), (541, 548)],
    "python-json.rst": [(353, 365), (615, 631), (659, 694)],
    "python-json-decoder.py": [(340, 351)],
}
print("OFFICIAL ORIGINAL SOURCE REINSPECTION")
for name, spans in ranges.items():
    # RFC raw text contains page formfeeds; conventional locator lines count LF only.
    lines = (BASE / "sources" / name).read_text().split("\n")
    for start, end in spans:
        print(f"{name}:{start}-{end}")
        print("\n".join(f"{index + 1}: {lines[index]}" for index in range(start - 1, end)))

saved = json.loads((BASE / "verification-results.json").read_bytes())
nan = next(item for item in saved["bounded_cases"] if item["case"] == "nan_extension")
infinity = next(item for item in saved["bounded_cases"] if item["case"] == "infinity_extension")
for item in (nan, infinity):
    assert item["parse_success"] is True and item["format"] is True and item["content"] is False
assert saved["original_fence_sha256"] == sha(new_fence)
assert saved["rubric"] == {"records": 7, "content_correct": 0, "style_correct": 7, "json_valid": 7}
assert not json.loads((EXTRACT / "extraction.json").read_bytes())["svg_references"]
print("REUSED ORIGINAL EXECUTION (SHA verified; not rerun)")
print(json.dumps({"fence_sha256": sha(new_fence), "nan": nan, "infinity": infinity,
                  "saved_rubric": saved["rubric"]}, ensure_ascii=False, indent=2))
facts = {
    "reviewer_task": initial["reviewer_task"], "reviewed_on": "2026-10-05",
    "initial_report_path": HISTORY.relative_to(ROOT).as_posix(), "initial_report_sha256": INITIAL_SHA,
    "initial_source_sha256": initial["source_sha256"], "current_source_sha256": CURRENT_SHA,
    "current_section_fully_read": True, "current_python_fence_sha256": sha(new_fence),
    "fence_raw_bytes_unchanged": True, "official_source_reinspection_ranges": ranges,
    "reused_artifacts_sha_verified": reused,
    "execution_scope": "This command rechecks raw section/source/evidence bytes; it does not rerun original fence, variants, rubric, model or training.",
    "reused_execution_support": "Initial original fence/14 bounded same-loop variants and historical rubric recomputation apply because fence and every evidence input are SHA-identical.",
    "resolution": "New wording confines parse failure to literal example prefix; newly stated format is default parse success plus one dict key and explicitly permits nonstandard NaN. This matches RFC8259/CPython and original NaN flag result.",
    "figure_sha256": {}, "environment": {"python": platform.python_version(), "executable": sys.executable, "device": "cpu", "libraries": "stdlib only"},
}
(HERE / "reuse-facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
print("RECHECK ASSERTIONS PASSED; original runs reused, not rerun")
