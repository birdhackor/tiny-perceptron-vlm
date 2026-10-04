"""Owner CPU source/evidence recheck for the single optional training link."""
from pathlib import Path
import difflib
import hashlib
import json
import platform
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("docs/technical-reviews/artifacts")
PRIOR_SHA = "4bd8f69add2ff4ddc73e1966716df6dc9d9f50993cd2f588340fd95ce1890890"
OLD_SHA = "0ee9e3536881a84760a62b7fa06125a7bf2ab7b07abd37740a67d35f90ead597"
NEW_SHA = "1f74c69cf2b2cc2b543f0a2d876a6d3eb4ae92baf834a2e77f3c94a491ad129d"
NEW208_SHA = "0bc720a18baa3ba542418ab149ded8146965bcf3da3ac7f36a9a8654471c956d"
SECTION_SHA = "734623a7813f3b202f6f81eee51ada41733cd5fc2060e8f8f33d76c8cce22483"
HISTORY = OUT / "natural-19.12-history" / PRIOR_SHA


def read(path):
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
    start = re.search(rb"(?m)^## " + re.escape(lesson.encode()) + rb" [^\n]*\n", data).start()
    following = re.search(rb"(?m)^## ", data[start + 1:])
    return data[start:start + 1 + following.start()] if following else data[start:]


prior_bytes = read(HISTORY / "report.json") if (ROOT / HISTORY / "report.json").exists() else read("docs/technical-reviews/19.12.json")
assert sha(prior_bytes) == PRIOR_SHA
prior = json.loads(prior_bytes)
assert prior["reviewer_task"] == "/root/natural_factual_19_12" and prior["verdict"] == "pass"
assert len(prior["claims"]) == 86 and all(c["status"] == "verified" for c in prior["claims"])
old = read(OUT / "natural-19.12-dependency-current20-0ee9e3536881.md")
new = read("course/chapters/20.md")
assert sha(old) == OLD_SHA and sha(new) == NEW_SHA
insertion = "想在自己的GPU重做這份配方，見[LoRA訓練實作](../../docs/natural-assistant/TRAINING.md)；它從底座另建一份LoRA，先驗證再固定自己的模型版本。".encode()
marker = "這份公開小包供推論使用。".encode()
assert old.count(marker) == 1 and new == old.replace(marker, marker + insertion)
assert sha(section(new, "20.8")) == NEW208_SHA
assert sha(section(read("course/chapters/19.md"), "19.12")) == SECTION_SHA
assert all(section(old, f"20.{i}") == section(new, f"20.{i}") for i in range(1, 8))
blocks = rb"(?ms)^```[^\n]*\n.*?^```[ \t]*$"
assert re.findall(blocks, section(old, "20.8")) == re.findall(blocks, section(new, "20.8"))
save(HISTORY / "report.json", prior_bytes)
entries = {}
for item in prior["sources"] + prior["artifacts"]:
    if "path" not in item:
        continue
    path = item["path"]
    data = old if path == "course/chapters/20.md" else read(path)
    assert sha(data) == item["sha256"], path
    archive_path = HISTORY / "registered-files" / path
    save(archive_path, data)
    entry = entries.setdefault(path, {"path": path, "sha256": sha(data), "archive_path": str(archive_path), "registrations": []})
    entry["registrations"].append(item["id"])
    if path == "course/chapters/20.md":
        entry["recovered_from"] = str(OUT / "natural-19.12-dependency-current20-0ee9e3536881.md")
for path, expected in prior["figure_sha256"].items():
    data = read(path)
    assert sha(data) == expected, path
    archive_path = HISTORY / "registered-files" / path
    save(archive_path, data)
    entries[path] = {"path": path, "sha256": sha(data), "archive_path": str(archive_path), "registrations": ["figure_sha256"]}
manifest = {"reviewer_task": prior["reviewer_task"], "prior_report_sha256": PRIOR_SHA, "true_prior_complete20_sha256": OLD_SHA,
    "initial_b821_whole_source_still_unavailable": True, "files": list(entries.values())}
save(HISTORY / "archive-manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
save(OUT / "natural-19.12-dependency-current20-1f74c69cf2b2.md", new)
save(OUT / "natural-19.12-training-link-current20.8.md", section(new, "20.8"))
diff = "".join(difflib.unified_diff(old.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True), fromfile="true saved complete0ee9", tofile="current complete1f74"))
save(OUT / "natural-19.12-training-link-source-diff.patch", diff.encode())

# Inspect only the scope asserted by the new link: a fresh own LoRA and own
# validation/selection. Reading its full text does not certify its GPU results.
guide = read("docs/natural-assistant/TRAINING.md")
guide_text = guide.decode()
assert "從固定的圖文底座建立一份新的LoRA" in guide_text
assert "這次訓練從底座新建LoRA" in guide_text
assert "只用驗證題決定要保留哪份自己的模型" in guide_text
assert "不能套到你的新輸出" in guide_text
excerpt = "\n\n".join(p for p in guide_text.split("\n\n") if any(key in p for key in ("建立一份新的LoRA", "這次訓練從底座新建LoRA", "只用驗證題決定", "不能套到你的新輸出")))
save(OUT / "natural-19.12-training-link-guide-excerpt.txt", ("Source: docs/natural-assistant/TRAINING.md\nFull source SHA-256: " + sha(guide) + "\nScope: textual fresh-LoRA/validation/own-version statements only; no GPU execution or quality certification.\n\n" + excerpt + "\n").encode())
result = {
    "reviewer_task": prior["reviewer_task"], "command": ".venv/bin/python docs/technical-reviews/artifacts/natural-19.12-training-link-recheck.py",
    "prior_report_sha256": PRIOR_SHA, "prior_report_archive": str(HISTORY / "report.json"), "archive_manifest": str(HISTORY / "archive-manifest.json"),
    "prior_complete_dependency_sha256": OLD_SHA, "current_complete_dependency_sha256": NEW_SHA,
    "current20.8_raw_sha256": NEW208_SHA, "current19.12_raw_sha256": SECTION_SHA,
    "actual_read_scope": "Full current20.8 read; actual complete0ee9/current1f74 source diff inspected. Linked TRAINING text read to inspect only fresh LoRA and own validation/version scope, not to certify command execution or GPU quality.",
    "only_change": insertion.decode(), "20.1_to_20.7_unchanged": True, "20.8_all_code_blocks_unchanged": True,
    "registered_source_evidence_and_figure_paths_matched": len(entries),
    "claim_impact": [{"id": c["id"], "statement_sha256": sha(c["statement"].encode()), "status": c["status"],
        "finding": "Unchanged statement and supporting evidence. Fresh own-LoRA optional route transfers no author score or quality guarantee; oldMoE/newDense+ASR boundary unchanged."} for c in prior["claims"]],
    "linked_guide": {"path": "docs/natural-assistant/TRAINING.md", "sha256": sha(guide), "checked_scope": "Fresh LoRA from fixed base; own validation before fixed own version; author scores cannot be applied to new weights", "gpu_execution_or_quality_verified_here": False},
    "retained_limits": "Existing Dense+1,605,632 LoRA update/separate ASR original records remain identical; scene17/36/open0/12, syntheticOCR18/18 vs external1/10, ASRchat1/12 and CPU failures still bound reliability. No new experiment reported.",
    "environment": {"python": platform.python_version(), "device": "cpu", "libraries": "Python standard library byte/hash/JSON/diff checks; no inference or model load"},
    "result": "Passed exact single insertion, unchanged sections/code/evidence/figures,86 claim identity and limited linked-guide scope checks; no GPU execution,download,Git or environment mutation",
}
(ROOT / OUT / "natural-19.12-training-link-recheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("result", "current_complete_dependency_sha256", "current20.8_raw_sha256", "current19.12_raw_sha256", "registered_source_evidence_and_figure_paths_matched")}, ensure_ascii=False, indent=2))
