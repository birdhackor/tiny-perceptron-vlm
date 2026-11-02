"""W.7 same-reviewer bounded callback: source identity only, no NN execution."""

import ast
import difflib
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = "docs/course-revision-20261029-omni/reviews/P5-actual/"
ORIGINAL = BASE + "technical-W.7/technical-original-v1.json"


def raw(path):
    return (ROOT / path).read_bytes()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def ref(path):
    value = raw(path)
    return {"path": path, "bytes": len(value), "sha256": digest(value)}


def page(freeze, name):
    return next(p for p in freeze["pages"] if p["page_id"] == name)


def section(path, name):
    text = raw(path).decode()
    found = re.search(r"^## " + re.escape(name) + r" [^\n]*\n.*?(?=^## |\Z)", text, re.M | re.S)
    assert found, (path, name)
    return found[0].encode()


original = json.loads(raw(ORIGINAL))
original_proof = json.loads(raw("docs/technical-reviews/artifacts/omni-P5-W.7/inspection-v2.json"))
old_canonical_path = original["canonical_source_freeze"]["path"]
old_numbered_path = original["source_freeze"]["path"]
new_canonical_path = BASE + "source-freeze-v10.json"
new_numbered_path = BASE + "technical-numbered-source-freeze-v5.json"
old_canonical = json.loads(raw(old_canonical_path))
old_numbered = json.loads(raw(old_numbered_path))
new_canonical = json.loads(raw(new_canonical_path))
new_numbered = json.loads(raw(new_numbered_path))
assert digest(raw(old_canonical_path)) == original["canonical_source_freeze"]["sha256"]
assert digest(raw(old_numbered_path)) == original["source_freeze"]["sha256"]
assert new_numbered["canonical_source_freeze"] == {
    "path": new_canonical_path, "sha256": digest(raw(new_canonical_path))
}

old_w7, new_w7 = page(old_numbered, "W.7"), page(new_numbered, "W.7")
old_w7_bytes, new_w7_bytes = raw(old_w7["snapshot"]), raw(new_w7["snapshot"])
assert digest(old_w7_bytes) == old_w7["source_sha256"] == original["source_sha256"]
assert digest(new_w7_bytes) == new_w7["source_sha256"]
assert old_w7_bytes == new_w7_bytes == section("course/first-steps.md", "W.7")
assert old_w7["figures_sha256"] == new_w7["figures_sha256"] == {}
old_first_steps, new_first_steps = page(old_canonical, "first-steps"), page(new_canonical, "first-steps")
assert raw(old_first_steps["snapshot"]) == raw(new_first_steps["snapshot"]) == raw("course/first-steps.md")

repo_sources = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        current = digest(raw(source["path"]))
        repo_sources.append({"id": source["id"], "path": source["path"],
                             "original_sha256": source["sha256"], "current_sha256": current,
                             "unchanged": source["sha256"] == current})
drift = [s for s in repo_sources if not s["unchanged"]]
assert len(drift) == 1 and drift[0]["id"] == "s_repo_28" and drift[0]["path"] == "course/chapters/19.md"
assert not [s for s in repo_sources if s["path"] == "course/chapters/20.md"]

artifact_rechecks = []
for item in original["artifacts"]:
    current = digest(raw(item["path"]))
    assert current == item["sha256"]
    artifact_rechecks.append({"id": item["id"], **ref(item["path"]), "original_unchanged": True})

mapped_sections = []
for mapping in original_proof["mapping"]:
    for s in mapping["sections"]:
        current = digest(section(s["path"], s["section"]))
        mapped_sections.append({"package": mapping["package"], "section": s["section"], "path": s["path"],
                                "original_sha256": s["section_sha256"], "current_sha256": current,
                                "unchanged": current == s["section_sha256"]})
mapped_drift = [s for s in mapped_sections if not s["unchanged"]]
assert len(mapped_drift) == 1 and mapped_drift[0]["section"] == "19.22"

old_sound, new_sound = page(old_canonical, "19.22"), page(new_canonical, "19.22")
old_sound_bytes, new_sound_bytes = raw(old_sound["snapshot"]), raw(new_sound["snapshot"])
assert digest(old_sound_bytes) == old_sound["source_sha256"]
assert digest(new_sound_bytes) == new_sound["source_sha256"]
assert new_sound_bytes == section("course/chapters/19.md", "19.22")
old_phrase = "都是尚未實測的預定門檻"
new_phrase = "都是在實測前固定的驗收門檻"
assert old_sound_bytes.count(old_phrase.encode()) == 1
assert old_sound_bytes.replace(old_phrase.encode(), new_phrase.encode(), 1) == new_sound_bytes

current_chapter = raw("course/chapters/19.md")
assert current_chapter.count(new_sound_bytes) == 1
restored = current_chapter.replace(new_sound_bytes, old_sound_bytes, 1)
assert digest(restored) == drift[0]["original_sha256"]
old_blocks = re.findall(r"```python\n(.*?)```", old_sound_bytes.decode(), re.S)
new_blocks = re.findall(r"```python\n(.*?)```", new_sound_bytes.decode(), re.S)
assert old_blocks == new_blocks
for block in new_blocks:
    ast.parse(block)

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reviewer_task": original["reviewer_task"],
    "review_type": "same-reviewer bounded current callback; not a new blind reading",
    "original_report": ref(ORIGINAL),
    "old_canonical_freeze": ref(old_canonical_path), "current_canonical_freeze": ref(new_canonical_path),
    "old_numbered_freeze": ref(old_numbered_path), "current_numbered_freeze": ref(new_numbered_path),
    "w7": {"old_snapshot": ref(old_w7["snapshot"]), "current_snapshot": ref(new_w7["snapshot"]),
           "current_source": "course/first-steps.md#W.7", "identical_bytes": True,
           "whole_first_steps_identical": True, "figure_sha256": {}},
    "repository_source_rechecks": repo_sources, "repository_source_drift": drift,
    "original_artifact_rechecks": artifact_rechecks,
    "mapped_section_rechecks": mapped_sections, "mapped_section_drift": mapped_drift,
    "chapter19": {"old_19_22": ref(old_sound["snapshot"]), "current_19_22": ref(new_sound["snapshot"]),
                  "restored_whole_chapter_sha256": digest(restored), "matches_original_whole_chapter": True,
                  "changed_from": old_phrase, "changed_to": new_phrase,
                  "diff": "".join(difflib.unified_diff(old_sound_bytes.decode().splitlines(True),
                                                        new_sound_bytes.decode().splitlines(True),
                                                        fromfile=old_sound["snapshot"], tofile=new_sound["snapshot"])),
                  "python_block_count": len(new_blocks), "python_blocks_identical": True,
                  "figures_unchanged": old_sound["figures_sha256"] == new_sound["figures_sha256"]},
    "chapter20_referenced_in_original_sources": False,
    "reviewer_assessment": "W.7本身與索引承諾不變；19.22只修正門檻所處階段的文字，未改表列codec/Talker內嵌計算、程式區塊、數字門檻或來源身份。其餘原始來源與三項已登記artifact SHA均不變，誠實沿用原核對；只更新s_repo_28到實際當前wholefile SHA。",
    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": "CPU; hashes/byte comparison/AST parsing only"},
    "not_executed": ["No NN forward/backward, training, tuning, optimizer update, final-family inference, Modal, workflow, installation, network or publishing.",
                     "No rerun of unchanged historical CPU/REPL/uv/dry-run evidence; original artifacts are explicitly reused.",
                     "No new browser, Windows, macOS, MPS, GPU, or Colab verification."],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
