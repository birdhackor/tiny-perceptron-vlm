"""核對新凍結原文、不可變證據與修訂後的型別及提示主張。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_20-13/technical-20-13-original-01.json"
OLD = ROOT / "docs/course-review-20261019/freeze-01/sources/20.13.md"
CURRENT = ROOT / "docs/course-review-20261019/freeze-05/sources/20.13.md"
FREEZE = ROOT / "docs/course-review-20261019/freeze-06.json"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T101629761807Z"
EXPECTED = "43740f13edcbe520166a05a6551bba39539d32a1b5409771abdcda97ed9c4681"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_section(path, identifier):
    raw = path.read_bytes()
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, header in enumerate(headers):
        if header[0].startswith(("## " + identifier + " ").encode()):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
            return raw[header.start():end]
    raise ValueError("指定原文不存在")


report = json.loads(ORIGINAL.read_bytes())
assert digest(ORIGINAL) == "6584c14500a5026245c55777d06ec241984ab9008f039b5153446b24579d96f4"
assert digest(CURRENT) == EXPECTED
assert digest(FREEZE) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
actual_section = frozen_section(ROOT / "course/chapters/20.md", "20.13")
assert actual_section == CURRENT.read_bytes()
artifacts = []
for item in report["artifacts"]:
    actual = digest(ROOT / item["path"])
    assert actual == item["sha256"]
    artifacts.append({"path": item["path"], "sha256": actual, "unchanged": True})
old_verification = json.loads((ARTIFACT / "verification.json").read_bytes())
inputs = []
for path, expected in old_verification["input_sha256"].items():
    actual = digest(ROOT / path)
    assert actual == expected
    inputs.append({"path": path, "sha256": actual, "unchanged": True})
repository_sources = []
for source in report["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(ROOT / source["path"])
        assert actual == source["sha256"]
        repository_sources.append({"path": source["path"], "sha256": actual, "unchanged": True})

command = [str(ROOT / ".venv/bin/python"), str(ARTIFACT / "verify.py")]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
assert completed.returncode == 0, completed.stderr
replayed = json.loads(completed.stdout)
for key in old_verification:
    if key not in {"git_revision", "environment"}:
        assert replayed[key] == old_verification[key], key

fence_run = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/20.md#20.13" --execute',
             "copied_files": {}}
for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
    path = RUN / name
    raw = path.read_bytes()
    fence_run["copied_files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(),
        "contents": json.loads(raw) if name.endswith(".json") else raw.decode()}
assert fence_run["copied_files"]["extraction.json"]["contents"]["source_sha256"] == EXPECTED
assert fence_run["copied_files"]["extraction.json"]["contents"]["figure_sha256"] == {}
assert fence_run["copied_files"]["execution.json"]["contents"]["exit_code"] == 0
assert fence_run["copied_files"]["stdout.txt"]["contents"] == "期待內容指紋 45de0d2401bf\n收到內容指紋 b43173684456\n內容符合清單 False\n"

string_checks = []
for text in ["abc", "繁體字"]:
    try:
        hashlib.sha256(text)
    except TypeError as error:
        string_checks.append({"input": text, "type": type(text).__name__, "accepted": False, "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("str 不應直接通過介面")

manifest_path = ROOT / "docs/natural-assistant/v4/manifest.json"
manifest = json.loads(manifest_path.read_bytes())
scenes = [row for row in manifest["rows"] if row["split"] == "test" and row["task"] == "scene" and row["references"]["qa_type"] == "scene"]
prompts = []
for row in scenes:
    rubric = row["references"]["rubric"]
    assert "concise Traditional-Chinese" in rubric
    assert row["references"]["target_language"] == "zh-Hant"
    prompts.append({"id": row["id"], "user": row["user"], "system": row.get("system"),
                    "rubric": rubric, "target_language": row["references"]["target_language"],
                    "explicit_one_or_two_sentences": any(term in row["user"] for term in ["一至兩句", "一兩句"])})
assert len(prompts) == 42
sentence_count = sum(row["explicit_one_or_two_sentences"] for row in prompts)
assert sentence_count == 37

diff_command = ["git", "diff", "--no-index", str(OLD.relative_to(ROOT)), str(CURRENT.relative_to(ROOT))]
diff = subprocess.run(diff_command, cwd=ROOT, capture_output=True, text=True)
assert diff.returncode == 1
precondition = frozen_section(ROOT / "course/chapters/05.md", "5.11")
assert "不能直接接受字串" in precondition.decode()

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-13/recheck-current-01.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu; standard library only"},
    "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(ORIGINAL)},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": digest(FREEZE)},
    "source_snapshot": {"path": str(CURRENT.relative_to(ROOT)), "sha256": digest(CURRENT), "equals_current_chapter_section": True},
    "precondition": {"source": "course/chapters/05.md#5.11", "section_sha256": hashlib.sha256(precondition).hexdigest(),
                     "relevant_text": "hashlib.sha256 不能直接接受字串；先用 encode(utf-8) 轉成 bytes"},
    "diff": {"command": diff_command, "returncode": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr},
    "original_artifacts": artifacts,
    "unchanged_previous_inputs": inputs,
    "repository_sources": repository_sources,
    "replay": {"command_argv": command, "returncode": completed.returncode, "stderr": completed.stderr,
               "result": replayed, "same_semantic_result_as_original": True},
    "section_execution": fence_run,
    "string_inputs": string_checks,
    "photo_summary": {"count": len(prompts), "explicit_one_or_two_sentence_prompts": sentence_count,
                      "other_prompts": len(prompts) - sentence_count,
                      "all_rubrics_require_concise_traditional_chinese": True,
                      "prompt_counts": dict(Counter(row["user"] for row in scenes)), "rows": prompts},
    "scope": "完整重播先前的計數、CER、內容指紋及下載核對；重新核對改寫主張。沒有新模型推論、訓練、圖片語意評分或錄音聆聽。"
}
print(json.dumps(output, ensure_ascii=False, indent=2))
