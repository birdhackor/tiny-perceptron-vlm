"""核對既有證據完整性、引用片段與目前程式的執行結果。"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_10-8/technical-10-8-original-01.json"
EXPECTED_ORIGINAL = "5b1028df7b162dd6d8382d8ee26965249b585a6bcaefb2155565ea8501386800"
OLD_REF = "5ef56bc756f3"
NEW_REF = "3af7f13ef959"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, lesson):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    matches = [i for i, h in enumerate(headers) if h[0].startswith(("## " + lesson + " ").encode())]
    assert len(matches) == 1
    index = matches[0]
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def invoke(argv):
    process = subprocess.run(argv, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, check=False)
    record = {"command_argv": argv, "exit_code": process.returncode, "stdout": process.stdout.decode(), "stderr": process.stderr.decode()}
    assert process.returncode == 0, record
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-run", type=Path, required=True)
    args = parser.parse_args()
    original_raw = (ROOT / ORIGINAL).read_bytes()
    assert digest(original_raw) == EXPECTED_ORIGINAL
    original = json.loads(original_raw)
    result = {"checked_at": datetime.now(UTC).isoformat(), "original_report": {"path": ORIGINAL, "sha256": digest(original_raw)}, "original_issues": original["issues"], "repository_sources": [], "reused_artifacts": [], "dependency_diffs": [], "dependency_section_checks": []}
    assert original["issues"] == []
    for source in original["sources"]:
        if source["kind"] != "repository_code":
            continue
        raw = (ROOT / source["path"]).read_bytes()
        result["repository_sources"].append({"source_id": source["id"], "path": source["path"], "original_sha256": source["sha256"], "current_sha256": digest(raw), "unchanged": digest(raw) == source["sha256"]})
    for artifact in original["artifacts"]:
        actual = digest((ROOT / artifact["path"]).read_bytes())
        assert actual == artifact["sha256"]
        result["reused_artifacts"].append({"id": artifact["id"], "path": artifact["path"], "sha256": actual, "unchanged": True})
    old_check = ROOT / "docs/technical-reviews/artifacts/review-20261019-10-8/self-check.txt"
    assert digest(old_check.read_bytes()) == "65575605eead9f1a5be4614eb6f916e50fbc2d050ec7f7d89bf248a0af1ebbab"
    expected_changes = {"course/chapters/10.md", "course/chapters/12.md"}
    assert {source["path"] for source in result["repository_sources"] if not source["unchanged"]} == expected_changes
    for path, lessons in [("course/chapters/10.md", ["10.5", "10.6", "10.7", "10.8"]), ("course/chapters/12.md", ["12.9"])]:
        diff = invoke(["git", "diff", OLD_REF, NEW_REF, "--", path])
        result["dependency_diffs"].append(diff)
        versions = {"original_commit": invoke(["git", "show", OLD_REF + ":" + path]), "updated_commit": invoke(["git", "show", NEW_REF + ":" + path])}
        old_bytes = versions["original_commit"]["stdout"].encode()
        new_bytes = versions["updated_commit"]["stdout"].encode()
        current = (ROOT / path).read_bytes()
        source = next(s for s in original["sources"] if s.get("path") == path)
        assert digest(old_bytes) == source["sha256"]
        assert new_bytes == current
        for lesson in lessons:
            values = {"original_commit": section(old_bytes, lesson), "updated_commit": section(new_bytes, lesson), "worktree": section(current, lesson)}
            assert len(set(values.values())) == 1
            result["dependency_section_checks"].append({"path": path, "lesson_id": lesson, "section_sha256": {key: digest(value) for key, value in values.items()}, "byte_identical": True, "git_show_commands": {key: versions[key]["command_argv"] for key in versions}})
    freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
    assert digest(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/10.8.md"
    assert section((ROOT / "course/chapters/10.md").read_bytes(), "10.8") == frozen.read_bytes()
    assert digest(frozen.read_bytes()) == original["source_sha256"]
    result["freeze"] = {"path": str(freeze_path.relative_to(ROOT)), "sha256": digest(freeze_path.read_bytes()), "source_sha256": digest(frozen.read_bytes()), "worktree_matches_frozen": True}
    run = args.section_run.resolve()
    assert run.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
    result["section_execution"] = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py \"course/chapters/10.md#10.8\" --execute", "files": {name: {"sha256": digest((run / name).read_bytes()), "text": (run / name).read_text()} for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt")}}
    extraction = json.loads(result["section_execution"]["files"]["extraction.json"]["text"])
    execution = json.loads(result["section_execution"]["files"]["execution.json"]["text"])
    assert extraction["source_sha256"] == original["source_sha256"]
    assert extraction["source_file_sha256"] == digest((ROOT / "course/chapters/10.md").read_bytes())
    assert extraction["figure_sha256"] == {} and execution["exit_code"] == 0
    behavior = invoke([str(ROOT / ".venv/bin/python"), "docs/technical-reviews/artifacts/review-20261019-10-8/verify_behavior.py"])
    result["behavior_execution"] = behavior
    parsed = json.loads(behavior["stdout"])
    previous = json.loads((ROOT / "docs/technical-reviews/artifacts/review-20261019-10-8/behavior-output.json").read_bytes())
    assert parsed == previous
    result["behavior_matches_original"] = True
    result["environment"] = parsed["environment"]
    result["environment"]["platform"] = platform.platform()
    result["claim_continuity"] = [{"claim_id": c["id"], "status": "verified", "statement_unchanged": True, "reason": "原文及相關引用片段未變；程式與既有來源的版本核對成立，本回合實跑輸出亦與原證據相同。"} for c in original["claims"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
