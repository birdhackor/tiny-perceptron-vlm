"""原審閱者複查：核對引用版本、證據保存與本節程式，不改寫原始紀錄。"""

import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    return digest((ROOT / path).read_bytes())


def execute(argv):
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    return {"command_argv": argv, "returncode": completed.returncode,
            "stdout": completed.stdout.decode(), "stderr": completed.stderr.decode()}


def section(raw, identifier):
    pattern = rb"(?m)^## " + re.escape(identifier.encode()) + rb" [^\r\n]*"
    match = re.search(pattern, raw)
    assert match is not None
    following = re.search(rb"(?m)^## ", raw[match.end():])
    end = match.end() + following.start() if following else len(raw)
    return raw[match.start():end]


def section_at(lines, index):
    headings = [line for line in lines[:index + 1] if line.startswith("## ")]
    return headings[-1].strip() if headings else "章前導言"


original_path = "docs/course-review-20261019/reviews/rv1019_technical_19-5/technical-19-5-original-01.json"
original_sha = "077339b6d7d5e908796ce1c22013998dd2a20a9ed75ba12d9243cadfed3628cc"
assert file_hash(original_path) == original_sha
original = json.loads((ROOT / original_path).read_bytes())
assert original["issues"] == []
expected_changed = {
    "course/chapters/19.md": ("6779026b198306f16b3235b035d742703e4e25f789f27603d6eebfa6f51db679", ["19.5", "19.12"]),
    "course/chapters/07.md": ("4b02242d4c281a8a0554a3a87bdc898c040bb7afa3250dc8c055184407952174", ["7.17"]),
    "course/chapters/13.md": ("e1e11b7d73dcf9ca366e196ccebccbedbbc50d651ddd7e80665eefa8755076f3", ["13.17"]),
}
sources = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        current_sha = file_hash(source["path"])
        changed = source["sha256"] != current_sha
        assert changed == (source["path"] in expected_changed)
        sources.append({"source_id": source["id"], "path": source["path"],
                        "original_sha256": source["sha256"], "current_sha256": current_sha,
                        "unchanged": not changed})
artifacts = []
for artifact in original["artifacts"]:
    current_sha = file_hash(artifact["path"])
    assert current_sha == artifact["sha256"]
    artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": current_sha, "unchanged": True})
old_self_check = "docs/technical-reviews/artifacts/review-20261019-19-5/self-check.txt"
assert file_hash(old_self_check) == "ef2d7e7ce5e6678d57bd4920da17235fd7c10c268e578ab8962822b3370f22cc"
old_result = json.loads((ROOT / BASE.relative_to(ROOT) / "verification.json").read_bytes())
cpu = old_result["historical_cpu_generation_audit"]
assert file_hash(cpu["receipt_path"]) == cpu["receipt_sha256"]
chapters = []
for path, (expected_sha, section_ids) in expected_changed.items():
    before = subprocess.run(["git", "show", "5ef56bc756f3:" + path], capture_output=True, check=True).stdout
    after = subprocess.run(["git", "show", "3af7f13ef959:" + path], capture_output=True, check=True).stdout
    current = (ROOT / path).read_bytes()
    original_source = next(source for source in sources if source["path"] == path)
    assert digest(before) == original_source["original_sha256"]
    assert digest(after) == digest(current) == expected_sha
    diff = execute(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", path])
    assert diff["returncode"] == 0
    section_checks = []
    for identifier in section_ids:
        earlier, now = section(before, identifier), section(current, identifier)
        assert earlier == now
        section_checks.append({"section": identifier, "heading": now.decode().splitlines()[0],
                               "original_sha256": digest(earlier), "current_sha256": digest(now),
                               "bytes": len(now), "identical_bytes": True})
    old_lines, new_lines = before.decode().splitlines(keepends=True), current.decode().splitlines(keepends=True)
    changed_blocks = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        heading = section_at(new_lines, c)
        assert not any(re.match(r"^## " + re.escape(identifier) + r" ", heading) for identifier in section_ids)
        changed_blocks.append({"heading": heading, "before": "".join(old_lines[a:b]),
                               "after": "".join(new_lines[c:d]),
                               "impact": "不在本頁或原報告明確引用的小節內；不改變本次沿用主張的支持內容。"})
    chapters.append({"path": path, "original_sha256": digest(before), "current_sha256": digest(current),
                     "diff": diff, "referenced_sections": section_checks, "changed_blocks": changed_blocks})
frozen_path = "docs/course-review-20261019/freeze-01/sources/19.5.md"
frozen = (ROOT / frozen_path).read_bytes()
assert digest(frozen) == original["source_sha256"]
assert section((ROOT / "course/chapters/19.md").read_bytes(), "19.5") == frozen
freeze_path = "docs/course-review-20261019/freeze-05.json"
assert file_hash(freeze_path) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(page for page in freeze["pages"] if page["page_id"] == "19.5")
assert page["source_sha256"] == original["source_sha256"] and page["figures_sha256"] == {}
run = execute([".venv/bin/python", "docs/review-tools/section_facts.py", "course/chapters/19.md#19.5", "--execute"])
assert run["returncode"] == 0, run
info = json.loads(run["stdout"])
assert info["source_sha256"] == original["source_sha256"] and info["python_fences"] == 3 and info["svg_references"] == 0
run_dir = Path(info["artifact_directory"]).resolve()
assert run_dir.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
run_files = {}
for name in ["stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json"]:
    raw = (run_dir / name).read_bytes()
    run_files[name] = {"sha256": digest(raw), "content": raw.decode()}
old_run = json.loads((BASE / "section-execution.json").read_bytes())
assert run_files["stdout.txt"]["content"] == old_run["tool_files"]["stdout.txt"]["content"]
assert run_files["stderr.txt"]["content"] == ""
environment = json.loads(run_files["environment.json"]["content"])
old_environment = json.loads(old_run["tool_files"]["environment.json"]["content"])
for name in ["python", "torch", "torch_git_version", "device_requested", "cuda_build", "cuda_available"]:
    assert environment[name] == old_environment[name]
assert environment["attempted_fences"] == [1, 2, 3] and environment["guard_events"] == []
namespace = {}
for block in re.findall(r"```python\n(.*?)```", frozen.decode(), re.S):
    exec(compile(block, "frozen_section", "exec"), namespace)
row = json.loads(json.dumps(namespace["row"]))
row["messages"][0]["role"] = "assistant"
changed_role = namespace["LibraryEncoder"](namespace["tok"], ".").encode(row)
role_targets = namespace["tok"].decode([value for value in changed_role["labels"] if value != -100], False)
assert role_targets == old_result["role_exercise"]
assessments = []
changed_source_ids = {source["source_id"] for source in sources if not source["unchanged"]}
for claim in original["claims"]:
    affected = sorted({ref["source_id"] for ref in claim["evidence"]} & changed_source_ids)
    assessments.append({"claim_id": claim["id"], "kind": claim["kind"], "status": "verified",
                        "statement_unchanged": True, "changed_whole_file_sources": affected,
                        "basis": "明確引用的小節逐位元組未變，已重讀並核對仍支持原主張；其他證據與實作雜湊未變。" if affected else "頁面、相關實作及沿用證據未變；原核對仍成立，純文字程式另已重跑。"})
receipt = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-5/recheck-current-01.py",
           "created_at": datetime.now(UTC).isoformat(), "original_report": {"path": original_path, "sha256": original_sha},
           "original_issue_count": 0, "resolved_issue_ids": [], "source_audit": sources,
           "artifact_audit": artifacts, "old_self_check_unchanged": True,
           "historical_cpu_receipt_unchanged": {"path": cpu["receipt_path"], "sha256": cpu["receipt_sha256"]},
           "chapter_comparisons": chapters, "source_sha256": digest(frozen), "changed_page_sentences": [],
           "section_command": run, "section_run": info, "copied_tool_files": run_files,
           "environment": {"python": environment["python"], "torch": environment["torch"],
                           "torch_git_version": environment["torch_git_version"], "device": "cpu"},
           "role_exercise": role_targets, "claim_assessments": assessments,
           "reused_evidence_scope": "原審實讀論文、固定資料包、字表重建及完整目標核對沿用原 artifact；本回合不重新下載或執行會改寫舊檔的 verify.py。"}
output = BASE / "recheck-current-01.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"receipt": output.relative_to(ROOT).as_posix(), "sha256": file_hash(output.relative_to(ROOT)),
                  "verdict": "pass", "claims": len(assessments), "changed_page_sentences": 0,
                  "chapter_changes": len(chapters), "unchanged_referenced_sections": sum(len(item["referenced_sections"]) for item in chapters),
                  "old_artifacts_unchanged": len(artifacts), "section_execution_exit_code": info["execution_exit_code"]}, ensure_ascii=False))
