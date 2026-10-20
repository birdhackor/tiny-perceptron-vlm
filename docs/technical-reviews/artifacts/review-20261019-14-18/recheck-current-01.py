"""保存複查輸出，核對未變教材與更新過的來源是否仍支持原主張。"""

import hashlib
import io
import json
import platform
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

out = Path(__file__).resolve().parent
root = out.parents[3]
original_path = root / "docs/course-review-20261019/reviews/rv1019_technical_14-18/technical-14-18-original-01.json"
run = root / "outputs/reviewer-tools/runs/20261020T084427291194Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, title):
    text = raw.decode()
    headers = list(re.finditer(r"(?m)^## [^\n]+", text))
    index = next(i for i, h in enumerate(headers) if title in h.group())
    start = headers[index].start()
    end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
    return text[start:end]


def command(args):
    done = subprocess.run(args, cwd=root, capture_output=True, check=True)
    return done.stdout


def write_new(name, data):
    with (out / name).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


original_raw = original_path.read_bytes()
assert sha(original_raw) == "b8b3b593c36f6eb6dbbe963d77bf3007bfedd0a77175fb50cbe24715f15c70fe"
original = json.loads(original_raw)
assert original["issues"] == []
old_artifacts = []
for artifact in original["artifacts"]:
    actual = sha((root / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    old_artifacts.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_check = root / "docs/technical-reviews/artifacts/review-20261019-14-18/self-check.txt"
assert sha(old_check.read_bytes()) == "c388d33dc868cb76aa5b5fd46ae768cc97cb36f8dec8159299a7be10ac724386"
old_artifacts.append({"path": str(old_check.relative_to(root)), "sha256": sha(old_check.read_bytes()), "unchanged": True})

frozen = (root / "docs/course-review-20261019/freeze-01/sources/14.18.md").read_bytes()
assert sha(frozen) == original["source_sha256"]
assert (run / "section.md").read_bytes() == frozen
freeze_path = root / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"

files = {}
for name in ["section.md", "fence-1.py", "bootstrap.py", "extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]:
    raw = (run / name).read_bytes()
    files[name] = {"sha256": sha(raw), "content": raw.decode()}
execution = json.loads(files["execution.json"]["content"])
environment = json.loads(files["environment.json"]["content"])
assert execution["exit_code"] == 0
old_bundle = json.loads((out / "section-execution.json").read_text())
assert files["stdout.txt"]["content"] == old_bundle["files"]["stdout.txt"]["content"]
assert files["fence-1.py"]["sha256"] == old_bundle["files"]["fence-1.py"]["sha256"]
stdout = io.StringIO()
namespace = {}
with redirect_stdout(stdout):
    exec(compile(files["fence-1.py"]["content"], "frozen-section.py", "exec"), namespace)
assert stdout.getvalue() == files["stdout.txt"]["content"]
rows = [{"name": name, "parameters": namespace["other"] + copies * namespace["block"], "rounds": rounds}
        for name, copies, rounds in namespace["designs"]]
assert [(r["parameters"], r["rounds"]) for r in rows] == [(1200000, 1), (1200000, 5), (5200000, 5)]
assert 200_000 == 200000
exercise = {"parameters": namespace["other"] + 2 * namespace["block"], "rounds": 2}
assert exercise == {"parameters": 2200000, "rounds": 2}
assert all(r["parameters"] != exercise["parameters"] and r["rounds"] != exercise["rounds"] for r in rows)

old_raw = command(["git", "show", "5ef56bc756f3:course/chapters/19.md"])
committed_raw = command(["git", "show", "3af7f13ef959:course/chapters/19.md"])
current_raw = (root / "course/chapters/19.md").read_bytes()
old_source = next(s for s in original["sources"] if s["id"] == "s-ch19")
assert sha(old_raw) == old_source["sha256"]
assert sha(current_raw) == "6779026b198306f16b3235b035d742703e4e25f789f27603d6eebfa6f51db679"
assert committed_raw == current_raw
heading = "什麼證據才足以把候選版叫成品？"
acceptance = section(current_raw, heading)
old_acceptance = section(old_raw, heading)
assert acceptance == old_acceptance
old_verification = json.loads((out / "verification.json").read_text())
assert sha(acceptance.encode()) == old_verification["prerequisites"]["course/chapters/19.md"]["section_sha256"]
quotes = [paragraph for paragraph in acceptance.split("\n\n") if any(needle in paragraph for needle in ["要交付成品", "若新成品沒有達主要門檻"])]
assert len(quotes) == 2
repo_delta = command(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/19.md"]).decode()
changed_lines = [line for line in repo_delta.splitlines() if (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---"))]
retained = []
for path, old in old_verification["prerequisites"].items():
    body = section((root / path).read_bytes(), old["heading"])
    retained.append({"path": path, "heading": old["heading"], "previous_section_sha256": old["section_sha256"],
                     "current_section_sha256": sha(body.encode()), "same_section": sha(body.encode()) == old["section_sha256"]})
assert all(item["same_section"] for item in retained)

bundle = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/14.md#14.18" --execute',
          "tool_run_directory": str(run.relative_to(root)), "files": files}
write_new("section-execution-current-01.json", bundle)

record = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(root)),
          "checked_at_utc": datetime.now(UTC).isoformat(),
          "environment": {"python": sys.version, "torch": str(environment["torch"]), "device": "cpu", "platform": platform.platform()},
          "original_report": {"path": str(original_path.relative_to(root)), "sha256": sha(original_raw)},
          "original_issue_ids": [], "resolved_issue_ids": [],
          "source_freeze": {"path": str(freeze_path.relative_to(root)), "sha256": sha(freeze_path.read_bytes())},
          "source_sha256": sha(frozen), "section_unchanged": True,
          "old_artifacts": old_artifacts, "rows": rows, "exercise": exercise,
          "stdout": stdout.getvalue(), "prerequisite_sections": retained,
          "repository_source": {"path": "course/chapters/19.md", "previous_sha256": sha(old_raw), "current_sha256": sha(current_raw),
                                "old_commit": "5ef56bc756f3", "current_commit": "3af7f13ef959", "current_matches_commit": True,
                                "read_heading": heading, "section_sha256": sha(acceptance.encode()),
                                "section_unchanged": True, "supporting_paragraphs": quotes},
          "diff_command": "git diff 5ef56bc756f3 3af7f13ef959 -- course/chapters/19.md", "diff_stdout": repo_delta,
          "changed_lines": changed_lines,
          "dependency_change_assessment": "差異均不在成品驗收小節；本節引用的事前門檻及未達標時標為候選或實驗的要求逐位元組未變。其他修改的數值或功能說明不屬於本頁的被引用主張，本次未代替那些小節的技術審閱。",
          "claim_recheck": [{"id": c["id"], "status": "verified", "statement_unchanged": True,
                             "basis": "已重讀凍結原文及原始判斷；正文與既有證據未變。" if c["id"] != "c29" else "已讀指定差異與現行成品驗收規則；相關段落完全相同。"} for c in original["claims"]],
          "independence": "只讀取原報告明確引用的既有 artifacts、本次指定工具新產生的輸出、教材與指定 Git 差異；未讀其他角色報告。",
          "result": "全部主張仍成立；原始程式重跑、參數算例與練習配對精確一致；來源檔修訂不影響被引用的成品驗收規則。"}
write_new("recheck-current-01.json", record)
print(json.dumps({"section_unchanged": True, "acceptance_section_unchanged": True, "claims": len(record["claim_recheck"]),
                  "original_issues": len(original["issues"]), "execution_exit_code": execution["exit_code"],
                  "current_repository_sha256": sha(current_raw)}, ensure_ascii=False))
