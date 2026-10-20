"""核對章檔差異、注意力算例與既有證據的版本。"""

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-3-6"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return {"command_argv": argv, "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(raw, name):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, heading in enumerate(headings) if heading[0].startswith(b"## " + name.encode() + b" "))
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    return raw[headings[index].start():end]


previous_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_3-6/technical-3-6-current-01.json"
previous_raw = previous_path.read_bytes()
assert digest(previous_raw) == "3475e81b42c9d3bcd15a42640b72a21eed23b7c65443b71bfb6a30d48f7da302"
previous = json.loads(previous_raw)
original_path = ROOT / previous["original_report"]["path"]
assert digest(original_path.read_bytes()) == previous["original_report"]["sha256"]
assert previous["issues"] == []
artifact_checks = []
for artifact in previous["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})

old_git = command(["git", "show", "ecf6dec71280:course/chapters/03.md"])
new_git = command(["git", "show", "c6b9339d2595:course/chapters/03.md"])
old_chapter = old_git["stdout"].encode()
current = (ROOT / "course/chapters/03.md").read_bytes()
assert digest(old_chapter) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert digest(current) == "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5"
assert current == new_git["stdout"].encode()
diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/chapters/03.md"])
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/3.6.md").read_bytes()
assert section(current, "3.6") == frozen == section(old_chapter, "3.6")
assert section(current, "3.7") == section(old_chapter, "3.7")
repository_checks = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        actual = digest((ROOT / source["path"]).read_bytes())
        if source["path"] != "course/chapters/03.md":
            assert actual == source["sha256"]
        repository_checks.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual})

run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T151820933582Z"
section_run = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/03.md#3.6" --execute',
    "extraction": json.loads((run_dir / "extraction.json").read_bytes()),
    "execution": json.loads((run_dir / "execution.json").read_bytes()),
    "environment": json.loads((run_dir / "environment.json").read_bytes()),
    "stdout": (run_dir / "stdout.txt").read_text(),
    "stderr": (run_dir / "stderr.txt").read_text(),
    "copied_file_sha256": {name: digest((run_dir / name).read_bytes()) for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]},
}
assert section_run["execution"]["exit_code"] == 0
assert section_run["extraction"]["source_file_sha256"] == digest(current)
assert section_run["extraction"]["source_sha256"] == previous["source_sha256"]
assert section_run["stdout"] == json.loads((BASE / "section-execution.json").read_bytes())["stdout"]
verification_run = command([str(ROOT / ".venv/bin/python"), "docs/technical-reviews/artifacts/review-20261019-3-6/verify.py"])
verification = json.loads(verification_run["stdout"])
assert verification == json.loads((BASE / "verification-output.json").read_bytes())

changed_section = section(current, "3.3").decode()
changed_fences = re.findall(r"```python\n(.*?)\n```", changed_section, re.S)
namespace = {"__name__": "__main__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for fence in changed_fences:
        exec(compile(fence, "course/chapters/03.md:query-key-example", "exec"), namespace)
has, need = namespace["has"], namespace["need"]
shared = has @ has.T
separate = need @ has.T
assert shared.tolist() == [[1.0, 0.0], [0.0, 1.0]]
assert separate.tolist() == [[0.0, 1.0], [1.0, 0.0]]
assert shared.argmax(dim=-1).tolist() == [0, 1]
new_clause = "本例 A、B 跟自己都得 1、彼此得 0，所以最高分都落在自己。"
assert new_clause in changed_section

png = ROOT / "outputs/course-review-20261019/figure-renders-04/ee44e229486282fde6009c41a6e771087b97576516c8133546d0d2438ea134b5.png"
assert digest(png.read_bytes()) == "b065b83424a78012d2be77bd81ff49673693a70a8284c858ecbbde5fc8536135"
figure = ROOT / "course/figures/rewrite-03-causal-results.svg"
assert digest(figure.read_bytes()) == previous["figure_sha256"][figure.relative_to(ROOT).as_posix()]
freeze = ROOT / "docs/course-review-20261019/freeze-07.json"
assert digest(freeze.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-6/recheck-current-02.py",
    "environment": verification["environment"],
    "previous_report": {"path": previous_path.relative_to(ROOT).as_posix(), "sha256": digest(previous_raw)},
    "original_report_sha256": digest(original_path.read_bytes()),
    "previous_issues": previous["issues"],
    "existing_artifacts": artifact_checks,
    "repository_sources": repository_checks,
    "git_versions": {"previous_commit": "ecf6dec71280", "current_commit": "c6b9339d2595", "previous_chapter_sha256": digest(old_chapter), "current_chapter_sha256": digest(current), "current_matches_requested_commit": True},
    "git_diff": diff,
    "section_unchanged": True,
    "next_section_unchanged": True,
    "source_sha256": digest(frozen),
    "freeze_sha256": digest(freeze.read_bytes()),
    "section_run": section_run,
    "independent_verification": verification_run,
    "independent_verification_identical_to_original": True,
    "changed_sentence_check": {
        "location": "course/chapters/03.md#3.3",
        "current_text": new_clause,
        "status": "verified",
        "executed_fences": len(changed_fences),
        "stdout": captured.getvalue(),
        "shared_scores": shared.tolist(),
        "separate_scores": separate.tolist(),
        "shared_highest_column": shared.argmax(dim=-1).tolist(),
        "derivation": "A=(1,0)、B=(0,1)，A·A=1×1+0×0=1，B·B=0×0+1×1=1，A·B=B·A=0。每列的自己分數1大於另一欄0，最高分確實落在自己。新句限定於這個指定算例，沒有把一般向量的最高分位置當成無條件保證。",
        "effect_on_target_section": "改動只在匹配說明，本節與其引用的精簡CausalAttention都逐位元組不變。"
    },
    "figure_inspection": {"png": png.relative_to(ROOT).as_posix(), "png_sha256": digest(png.read_bytes()), "svg_sha256": digest(figure.read_bytes()), "viewed_again": True, "details": "再次以view_image看過指定PNG；四列query的有效卡、灰色未來卡、比例1／1/2／1/3／1/4，以及輸出[1,10]／[1.5,15]／[2,20]／[2.5,25]均與實跑吻合；改最後V時前三輸出不變。"},
    "claims_revalidated": [claim["id"] for claim in previous["claims"]],
    "independence": {"other_review_contents_read": False, "allowed_evidence_read": [artifact["path"] for artifact in previous["artifacts"]], "details": "只讀自己的上一份報告、明示證據、指定凍結原文／工具／圖與一般教材程式；工具執行目錄僅用於複製本次輸出，未讀其他outputs。"},
    "result": "全部核對通過；原30項主張仍成立，改寫的匹配算例敘述與手算及實跑一致。"
}, ensure_ascii=False, indent=2, allow_nan=False))
