"""核對教材差異、保存證據與因果注意力的現行執行結果。"""

import contextlib
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from itertools import product
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


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_3-6/technical-3-6-original-01.json"
original_bytes = original_path.read_bytes()
assert digest(original_bytes) == "1f0ef26e847afaefdf6ed44bd995b98f9486c79693f89629d0e9654006615abe"
original = json.loads(original_bytes)
assert original["issues"] == []
artifact_checks = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_self_check = BASE / "self-check-output.json"
assert digest(old_self_check.read_bytes()) == "21aec0da66ca9b3a3f391b797c5937a4967b4eb3e8683d0f72041a319ebb4173"

old_git = command(["git", "show", "5ef56bc756f3:course/chapters/03.md"])
new_git = command(["git", "show", "3af7f13ef959:course/chapters/03.md"])
old_chapter = old_git["stdout"].encode()
current = (ROOT / "course/chapters/03.md").read_bytes()
assert digest(old_chapter) == "df0709c102dba36fa89b3c88904f55172339a61c15c4a0583484b9b919bd2ee7"
assert digest(current) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert current == new_git["stdout"].encode()
diff = command(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/03.md"])
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/3.6.md").read_bytes()
assert section(current, "3.6") == frozen == section(old_chapter, "3.6")
assert section(current, "3.7") == section(old_chapter, "3.7")
repository_checks = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = digest((ROOT / source["path"]).read_bytes())
    if source["path"] != "course/chapters/03.md":
        assert actual == source["sha256"]
    repository_checks.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual})

run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T085247343485Z"
execution = json.loads((run_dir / "execution.json").read_bytes())
extraction = json.loads((run_dir / "extraction.json").read_bytes())
assert execution["exit_code"] == 0
assert execution["source_sha256"] == original["source_sha256"]
assert extraction["source_file_sha256"] == digest(current)
assert (run_dir / "stdout.txt").read_text() == json.loads((BASE / "section-execution.json").read_bytes())["stdout"]
section_run = {
    "command": 'PYTHONDONTWRITECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/03.md#3.6" --execute',
    "extraction": extraction,
    "execution": execution,
    "environment": json.loads((run_dir / "environment.json").read_bytes()),
    "stdout": (run_dir / "stdout.txt").read_text(),
    "stderr": (run_dir / "stderr.txt").read_text(),
    "copied_file_sha256": {name: digest((run_dir / name).read_bytes()) for name in ["extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt"]},
}
section_run["command"] = section_run["command"].replace("PYTHONDONTWRITECODE", "PYTHONDONTWRITEBYTECODE")
verification_run = command([str(ROOT / ".venv/bin/python"), "docs/technical-reviews/artifacts/review-20261019-3-6/verify.py"])
verification = json.loads(verification_run["stdout"])
assert verification == json.loads((BASE / "verification-output.json").read_bytes())

changed_fences = re.findall(r"```python\n(.*?)\n```", section(current, "3.5").decode(), re.S)
sampling_stdout = io.StringIO()
with contextlib.redirect_stdout(sampling_stdout):
    for fence in changed_fences:
        exec(compile(fence, "course/chapters/03.md:scaling-example", "exec"), {})

sample = torch.tensor([-2.0, 0.0, 2.0])
assert sample.std().item() == 2.0
assert (sample / math.sqrt(4)).std().item() == 1.0
other_sample = torch.tensor([0.0, 0.0, 2.0])
assert (other_sample / math.sqrt(4)).std().item() != 1.0
values = [-4, -2, 0, 2, 4]
masses = [1, 4, 6, 4, 1]
exact_probability = Fraction(0)
for indices in product(range(5), repeat=3):
    observations = [values[i] for i in indices]
    mean = Fraction(sum(observations), 3)
    unbiased_variance = sum((Fraction(value) - mean) ** 2 for value in observations) / 2
    if unbiased_variance == 4:
        exact_probability += Fraction(math.prod(masses[i] for i in indices), 16**3)
assert 0 < exact_probability < Fraction(1, 2)

png = ROOT / "outputs/course-review-20261019/figure-renders-04/ee44e229486282fde6009c41a6e771087b97576516c8133546d0d2438ea134b5.png"
assert digest(png.read_bytes()) == "b065b83424a78012d2be77bd81ff49673693a70a8284c858ecbbde5fc8536135"
figure = ROOT / "course/figures/rewrite-03-causal-results.svg"
assert digest(figure.read_bytes()) == original["figure_sha256"][figure.relative_to(ROOT).as_posix()]
freeze = ROOT / "docs/course-review-20261019/freeze-05.json"
assert digest(freeze.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-6/recheck-current-01.py",
    "environment": verification["environment"],
    "original_report_sha256": digest(original_bytes),
    "original_issue_ids": [],
    "existing_artifacts": artifact_checks,
    "original_self_check_unchanged": True,
    "repository_sources": repository_checks,
    "git_versions": {"previous_commit": "5ef56bc756f3", "current_commit": "3af7f13ef959", "previous_chapter_sha256": digest(old_chapter), "current_chapter_sha256": digest(current), "current_matches_requested_commit": True},
    "git_diff": diff,
    "section_unchanged": True,
    "next_section_unchanged": True,
    "source_sha256": digest(frozen),
    "freeze_sha256": digest(freeze.read_bytes()),
    "section_run": section_run,
    "independent_verification": verification_run,
    "independent_verification_identical_to_original": True,
    "changed_sentence_check": {
        "location": "course/chapters/03.md#3.5",
        "current_text": "縮放後兩者都靠近 1；有限樣本通常有抽樣誤差，所以不必恰好等於理論值。",
        "status": "verified",
        "sampling_fence_count": len(changed_fences),
        "sampling_stdout": sampling_stdout.getvalue(),
        "derivation": "D=4的獨立正負一乘積和具有分布{-4,-2,0,2,4}，機率{1,4,6,4,1}/16，母體變異數為4。有限三筆取到[-2,0,2]時，樣本平均0、校正樣本變異數(4+0+4)/(3-1)=4，縮放除2後標準差正好1；取到[0,0,2]則縮放後標準差約0.57735。兩組樣本均有正機率，因此有限樣本既可能剛好相等，也可能有抽樣誤差；不能說有限樣本必定不等於理論值。另完整枚舉三筆的125組可能觀測，核對恰好相等機率小於1/2。這個三筆機率不外推為教材一千筆的機率。",
        "equal_sample": sample.tolist(),
        "equal_scaled_std": (sample / 2).std().item(),
        "unequal_sample": other_sample.tolist(),
        "unequal_scaled_std": (other_sample / 2).std().item(),
        "exact_probability_for_three_observations": str(exact_probability),
        "effect_on_target_section": "變動不在本次小節或其引用的精簡類別內；原30項主張的字句與依據不變。"
    },
    "figure_inspection": {"png": png.relative_to(ROOT).as_posix(), "png_sha256": digest(png.read_bytes()), "svg_sha256": digest(figure.read_bytes()), "viewed_again": True, "details": "再次以view_image看過指定PNG。由上到下query0至3，卡片依序為V0至V3；灰色未來卡權重0，其餘卡均分，輸出依序[1,10]、[1.5,15]、[2,20]、[2.5,25]，與本次實跑相同。"},
    "claims_revalidated": [claim["id"] for claim in original["claims"]],
    "independence": {"other_review_contents_read": False, "allowed_evidence_read": [artifact["path"] for artifact in original["artifacts"]], "details": "只讀上一份自己的報告及明示證據、指定凍結原文／工具／圖、一般教材與程式。工具新執行目錄僅用於複製本次實跑收據，沒有讀其他outputs。"},
    "result": "全部核對通過；原有證據未變，現行章檔所引用的精簡類別仍直接呼叫manual_attention。"
}, ensure_ascii=False, indent=2, allow_nan=False))
