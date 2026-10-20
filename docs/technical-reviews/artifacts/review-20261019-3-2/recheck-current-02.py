"""以現行來源及實際執行結果核對原節與指定上下文差異。"""

import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
BASE = Path(__file__).parent
PREVIOUS = Path("docs/course-review-20261019/reviews/rv1019_technical_3-2/technical-3-2-current-01.json")
ORIGINAL = PREVIOUS.with_name("technical-3-2-original-01.json")
CHAPTER = Path("course/chapters/03.md")
FROZEN = Path("docs/course-review-20261019/freeze-01/sources/3.2.md")
FREEZE = Path("docs/course-review-20261019/freeze-07.json")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
    return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def section(raw, lesson):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, header in enumerate(headers) if header[0].startswith(("## " + lesson + " ").encode()))
    return raw[headers[index].start():headers[index + 1].start()]


previous_bytes = PREVIOUS.read_bytes()
assert digest(previous_bytes) == "08bde6e04d100c6dd498374bc21c237fab6474cdcb19ecfaaaf99dc0f3d577d9"
assert digest(ORIGINAL.read_bytes()) == "da54b239c97a73343b7ac79b451511dc5c23f1ee9157f79a0c19fd91e4848c04"
previous = json.loads(previous_bytes)
assert previous["issues"] == []
preserved = [{"path": PREVIOUS.as_posix(), "sha256": digest(previous_bytes)}, {"path": ORIGINAL.as_posix(), "sha256": digest(ORIGINAL.read_bytes())}]
for artifact in previous["artifacts"]:
    actual = digest(Path(artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    preserved.append({"path": artifact["path"], "sha256": actual})
for name in ("self-check-01.txt", "self-check-current-01.txt"):
    path = BASE / name
    assert digest(path.read_bytes()) == "a9d26af0c12155a893d8c4f0aa434bdf1ab7bbdf7319d909e3474355f220cc3d"
    preserved.append({"path": path.as_posix(), "sha256": digest(path.read_bytes())})

chapter = CHAPTER.read_bytes()
assert digest(chapter) == "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5"
assert digest(FROZEN.read_bytes()) == "8b53aaede279bed9f24de8c59048d24ef462e42f448c1ada2db8f0001a13bf27"
assert digest(FREEZE.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
old_git = run(["git", "show", "ecf6dec71280:course/chapters/03.md"])
new_git = run(["git", "show", "c6b9339d2595:course/chapters/03.md"])
old_chapter = old_git["stdout"].encode("utf-8")
assert digest(old_chapter) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert new_git["stdout"].encode("utf-8") == chapter
change = run(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", CHAPTER.as_posix()])
section_equalities = {}
for lesson in ("3.1", "3.2", "3.5", "3.6"):
    current_section = section(chapter, lesson)
    assert current_section == section(old_chapter, lesson)
    section_equalities[lesson] = {"unchanged": True, "sha256": digest(current_section)}
assert section(chapter, "3.2") == FROZEN.read_bytes()

before = "各位置向量長度相同時，跟自己的分數也最高（或並列最高）"
after = "本例 A、B 跟自己都得 1、彼此得 0，所以最高分都落在自己"
assert old_chapter.decode("utf-8").count(before) == 1
assert old_chapter.decode("utf-8").replace(before, after).encode("utf-8") == chapter
new_sentence = "一般來說，共用一份時 A 對 B 的分數一定等於 B 對 A 的分數；" + after + "。"
assert new_sentence in chapter.decode("utf-8")

section_command = run([sys.executable, "docs/review-tools/section_facts.py", "course/chapters/03.md#3.2", "--execute"])
summary = json.loads(section_command["stdout"])
assert summary["source_sha256"] == digest(FROZEN.read_bytes())
assert summary["execution_exit_code"] == 0 and summary["python_fences"] == 1 and summary["svg_references"] == 0
run_directory = Path(summary["artifact_directory"])
assert run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
copied_files = []
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_directory / name).read_bytes()
    copied_files.append({"filename": name, "sha256": digest(raw), "content_utf8": raw.decode("utf-8")})
assert (run_directory / "stdout.txt").read_bytes() == (BASE / "section-stdout.txt").read_bytes()
assert (run_directory / "fence-1.py").read_bytes() == (BASE / "section-fence-1.py").read_bytes()

variants = run([sys.executable, (BASE / "verify_examples.py").as_posix()])
assert json.loads(variants["stdout"]) == json.loads((BASE / "verify-examples.json").read_bytes())

context_section = section(chapter, "3.3")
context_title = context_section.splitlines()[0].decode("utf-8").split(" ", 2)[2]
fences = re.findall(rb"```python\n(.*?)```", context_section, re.S)
assert len(fences) == 2
context_stdout = io.StringIO()
namespace = {"__name__": "__main__"}
with contextlib.redirect_stdout(context_stdout):
    for raw in fences:
        exec(compile(raw, CHAPTER.as_posix() + "#" + context_title, "exec"), namespace)
has = namespace["has"]
need = namespace["need"]
shared = has @ has.T
separate = need @ has.T
assert torch.equal(shared, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
assert torch.equal(separate, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
assert torch.equal(shared, shared.T)
assert torch.equal(shared.argmax(dim=-1), torch.tensor([0, 1]))
manual = [[math.fsum(x * y for x, y in zip(a, b, strict=True)) for b in has.tolist()] for a in has.tolist()]
assert shared.tolist() == manual

for entry in preserved:
    assert digest(Path(entry["path"]).read_bytes()) == entry["sha256"]
assert digest(CHAPTER.read_bytes()) == digest(chapter)

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-2/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"},
    "previous_report": {"path": PREVIOUS.as_posix(), "sha256": digest(previous_bytes), "issues": previous["issues"]},
    "original_report": {"path": ORIGINAL.as_posix(), "sha256": digest(ORIGINAL.read_bytes())},
    "preserved_files": preserved,
    "current_repository_source": {"path": CHAPTER.as_posix(), "sha256": digest(chapter), "previous_sha256": digest(old_chapter), "matches_named_new_commit": True},
    "source_freeze": {"path": FREEZE.as_posix(), "sha256": digest(FREEZE.read_bytes())},
    "git_diff": change,
    "git_show_commands": [old_git["argv"], new_git["argv"]],
    "section_equalities": section_equalities,
    "changed_context": {"title": context_title, "before_clause": before, "after_clause": after, "new_sentence": new_sentence, "only_change_in_chapter": True},
    "section_run": {"command": section_command, "summary": summary, "copied_files": copied_files},
    "variants_rerun": variants,
    "variants_match_prior": True,
    "changed_context_verification": {
        "code_sha256": [digest(raw) for raw in fences],
        "stdout": context_stdout.getvalue(),
        "has": has.tolist(), "need": need.tolist(),
        "shared_scores": shared.tolist(), "separate_scores": separate.tolist(),
        "shared_scores_scalar_reference": manual,
        "highest_indices_per_query": shared.argmax(dim=-1).tolist(),
        "comparison_tolerance": "本例僅有 0 與 1，PyTorch float32 與純量點積精確相等。",
        "derivation": "對實數向量，共用同一份表示時 s_AB=sum_i(a_i*b_i)=sum_i(b_i*a_i)=s_BA，故匹配分數對稱。本例 a=[1,0]、b=[0,1]；a·a=1、b·b=1、a·b=b·a=0；每列唯一最高分均位於自己的位置。這是本例的確定性計算，不據此推論任意向量都以自己為最高分。",
        "scope": "指定修改句位在相鄰的角色說明小節；本次只核對該句與其算例，不替該相鄰小節做完整正式審閱。"
    },
    "all_previous_files_unchanged_after_execution": True,
    "result": "全部斷言通過；受審小節與凍結原文相同，原節與變體重跑一致，唯一修改句的對稱性及自己得一、彼此得零均成立。"
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
