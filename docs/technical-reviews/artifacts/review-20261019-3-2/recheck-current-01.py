"""核對目前教材來源、既有證據與有限樣本敘述，另存本次執行收據。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
BASE = Path(__file__).parent
ORIGINAL = Path("docs/course-review-20261019/reviews/rv1019_technical_3-2/technical-3-2-original-01.json")
CHAPTER = Path("course/chapters/03.md")
FROZEN = Path("docs/course-review-20261019/freeze-01/sources/3.2.md")
FREEZE = Path("docs/course-review-20261019/freeze-05.json")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    completed = subprocess.run(argv, capture_output=True, text=True, check=True)
    return {"argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


original_raw = ORIGINAL.read_bytes()
assert digest(original_raw) == "da54b239c97a73343b7ac79b451511dc5c23f1ee9157f79a0c19fd91e4848c04"
original = json.loads(original_raw)
assert original["issues"] == []
old_artifacts = []
for artifact in original["artifacts"]:
    actual = digest(Path(artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    old_artifacts.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_check = BASE / "self-check-01.txt"
assert digest(old_check.read_bytes()) == "a9d26af0c12155a893d8c4f0aa434bdf1ab7bbdf7319d909e3474355f220cc3d"

chapter = CHAPTER.read_bytes()
assert digest(chapter) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert digest(FROZEN.read_bytes()) == "8b53aaede279bed9f24de8c59048d24ef462e42f448c1ada2db8f0001a13bf27"
assert digest(FREEZE.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
old_chapter = subprocess.run(["git", "show", "5ef56bc756f3:course/chapters/03.md"], capture_output=True, check=True).stdout
new_commit_chapter = subprocess.run(["git", "show", "3af7f13ef959:course/chapters/03.md"], capture_output=True, check=True).stdout
assert digest(old_chapter) == "df0709c102dba36fa89b3c88904f55172339a61c15c4a0583484b9b919bd2ee7"
assert chapter == new_commit_chapter
change = command(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", CHAPTER.as_posix()])


def section(raw, lesson):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, header in enumerate(headers) if header[0].startswith(("## " + lesson + " ").encode()))
    return raw[headers[index].start():headers[index + 1].start()]


section_equalities = {}
for lesson in ("3.1", "3.2", "3.6"):
    assert section(chapter, lesson) == section(old_chapter, lesson)
    section_equalities[lesson] = {"unchanged": True, "sha256": digest(section(chapter, lesson))}
assert section(chapter, "3.2") == FROZEN.read_bytes()
before = "縮放後兩者都靠近 1，樣本有限所以不會恰好等於理論值。"
after = "縮放後兩者都靠近 1；有限樣本通常有抽樣誤差，所以不必恰好等於理論值。"
assert section(old_chapter, "3.5").decode().replace(before, after) == section(chapter, "3.5").decode()

section_run = command([sys.executable, "docs/review-tools/section_facts.py", "course/chapters/03.md#3.2", "--execute"])
summary = json.loads(section_run["stdout"])
assert summary["source_sha256"] == digest(FROZEN.read_bytes())
assert summary["execution_exit_code"] == 0
assert summary["python_fences"] == 1 and summary["svg_references"] == 0
run_directory = Path(summary["artifact_directory"])
assert run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
run_files = []
for filename in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_directory / filename).read_bytes()
    run_files.append({"filename": filename, "sha256": digest(raw), "content_utf8": raw.decode("utf-8")})
assert digest((run_directory / "stdout.txt").read_bytes()) == digest((BASE / "section-stdout.txt").read_bytes())
assert digest((run_directory / "fence-1.py").read_bytes()) == digest((BASE / "section-fence-1.py").read_bytes())

variants = command([sys.executable, (BASE / "verify_examples.py").as_posix()])
variant_data = json.loads(variants["stdout"])
assert variant_data == json.loads((BASE / "verify-examples.json").read_bytes())

import torch

scaling_body = section(chapter, "3.5")
scaling_code = re.findall(rb"```python\n(.*?)```", scaling_body, re.S)
assert len(scaling_code) == 1
namespace = {"__name__": "__main__"}
scaling_stdout = io.StringIO()
with contextlib.redirect_stdout(scaling_stdout):
    exec(compile(scaling_code[0], "course/chapters/03.md#scaling-example", "exec"), namespace)

# 每個值都能由六十四個正負一乘積加總、再除以八得到。
scaled = torch.tensor([1.0, -1.0] * 497 + [1.5, -1.5, 0.5, -0.5, 0.0, 0.0], dtype=torch.float32)
raw_scores = scaled * 8
positive_counts = ((64 + raw_scores) / 2).to(torch.int64)
keys = torch.where(torch.arange(64)[None, :] < positive_counts[:, None], 1.0, -1.0)
queries = torch.ones_like(keys)
reconstructed = (queries * keys).sum(dim=-1) / math.sqrt(64)
assert torch.equal(reconstructed, scaled)
assert len(scaled) == 1000
scalar_values = scaled.tolist()
mean = math.fsum(scalar_values) / len(scalar_values)
sum_squared_deviations = math.fsum((value - mean) ** 2 for value in scalar_values)
std_reference = math.sqrt(sum_squared_deviations / (len(scalar_values) - 1))
assert mean == 0 and sum_squared_deviations == 999
assert std_reference == scaled.std(correction=1).item() == 1.0

for artifact in original["artifacts"]:
    assert digest(Path(artifact["path"]).read_bytes()) == artifact["sha256"]
assert digest(ORIGINAL.read_bytes()) == digest(original_raw)
assert digest(CHAPTER.read_bytes()) == digest(chapter)

receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-2/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"},
    "original_report": {"path": ORIGINAL.as_posix(), "sha256": digest(original_raw), "issues": original["issues"]},
    "prior_artifact_checks": old_artifacts,
    "prior_self_check": {"path": old_check.as_posix(), "sha256": digest(old_check.read_bytes()), "unchanged": True},
    "current_repository_source": {"path": CHAPTER.as_posix(), "sha256": digest(chapter), "old_sha256": digest(old_chapter), "matches_named_new_commit": True},
    "source_freeze": {"path": FREEZE.as_posix(), "sha256": digest(FREEZE.read_bytes())},
    "git_diff": change,
    "section_equalities": section_equalities,
    "changed_context": {"section": "匹配特徵越多，為什麼要除以平方根？", "before": before, "after": after, "only_change_in_context": True, "unchanged_claim_support": "前節手填權重、受審小節全部原文與後續遮罩均未變；縮放小節唯一修改為有限樣本的敘述，scores/math.sqrt(d) 未變。"},
    "section_run": {"command": section_run, "summary": summary, "copied_files": run_files},
    "variants_rerun": variants,
    "variants_match_prior": True,
    "changed_context_verification": {
        "scaling_example_code_sha256": digest(scaling_code[0]),
        "scaling_example_stdout": scaling_stdout.getvalue(),
        "scaling_example_configuration": {"seed": 42, "dimensions": [4, 64], "samples_per_dimension": 1000, "dtype": "torch.float32", "device": "cpu"},
        "finite_exact_example": {"samples": 1000, "dimension": 64, "scaled_values_recipe": "[1,-1] 重複 497 次，再加 [1.5,-1.5,0.5,-0.5,0,0]", "query": "每格均為 +1", "key_construction": "各列前 (64+8*scaled_value)/2 格為 +1，其餘為 -1", "mean": mean, "sum_squared_deviations": sum_squared_deviations, "correction": 1, "denominator": 999, "std_reference": std_reference, "std_torch": scaled.std(correction=1).item()},
        "derivation": "獨立正負一乘積各自變異數為 1，六十四格和的理論變異數為 64，除以八後理論標準差為 1。有限樣本統計量取決於抽到的值；不保證和母體標準差相等，也不排除相等。所構造的一千個可達縮放分數平均為 0、平方偏差和為 999，以樣本分母 999 計算恰得標準差 1。原碼 seed=42 的兩次有限抽樣則顯示與 1 的小偏離。因此新句保留抽樣誤差的限制，並避免斷言有限樣本必不相等。",
        "scope": "只核對指定差異的這一句與其尺度示範，不替整個連結小節做正式審閱；沒有訓練或品質實驗。"
    },
    "all_original_files_unchanged_after_execution": True,
    "result": "全部斷言通過；受審小節與凍結原文相同，原節及變體結果吻合，章檔唯一修改不破壞既有主張的依據，有限樣本的新敘述成立。"
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
