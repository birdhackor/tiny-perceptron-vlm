"""保存章檔差異、來源一致性與加權讀取的複查收據。"""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
REPORT_PATH = "docs/course-review-20261019/reviews/rv1019_technical_3-1/technical-3-1-current-01.json"
FROZEN_SOURCE = "docs/course-review-20261019/freeze-01/sources/3.1.md"
FREEZE_PATH = "docs/course-review-20261019/freeze-07.json"
CHAPTER = "course/chapters/03.md"
BEFORE = "ecf6dec71280"
AFTER = "c6b9339d2595"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return (ROOT / path).read_bytes()


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## ([^ ]+) [^\r\n]+", raw))
    index = next(i for i, h in enumerate(headers) if h[1].decode() == identifier)
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def fences(raw):
    return re.findall(r"```python\n(.*?)```", raw.decode(), re.S)


def run(argv, env=None):
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, check=True)
    return {"command_argv": argv, "exit_code": proc.returncode,
            "stdout": proc.stdout.decode(), "stderr": proc.stderr.decode()}


def execute(code, name):
    namespace = {"__name__": "__main__", "torch": torch}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    return namespace, stdout.getvalue()


torch.set_default_device("cpu")
previous_raw = read(REPORT_PATH)
assert sha(previous_raw) == "2c4e8859185a6f7f4a9f2264c036e160cb3ebfb2c632a2a71b9f310b411876f0"
previous = json.loads(previous_raw)
assert previous["issues"] == []
original_record = previous["original_report"]
assert sha(read(original_record["path"])) == original_record["sha256"]
retained_artifacts = []
for item in previous["artifacts"]:
    actual = sha(read(item["path"]))
    assert actual == item["sha256"], "既有證據必須保持原樣"
    retained_artifacts.append({"path": item["path"], "expected_sha256": item["sha256"],
                               "actual_sha256": actual, "unchanged": True})

chapter_raw = read(CHAPTER)
assert sha(chapter_raw) == "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5"
before_receipt = run(["git", "show", f"{BEFORE}:{CHAPTER}"])
after_receipt = run(["git", "show", f"{AFTER}:{CHAPTER}"])
before_raw = before_receipt["stdout"].encode()
after_raw = after_receipt["stdout"].encode()
assert sha(before_raw) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert after_raw == chapter_raw
frozen = read(FROZEN_SOURCE)
assert sha(frozen) == "7bf9685d97502adcd7bacfddf20585dc33be84b0d1bf4aee09836674f6d95bdf"
freeze_raw = read(FREEZE_PATH)
assert sha(freeze_raw) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
page = next(p for p in json.loads(freeze_raw)["pages"] if p["page_id"] == previous["lesson_id"])
assert page["source_sha256"] == sha(frozen)
assert page["figures_sha256"] == {}
comparisons = []
for identifier in ["3.1", "3.2"]:
    old_section = section(before_raw, identifier)
    new_section = section(chapter_raw, identifier)
    assert old_section == new_section
    comparisons.append({"lesson_id": identifier, "previous_sha256": sha(old_section),
                         "current_sha256": sha(new_section), "bytes_equal": True})
assert section(chapter_raw, previous["lesson_id"]) == frozen

helper = "docs/review-tools/section_facts.py"
assert sha(read(helper)) == "d729cfba5a1bb772a3956cfbc85f8f13103efd740d93ac0358888724e3e9b814"
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
launcher = run([".venv/bin/python", helper, CHAPTER + "#" + previous["lesson_id"], "--execute"], env)
launch_metadata = json.loads(launcher["stdout"])
run_directory = Path(launch_metadata["artifact_directory"])
assert run_directory.is_relative_to(ROOT / "outputs/reviewer-tools/runs")
copied = {}
for name in ["stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "bootstrap.py"]:
    raw = (run_directory / name).read_bytes()
    copied[name] = {"sha256": sha(raw), "content_utf8": raw.decode()}
extraction = json.loads(copied["extraction.json"]["content_utf8"])
execution = json.loads(copied["execution.json"]["content_utf8"])
worker_environment = json.loads(copied["environment.json"]["content_utf8"])
assert execution["exit_code"] == 0
assert extraction["source_file_sha256"] == sha(chapter_raw)
assert extraction["source_sha256"] == sha(frozen)
assert extraction["figure_sha256"] == {}
assert worker_environment["guard_events"] == []
assert copied["stderr.txt"]["content_utf8"] == ""

weighted_code = fences(frozen)[0]
namespace, stdout = execute(weighted_code, "weighted-read")
values, weights, output = [namespace[name] for name in ["values", "weights", "output"]]
assert stdout == copied["stdout.txt"]["content_utf8"]
assert torch.equal(output, (weights[:, None] * values).sum(dim=0))
assert torch.equal(output, torch.matmul(weights, values))
assert list(output.shape) == [2]
assert not any(t.requires_grad for t in [values, weights, output])
values_before = values.clone()
reverse = torch.tensor([0.1, 0.9]) @ values
assert torch.allclose(reverse, torch.tensor([0.2, 3.6]))
assert torch.equal(values_before, values)
exact = {
    "original": [str(Decimal("0.9") * 2), str(Decimal("0.1") * 4)],
    "reversed": [str(Decimal("0.1") * 2), str(Decimal("0.9") * 4)],
    "contributions": [str(Decimal("0.9") * 1), str(Decimal("0.1") * 100)],
}
assert exact == {"original": ["1.8", "0.4"], "reversed": ["0.2", "3.6"], "contributions": ["0.9", "10.0"]}
calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(weighted_code)) if isinstance(n, ast.Call)]
assert set(calls) == {"torch.tensor", "print", "torch.allclose"}

next_code = fences(section(chapter_raw, "3.2"))[0]
old_numeric = json.loads(read("docs/technical-reviews/artifacts/review-20261019-3-1/verification.json"))
assert next_code == old_numeric["next_section"]["code"]
next_namespace, next_stdout = execute(next_code, "content-dependent-weights")
changed_query = torch.tensor([0.0, 1.0])
changed_weights = (changed_query @ next_namespace["keys"].T).softmax(dim=0)
assert changed_weights.argmax().item() == 1
layer = torch.nn.Linear(2, 1, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[-2.0, 4.0]]))
assert layer(torch.tensor([1.0, 1.0])).item() == 2.0
assert layer.weight.sum().item() == 2.0

matching_code = fences(section(chapter_raw, "3.3"))[1]
matching_namespace, matching_stdout = execute(matching_code, "shared-and-separate-representations")
has, need = matching_namespace["has"], matching_namespace["need"]
shared = has @ has.T
separate = need @ has.T
assert torch.equal(shared, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
assert torch.equal(separate, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
assert shared.argmax(dim=-1).tolist() == [0, 1]
assert torch.equal(shared, shared.T)

assert sha(read(CHAPTER)) == sha(chapter_raw), "核對期間章檔內容必須一致"
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-1/verify-current-02.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
                    "device": "cpu", "platform": platform.platform()},
    "previous_report": {"path": REPORT_PATH, "sha256": sha(previous_raw), "issue_count": len(previous["issues"])},
    "original_report": original_record,
    "retained_artifacts": retained_artifacts,
    "chapter": {"path": CHAPTER, "previous_sha256": sha(before_raw), "current_sha256": sha(chapter_raw), "current_equals_specified_commit": True},
    "git_show_receipts": [{"command_argv": r["command_argv"], "exit_code": r["exit_code"], "stderr": r["stderr"],
                           "stdout_sha256": sha(r["stdout"].encode())} for r in [before_receipt, after_receipt]],
    "diff": run(["git", "diff", BEFORE, AFTER, "--", CHAPTER]),
    "section_comparison": comparisons,
    "source_freeze": {"path": FREEZE_PATH, "sha256": sha(freeze_raw), "page": page},
    "section_execution": {"launcher": launcher, "copied_files": copied},
    "original_example": {"stdout": stdout, "dtype": str(values.dtype), "values": values.tolist(), "weights": weights.tolist(),
                         "output": output.tolist(), "output_shape": list(output.shape), "reverse_output": reverse.tolist(),
                         "values_unchanged": True, "requires_grad": {k:namespace[k].requires_grad for k in ["values", "weights", "output"]},
                         "calls": calls, "exact_decimal": exact},
    "next_example": {"code_unchanged": True, "stdout": next_stdout, "scores": next_namespace["scores"].tolist(),
                     "weights": next_namespace["weights"].tolist(), "changed_query_weights": changed_weights.tolist()},
    "linear_example": {"weight": layer.weight.tolist(), "weight_sum": layer.weight.sum().item(), "output": [2.0]},
    "changed_sentence_check": {
        "scope": "只核對指定章檔改句，不宣稱完整審閱其所在頁面", "code": matching_code, "stdout": matching_stdout,
        "shared_scores": shared.tolist(), "separate_scores": separate.tolist(), "highest_column_per_query": shared.argmax(dim=-1).tolist(),
        "status": "verified",
        "derivation": "人工例子令 A=(1,0)、B=(0,1)。A·A=1×1+0×0=1，B·B=0×0+1×1=1，A·B=1×0+0×1=0，B·A=0×1+1×0=0；故共用表示的分數表為 [[1,0],[0,1]]，每列最高值均在自己的欄。一般實向量的點積對稱性為 Σ_j a_j b_j=Σ_j b_j a_j，與此算例一致。新句只指這兩個人工向量，不宣稱所有向量都必然自匹配最高。",
    },
    "result": "所有斷言成立；本頁與其依賴的後續範例不變，既有七份證據不變。",
}
with (ARTIFACTS / "recheck-current-02.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
