"""核對章檔變更、保留證據與加權讀取的執行結果。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).resolve().parent
ORIGINAL_REPORT = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_3-1/technical-3-1-original-01.json"
RUN = ROOT / "outputs/reviewer-tools/runs/20261020T085300563677Z"
SOURCE = "course/chapters/03.md"
BEFORE = "5ef56bc756f3"
AFTER = "3af7f13ef959"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## ([^ ]+) [^\r\n]+", raw))
    index = next(i for i, item in enumerate(headers) if item[1].decode() == identifier)
    end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
    return raw[headers[index].start():end]


def code_fence(raw):
    return raw.decode().split("```python\n", 1)[1].split("```", 1)[0]


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": argv, "exit_code": result.returncode,
            "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def execute(code, name):
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    return namespace, stdout.getvalue()


torch.set_default_device("cpu")
original_raw = ORIGINAL_REPORT.read_bytes()
assert sha(original_raw) == "67ffde63daebe325dbc703b9d41e3e024d78ec3ab713490ab27b2176d3c248a5"
original = json.loads(original_raw)
assert original["issues"] == []
old_artifacts = []
for item in original["artifacts"]:
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"], "既有證據內容必須維持不變"
    old_artifacts.append({"path": item["path"], "expected_sha256": item["sha256"],
                          "actual_sha256": actual, "unchanged": True})

raw = (ROOT / SOURCE).read_bytes()
assert sha(raw) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
old_raw = subprocess.run(["git", "show", f"{BEFORE}:{SOURCE}"], cwd=ROOT,
                         capture_output=True, check=True).stdout
new_commit_raw = subprocess.run(["git", "show", f"{AFTER}:{SOURCE}"], cwd=ROOT,
                                capture_output=True, check=True).stdout
assert sha(old_raw) == "df0709c102dba36fa89b3c88904f55172339a61c15c4a0583484b9b919bd2ee7"
assert new_commit_raw == raw
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/3.1.md").read_bytes()
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
page = next(p for p in json.loads(freeze_path.read_bytes())["pages"] if p["page_id"] == original["lesson_id"])
assert page["source_sha256"] == sha(frozen)
assert page["figures_sha256"] == {}

section_checks = []
for identifier in ["3.1", "3.2"]:
    current_section = section(raw, identifier)
    previous_section = section(old_raw, identifier)
    assert current_section == previous_section
    if identifier == original["lesson_id"]:
        assert current_section == frozen
    section_checks.append({"lesson_id": identifier, "current_sha256": sha(current_section),
                           "previous_sha256": sha(previous_section), "bytes_equal": True})

current_code = code_fence(section(raw, "3.1"))
namespace, original_stdout = execute(current_code, "weighted-read")
values = namespace["values"]
weights = namespace["weights"]
output = namespace["output"]
assert output.shape == (2,)
assert torch.equal(output, (weights[:, None] * values).sum(dim=0))
assert torch.equal(output, torch.matmul(weights, values))
before_values = values.clone()
reverse = torch.tensor([0.1, 0.9]) @ values
assert torch.allclose(reverse, torch.tensor([0.2, 3.6]))
assert torch.equal(before_values, values)
assert not any(t.requires_grad for t in (values, weights, output))
assert [Decimal("0.9") * 2, Decimal("0.1") * 4] == [Decimal("1.8"), Decimal("0.4")]
assert [Decimal("0.1") * 2, Decimal("0.9") * 4] == [Decimal("0.2"), Decimal("3.6")]
assert Decimal("0.9") * 1 == Decimal("0.9")
assert Decimal("0.1") * 100 == Decimal("10")

next_code = code_fence(section(raw, "3.2"))
old_verification = json.loads((ARTIFACTS / "verification.json").read_bytes())
assert next_code == old_verification["next_section"]["code"]
next_namespace, next_stdout = execute(next_code, "content-dependent-weights")
assert next_namespace["weights"].tolist() == old_verification["next_section"]["weights"]
changed_q = torch.tensor([0.0, 1.0])
changed_q_weights = (changed_q @ next_namespace["keys"].T).softmax(dim=0)
assert changed_q_weights.argmax().item() == 1

layer = torch.nn.Linear(2, 1, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[-2.0, 4.0]]))
assert layer(torch.tensor([1.0, 1.0])).item() == 2.0
assert layer.weight.sum().item() == 2.0

scale_namespace, scale_stdout = execute(code_fence(section(raw, "3.5")), "sampling-scale")
assert scale_namespace["q"].shape == (1000, 64)
scale_examples = {
    "population_std_for_widths": {str(d): math.sqrt(d) for d in [4, 64]},
    "constant_zero_sample_std": torch.zeros(1000).std().item(),
    "finite_three_sample_std": torch.tensor([-2.0, 0.0, 2.0]).std().item(),
    "finite_three_sample_scaled_std": torch.tensor([-1.0, 0.0, 1.0]).std().item(),
    "seeded_original_stdout": scale_stdout,
    "derivation": "每格乘積獨立且等機率為 ±1，故均值 0、變異數 1；D 格相加變異數 D，理論標準差 sqrt(D)。樣本標準差由實際抽出的分數計算，不是直接填入母體標準差。對偶數 D，分數 0 有正機率，連續抽到全零仍有正機率，此時樣本標準差為 0；有限樣本因此不必等於母體理論值。另以 D=4 的可行取值 [-2,0,2] 示範有限樣本也可能恰好得到標準差 2；它不是固定一千組配置的重現。",
}
assert scale_examples["constant_zero_sample_std"] == 0.0
assert scale_examples["finite_three_sample_std"] == 2.0
assert scale_examples["finite_three_sample_scaled_std"] == 1.0

copied = {}
for name in ["stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "bootstrap.py"]:
    content = (RUN / name).read_bytes()
    copied[name] = {"sha256": sha(content), "content_utf8": content.decode()}
metadata = json.loads(copied["extraction.json"]["content_utf8"])
execution = json.loads(copied["execution.json"]["content_utf8"])
environment = json.loads(copied["environment.json"]["content_utf8"])
assert execution["exit_code"] == 0
assert metadata["source_sha256"] == sha(frozen)
assert metadata["source_file_sha256"] == sha(raw)
assert copied["stdout.txt"]["content_utf8"] == original_stdout
assert copied["stderr.txt"]["content_utf8"] == ""
assert environment["guard_events"] == []
assert sha((ROOT / SOURCE).read_bytes()) == sha(raw), "核對期間來源內容必須維持一致"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-1/verify-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version),
                    "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": str(ORIGINAL_REPORT.relative_to(ROOT)), "sha256": sha(original_raw), "issue_count": len(original["issues"])},
    "retained_artifacts": old_artifacts,
    "chapter": {"path": SOURCE, "previous_sha256": sha(old_raw), "current_sha256": sha(raw),
                "current_equals_specified_commit": raw == new_commit_raw},
    "diff": command(["git", "diff", BEFORE, AFTER, "--", SOURCE]),
    "section_comparison": section_checks,
    "source_freeze": {"path": str(freeze_path.relative_to(ROOT)), "sha256": sha(freeze_path.read_bytes()), "page": page},
    "section_execution": {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/03.md#3.1" --execute',
        "launcher_stdout": json.dumps({"artifact_directory": str(RUN), "source_sha256": sha(frozen), "python_fences": 1, "svg_references": 0, "execution_exit_code": 0}, ensure_ascii=False) + "\n",
        "launcher_exit_code": 0, "copied_files": copied},
    "original_example": {"stdout": original_stdout, "dtype": str(values.dtype), "output": output.tolist(),
                         "output_shape": list(output.shape), "reverse_output": reverse.tolist(), "values_unchanged": True,
                         "requires_grad": {"values": values.requires_grad, "weights": weights.requires_grad, "output": output.requires_grad}},
    "next_example": {"code_unchanged": True, "stdout": next_stdout, "scores": next_namespace["scores"].tolist(),
                     "weights": next_namespace["weights"].tolist(), "changed_query": changed_q.tolist(),
                     "changed_query_weights": changed_q_weights.tolist(), "current_chapter_sha256": sha(raw)},
    "linear_example": {"weight": layer.weight.tolist(), "weight_sum": layer.weight.sum().item(), "output": [2.0]},
    "changed_sentence_check": {"scope": "僅核對指定章檔差異；不擴大成本頁以外的完整審閱", "status": "verified", "sampling": scale_examples},
    "result": "所有斷言成立；本頁與原先引用的後續範例未變，前份原始證據未變。",
}
with (ARTIFACTS / "recheck-current-01.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
