"""核對依賴教材差異、引用區段與共用表示的分數。"""

import ast
import hashlib
import io
import json
import platform
import re
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import torch


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(*args):
    result = subprocess.run(list(args), check=True, capture_output=True)
    return {"command_argv": list(args), "exit_code": result.returncode,
            "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, header in enumerate(headers):
        if header[0].startswith(("## " + identifier + " ").encode()):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(raw)
            return raw[header.start():end]
    raise ValueError("找不到指定小節")


def extract_class(raw, identifier):
    for code in re.findall(r"```python\n(.*?)```", raw.decode(), re.S):
        for node in ast.parse(code).body:
            if isinstance(node, ast.ClassDef) and node.name == identifier:
                return ast.get_source_segment(code, node).encode()
    raise ValueError("找不到指定類別")


torch.set_num_threads(1)
torch.set_default_device("cpu")
previous_path = Path("docs/course-review-20261019/reviews/rv1019_technical_4-5/technical-4-5-current-01.json")
assert digest(previous_path.read_bytes()) == "7d024e6ca4bc2dab1f19353ef73ad0343e1575e3870086c594f281cfdb9eed53"
previous = json.loads(previous_path.read_text())
assert previous["issues"] == []
source_checks = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(Path(source["path"]).read_bytes())
        source_checks.append({"source_id": source["id"], "path": source["path"],
                              "previous_sha256": source["sha256"], "current_sha256": actual,
                              "unchanged": source["sha256"] == actual})
assert [s["path"] for s in source_checks if not s["unchanged"]] == ["course/chapters/03.md"]
artifact_checks = []
for artifact in previous["artifacts"]:
    actual = digest(Path(artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_receipt = command("git", "show", "ecf6dec71280:course/chapters/03.md")
new_receipt = command("git", "show", "c6b9339d2595:course/chapters/03.md")
old = old_receipt["stdout"].encode()
new = new_receipt["stdout"].encode()
current = Path("course/chapters/03.md").read_bytes()
assert digest(old) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
assert digest(new) == digest(current) == "5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5"
diff = command("git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/chapters/03.md")
assert section(old, "3.7") == section(current, "3.7")
assert section(old, "3.5") == section(current, "3.5")
assert extract_class(old, "CausalAttention") == extract_class(current, "CausalAttention")
query_section = section(current, "3.3")
fences = re.findall(r"```python\n(.*?)```", query_section.decode(), re.S)
assert len(fences) == 2
namespace = {}
stdout = io.StringIO()
with redirect_stdout(stdout):
    for code in fences:
        exec(compile(code, "course/chapters/03.md", "exec"), namespace)
has = namespace["has"]
need = namespace["need"]
shared = has @ has.T
separate = need @ has.T
assert torch.equal(shared, torch.eye(2))
assert torch.equal(separate, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
assert torch.equal(shared.argmax(-1), torch.arange(2))
assert shared[0, 1].item() == shared[1, 0].item()
main = section(Path("course/chapters/04.md").read_bytes(), "4.5")
frozen = Path("docs/course-review-20261019/freeze-01/sources/4.5.md").read_bytes()
assert main == frozen
record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-5/verify_dependencies-current-02.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": "torch.float32"},
    "previous_report_sha256": digest(previous_path.read_bytes()), "previous_issues": previous["issues"],
    "source_checks": source_checks, "artifact_checks": artifact_checks, "diff_receipt": diff,
    "snapshot_receipts": [{"command_argv": old_receipt["command_argv"], "exit_code": 0, "sha256": digest(old)},
                          {"command_argv": new_receipt["command_argv"], "exit_code": 0, "sha256": digest(new)}],
    "unchanged_sections": [{"sha256": digest(section(current, identifier)), "identifier": identifier, "byte_equal": True} for identifier in ["3.5", "3.7"]],
    "unchanged_attention_class": {"sha256": digest(extract_class(current, "CausalAttention")), "byte_equal": True},
    "current_section_equals_freeze": True, "source_sha256": digest(frozen),
    "figure_sha256": digest(Path("course/figures/rewrite-04-block-flow.svg").read_bytes()),
    "png_sha256": digest(Path("outputs/course-review-20261019/figure-renders-04/510aa00b790a2902da7629092dcb4bd883d0d19f8102ec1518a76e9315f3b2bb.png").read_bytes()),
    "freeze_sha256": digest(Path("docs/course-review-20261019/freeze-07.json").read_bytes()),
    "changed_sentence": "一般來說，共用一份時 A 對 B 的分數一定等於 B 對 A 的分數；本例 A、B 跟自己都得 1、彼此得 0，所以最高分都落在自己。",
    "original_fence_stdout": stdout.getvalue(), "original_fence_sha256": [digest(code.encode()) for code in fences],
    "has": has.tolist(), "need": need.tolist(), "shared_scores": shared.tolist(), "separate_scores": separate.tolist(),
    "maximum_score_indices": shared.argmax(-1).tolist(),
    "derivation": "本例 A=(1,0)、B=(0,1)：A·A=1×1+0×0=1，B·B=0×0+1×1=1，A·B=1×0+0×1=0，B·A=0×1+1×0=0。共用表示的實數點積 A·B=Σ_i A_i B_i=Σ_i B_i A_i=B·A，故分數矩陣對稱；本例每列自己得 1、另一位置得 0，最高分嚴格落在自己。改寫後限定本例，不再以向量長度的語意推論任意輸入的自我分數。",
    "result": "所有斷言通過；引用的注意力類別與縮放示範未變，新句的分數與最高分位置正確。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
