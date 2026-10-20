"""核對來源差異、精簡注意力類別與有限樣本的標準差。"""

import ast
import hashlib
import io
import json
import math
import platform
import re
import subprocess
from contextlib import redirect_stdout
from pathlib import Path

import torch


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    command = ["git", *args]
    result = subprocess.run(command, check=True, capture_output=True)
    return {"command_argv": command, "exit_code": result.returncode,
            "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def section(raw, identifier):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, heading in enumerate(headings):
        if heading[0].startswith(("## " + identifier + " ").encode()):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
            return raw[heading.start():end]
    raise ValueError("找不到指定小節")


def class_text(raw, name):
    for code in re.findall(r"```python\n(.*?)```", raw.decode(), re.S):
        for node in ast.parse(code).body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return ast.get_source_segment(code, node).encode()
    raise ValueError("找不到指定類別")


torch.set_num_threads(1)
torch.set_default_device("cpu")
original_path = Path("docs/course-review-20261019/reviews/rv1019_technical_4-5/technical-4-5-original-01.json")
original = json.loads(original_path.read_text())
assert digest(original_path.read_bytes()) == "9396fc90b4b01bd6e7f66d188e9691a2c1739a6c73ba847f2b9ec40725d613c4"
source_checks = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = digest(Path(source["path"]).read_bytes())
        source_checks.append({"source_id": source["id"], "path": source["path"],
                              "previous_sha256": source["sha256"], "current_sha256": actual,
                              "unchanged": source["sha256"] == actual})
assert [s["path"] for s in source_checks if not s["unchanged"]] == ["course/chapters/03.md"]
artifact_checks = []
for artifact in original["artifacts"]:
    actual = digest(Path(artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    artifact_checks.append({"path": artifact["path"], "sha256": actual, "unchanged": True})
old_read = git("show", "5ef56bc756f3:course/chapters/03.md")
new_read = git("show", "3af7f13ef959:course/chapters/03.md")
old = old_read["stdout"].encode()
new = new_read["stdout"].encode()
current = Path("course/chapters/03.md").read_bytes()
assert digest(old) == "df0709c102dba36fa89b3c88904f55172339a61c15c4a0583484b9b919bd2ee7"
assert digest(current) == digest(new) == "61ae7986d5c2786178e680ec2e6299ba4d162b1feb6f9b5acbb94c175351cd09"
diff = git("diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/03.md")
prior_attention = section(old, "3.7")
current_attention = section(current, "3.7")
assert prior_attention == current_attention
prior_class = class_text(old, "CausalAttention")
current_class = class_text(current, "CausalAttention")
assert prior_class == current_class
scaled_section = section(current, "3.5")
code, = re.findall(r"```python\n(.*?)```", scaled_section.decode(), re.S)
stdout = io.StringIO()
with redirect_stdout(stdout):
    exec(compile(code, "course/chapters/03.md", "exec"), {})
rows = []
for line in stdout.getvalue().splitlines():
    width, unscaled, scaled = line.split()
    rows.append({"width": int(width), "sample_count": 1000,
                 "observed_std": float(unscaled), "observed_scaled_std": float(scaled),
                 "population_std": math.sqrt(int(width)), "population_scaled_std": 1.0})
positive = torch.tensor([12.0] * 2 + [4.0] * 4 + [8.0] * 494, dtype=torch.float64)
finite_sample = torch.cat((positive, -positive))
assert finite_sample.numel() == 1000
assert finite_sample.sum().item() == 0.0
assert finite_sample.square().sum().item() == 63936.0
assert finite_sample.std(correction=1).item() == 8.0
assert (finite_sample / 8.0).std(correction=1).item() == 1.0
chapter = Path("course/chapters/04.md").read_bytes()
frozen = Path("docs/course-review-20261019/freeze-01/sources/4.5.md").read_bytes()
assert section(chapter, "4.5") == frozen
record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-5/verify_dependencies-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "sample_dtype": "torch.float32", "constructed_sample_dtype": "torch.float64"},
    "original_report_sha256": digest(original_path.read_bytes()),
    "original_issues": original["issues"], "source_checks": source_checks, "artifact_checks": artifact_checks,
    "diff_receipt": diff,
    "git_snapshot_receipts": [{"command_argv": old_read["command_argv"], "exit_code": old_read["exit_code"], "sha256": digest(old)},
                              {"command_argv": new_read["command_argv"], "exit_code": new_read["exit_code"], "sha256": digest(new)}],
    "unchanged_attention_section": {"sha256": digest(current_attention), "byte_equal": True},
    "unchanged_attention_class": {"sha256": digest(current_class), "byte_equal": True},
    "current_section_equals_freeze": True, "source_sha256": digest(frozen),
    "figure_sha256": digest(Path("course/figures/rewrite-04-block-flow.svg").read_bytes()),
    "png_sha256": digest(Path("outputs/course-review-20261019/figure-renders-04/510aa00b790a2902da7629092dcb4bd883d0d19f8102ec1518a76e9315f3b2bb.png").read_bytes()),
    "source_freeze_sha256": digest(Path("docs/course-review-20261019/freeze-05.json").read_bytes()),
    "changed_sentence": "縮放後兩者都靠近 1；有限樣本通常有抽樣誤差，所以不必恰好等於理論值。",
    "statistical_derivation": "每一獨立乘積 Z=QK 等機率為 ±1，E[Z]=0、Var(Z)=1；S 為 d 個 Z 相加，Var(S)=d、SD(S)=sqrt(d)。S/sqrt(d) 的母體標準差為 1。有限樣本的標準差是由隨機樣本算出的統計量，並非被公式強制等於母體值；也不能由樣本有限推出必然不等。",
    "sampling_code_sha256": digest(code.encode()), "sampling_stdout": stdout.getvalue(), "sampling_rows": rows,
    "exact_finite_sample": {"sample_count": 1000, "width": 64, "positive_value_counts": {"12": 2, "4": 4, "8": 494},
                            "negative_value_counts": {"-12": 2, "-4": 4, "-8": 494},
                            "sum": 0, "sum_of_squares": 63936, "correction": 1,
                            "variance_calculation": "63936/(1000-1)=64", "std": 8.0, "scaled_std": 1.0,
                            "scope": "每個取值都可由 64 個 ±1 之和產生，故這種有限樣本具有非零機率；只用來排除必然不等，不估計這種事件頻率。"},
    "result": "所有斷言通過；前置注意力小節與類別逐位元不變，修訂後抽樣誤差敘述成立。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
