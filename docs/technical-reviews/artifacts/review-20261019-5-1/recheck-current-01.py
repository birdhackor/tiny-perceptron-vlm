"""複查未變的頁面、相依文件差異、裝置選擇與比較結論的範圍。"""

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
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch

from tiny_perceptron.quantization import QuantizedLinear
from tiny_perceptron.training import choose_device


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(args):
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, env=os.environ.copy())
    return {"argv": args, "exit_code": completed.returncode,
            "stdout": completed.stdout.decode(), "stderr": completed.stderr.decode()}


def section(raw, identifier):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, match in enumerate(headers) if match[0].startswith(b"## " + identifier.encode() + b" "))
    return raw[headers[index].start():headers[index + 1].start() if index + 1 < len(headers) else len(raw)]


original_path = "docs/course-review-20261019/reviews/rv1019_technical_5-1/technical-5-1-original-01.json"
original_raw = (ROOT / original_path).read_bytes()
assert sha(original_raw) == "c896a85b234151f9a0b747dc9cb2026d24ee8b6cc539c50f4048dfb555904342"
original = json.loads(original_raw)
assert original["issues"] == []
artifact_checks = []
for item in original["artifacts"]:
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"]
    artifact_checks.append({"id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": actual})
source_checks = []
for item in original["sources"]:
    if "path" not in item:
        continue
    actual = sha((ROOT / item["path"]).read_bytes())
    source_checks.append({"id": item["id"], "path": item["path"], "recorded_sha256": item["sha256"],
                          "current_sha256": actual, "unchanged": actual == item["sha256"]})
assert [item["path"] for item in source_checks if not item["unchanged"]] == ["course/training.md"]

baseline = run(["git", "show", "5ef56bc756f3:course/training.md"])
assert baseline["exit_code"] == 0
old_raw = baseline["stdout"].encode()
current_raw = (ROOT / "course/training.md").read_bytes()
assert sha(old_raw) == "46a4758ac655ac28f82312cec0d2cb23f419e6f2bfaede10c03a38da20613c21"
assert sha(current_raw) == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
change = run(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", "course/training.md"])
assert change["exit_code"] == 0
old_t4, current_t4 = section(old_raw, "T.4"), section(current_raw, "T.4")
assert old_t4 == current_t4
assert b"--experiment text_foundation --device cuda" in current_t4
assert b"--steps 200" in current_t4 and b"layers=1" in current_t4

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/5.1.md").read_bytes()
assert frozen == section((ROOT / "course/chapters/05.md").read_bytes(), "5.1")
assert sha(frozen) == original["source_sha256"]
run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T102329830800Z"
section_files = {}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_dir / name).read_bytes()
    section_files[name] = {"sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
assert section_files["execution.json"]["content"]["exit_code"] == 0
assert section_files["extraction.json"]["content"]["source_sha256"] == sha(frozen)
assert section_files["extraction.json"]["content"]["figure_sha256"] == {}
stdout = section_files["stdout.txt"]["content"]
assert stdout.count("第一步embedding梯度大小") == 1
numbers = re.search(r"固定例子代價 ([0-9.eE+-]+) → ([0-9.eE+-]+)", stdout)
before, after = map(float, numbers.groups())
assert after < before and abs(before - 2.27) < 0.005 and abs(after - 0.0004) < 0.00005

fence = section_files["fence-1.py"]["content"]
modified = fence.replace("y = torch.tensor([[2, 3, 4]])", "y = torch.tensor([[-100, -100, -100]])")
assert modified != fence
captured = io.StringIO()
torch.set_num_threads(1)
torch.set_default_device("cpu")
try:
    with contextlib.redirect_stdout(captured):
        exec(compile(modified, "all-ignored-labels", "exec"), {"__name__": "__main__"})
except ValueError as error:
    empty_labels = {"error_type": type(error).__name__, "message": str(error), "stdout": captured.getvalue()}
else:
    raise AssertionError("全忽略目標應拒絕計算")
assert "所有 labels 都被忽略" in empty_labels["message"] and empty_labels["stdout"] == ""

cli_files = ["scripts/train.py", "scripts/evaluate.py", "scripts/infer.py", "scripts/course_experiments/run.py",
             "scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"]
cli_checks = []
for filename in cli_files:
    text = (ROOT / filename).read_text()
    flags = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if any(isinstance(arg, ast.Constant) and arg.value == "--device" for arg in node.args):
                flags.append(ast.unparse(node))
    help_result = run([str(ROOT / ".venv/bin/python"), filename, "--help"])
    assert help_result["exit_code"] == 0
    if filename in ("scripts/quantize.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py"):
        assert flags == [] and "--device" not in help_result["stdout"]
    elif filename == "scripts/course_experiments/run.py":
        assert len(flags) == 1 and 'default=\'cpu\'' in flags[0]
    else:
        assert len(flags) == 1 and 'default=\'auto\'' in flags[0]
    cli_checks.append({"path": filename, "sha256": sha(text.encode()), "device_arguments": flags, "help_execution": help_result})
branches = []
for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        observed = str(choose_device("auto"))
        assert observed == expected
        branches.append({"mock_cuda_available": cuda, "mock_mps_available": mps, "observed_device": observed})

layer = torch.nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1.0, 0.1], [0.2, 0.04]]))
packed = QuantizedLinear(layer, bits=4)
x = torch.tensor([[1.0, 0.0]])
float_scores, quantized_scores = layer(x).detach(), packed(x).detach()
assert int(float_scores.argmax(-1).item()) == int(quantized_scores.argmax(-1).item()) == 0
quantization_example = {
    "input": x.tolist(), "float_scores": float_scores.tolist(), "quantized_scores": quantized_scores.tolist(),
    "float_prediction": 0, "quantized_prediction": 0, "expected_target": 1,
    "meaning": "兩個答案一致但都不等於期望目標，說明一致性與答對是兩個問題；這不是課文檢查點的實測。",
}

url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
source_command = ["curl", "-fsSL", "--max-time", "30", url]
source_run = subprocess.run(source_command, capture_output=True, check=True)
class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
parser = PlainText()
parser.feed(source_run.stdout.decode())
text = " ".join(" ".join(parser.parts).split())
excerpt = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
assert excerpt in text
reproducibility = {
    "url": url, "version": "PyTorch 2.14 documentation", "accessed_on": datetime.now(UTC).date().isoformat(),
    "checked_original": True, "command": source_command, "exit_code": source_run.returncode,
    "stderr": source_run.stderr.decode(), "full_response_sha256": sha(source_run.stdout),
    "full_response_bytes": len(source_run.stdout), "excerpt": excerpt,
    "locator": "Reproducibility 開頭段落",
    "inspection_note": "官方未保證跨版本或平台完全重現；未因此宣稱每次跨機都不同，也未量測此固定配方的差異幅度。",
}
head = run(["git", "rev-parse", "HEAD"])
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-1/recheck-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "head": head, "original_report": {"path": original_path, "sha256": sha(original_raw)},
    "previous_issues": original["issues"], "original_artifact_integrity": artifact_checks,
    "repository_source_integrity": source_checks,
    "training_change": {"diff_execution": change, "baseline_command": baseline["argv"], "baseline_exit_code": baseline["exit_code"],
                        "old_sha256": sha(old_raw), "current_sha256": sha(current_raw), "t4_unchanged": True,
                        "t4_sha256": sha(current_t4), "t4_text": current_t4.decode()},
    "page_source_sha256": sha(frozen),
    "section_execution": {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.1" --execute',
        "copied_from": str(run_dir.relative_to(ROOT)), "files": section_files,
    },
    "fresh_fixed_example": {"before": before, "after": after}, "all_ignored_labels": empty_labels,
    "cli_checks": cli_checks, "mocked_auto_device_branches": branches,
    "quantization_comparison_example": quantization_example, "reproducibility_source": reproducibility,
    "result": "所有核對通過；原頁、舊證據與所引用的文字配方未變，三處相依文件修訂未推翻原判斷。",
}
output = Path(__file__).with_name("recheck-evidence-current-01.json")
with output.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"path": str(output.relative_to(ROOT)), "sha256": sha(output.read_bytes()), "result": result["result"],
                  "fixed_example": result["fresh_fixed_example"]}, ensure_ascii=False))
