"""補核 FFN 的逐位置計算、row_limit=20 與官方 MiB 格式化算式。"""

import ast
import contextlib
import datetime
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[4]
here = Path(__file__).resolve().parent
cache = Path("/workspace/work/reference/omni-technical-technical_16_1-v1")
snapshot = root / "docs/course-review-20261019/freeze-01/sources/16.1.md"
code = re.findall(r"```python\n(.*?)```", snapshot.read_text(), re.S)[0]
assert code.count("row_limit=10") == 1
namespace = {"__name__": "__inline_row20_check__"}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(code.replace("row_limit=10", "row_limit=20"), str(snapshot), "exec"), namespace)
assert len(re.findall(r"^\s+aten::", stream.getvalue(), re.M)) == 20
assert "aten::mm" in stream.getvalue() and "aten::addmm" in stream.getvalue()
ffn = namespace["model"].blocks[0].ffn
x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 24
with torch.no_grad():
    joint = ffn(x)
    separate = torch.cat([ffn(x[:, i : i + 1]) for i in range(3)], dim=1)
error = float((joint - separate).abs().max())
assert torch.allclose(joint, separate, rtol=1e-6, atol=1e-7)
memory_path = cache / "torch__cuda__memory.py"
memory_tree = ast.parse(memory_path.read_text())
summary = next(n for n in memory_tree.body if isinstance(n, ast.FunctionDef) and n.name == "memory_summary")
formatter = next(n for n in summary.body if isinstance(n, ast.FunctionDef) and n.name == "_format_size")
format_namespace = {}
exec(compile(ast.Module(body=[formatter], type_ignores=[]), str(memory_path), "exec"), format_namespace)
# 官方 formatter 依同欄最大值 pref_sz 選單位，這裡用 1 GiB 作為欄寬參考。
unit = format_namespace["_format_size"](2**20, 2**30).strip()
assert unit == "1 MiB"
out = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_16_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_16_1-v1/targeted_checks.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "CPU", "threads": str(torch.get_num_threads())},
    "row_limit_20_stdout": stream.getvalue(),
    "FFN_input_shape": list(x.shape),
    "FFN_joint_vs_individual_position_max_abs_error": error,
    "FFN_tolerance": "rtol=1e-6, atol=1e-7 (FP32 matrix kernels may accumulate in a different order)",
    "MiB_official_formatter_source": {"url": "https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/cuda/memory.py#L672", "cache": str(memory_path), "sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(), "observed": unit, "input_bytes": 2**20, "column_reference_bytes": 2**30},
    "result": "全部斷言成立；只做 CPU 前向與純格式化算式，未訓練、未執行 CUDA API。",
}
target = here / "targeted-checks.json"
with target.open("x") as f:
    f.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(target.relative_to(root)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "FFN_max_abs_error": error, "MiB": unit, "result": out["result"]}, ensure_ascii=False))
