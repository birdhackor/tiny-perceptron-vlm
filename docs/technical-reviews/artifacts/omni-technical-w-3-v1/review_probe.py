"""W.3 的有界 CPU 核對；直接執行凍結教材內嵌的唯一 Python 區塊。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3/sections/W.3.md"
EXPECTED_SOURCE_SHA256 = "18dbcb83b81379704f3092a93eb1def8e9325802aa37d13068c7a9db488851e3"
FIGURE = ROOT / "course/figures/rewrite-W-3-axes.svg"
EXPECTED_FIGURE_SHA256 = "74c93bfd89b0b164573f8dea1506179982c79250b77bfc862102895b02387138"
COMMAND = (
    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra cpu python "
    "docs/technical-reviews/artifacts/omni-technical-w-3-v1/review_probe.py "
    "> docs/technical-reviews/artifacts/omni-technical-w-3-v1/execution.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(block):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(block, "W.3-frozen-inline.py", "exec"), namespace)
    return namespace["scores"], output.getvalue()


def expected_error(operation):
    try:
        operation()
    except RuntimeError as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("預期 RuntimeError，但呼叫成功")


torch.set_num_threads(1)
assert sha256(SNAPSHOT) == EXPECTED_SOURCE_SHA256
assert sha256(FIGURE) == EXPECTED_FIGURE_SHA256
raw = SNAPSHOT.read_text(encoding="utf-8")
live = (ROOT / "course/first-steps.md").read_text(encoding="utf-8")
live_section = re.search(r"(?ms)^## W\.3 .*?(?=^## |\Z)", live).group()
assert live_section == raw
blocks = re.findall(r"```python\n(.*?)```", raw, re.S)
assert len(blocks) == 1
block = blocks[0]
scores, stdout = execute(block)
assert scores.device.type == "cpu"
assert scores.shape == (2, 3)
assert scores[0].tolist() == [60.0, 70.0, 80.0]
assert scores[1, 2].item() == 100.0
assert scores.mean(dim=1).tolist() == [70.0, 90.0]
assert scores.mean(dim=0).tolist() == [70.0, 80.0, 90.0]
assert torch.equal(scores.mean(dim=-1), scores.mean(dim=1))

before = scores.clone()
shape_tuple = tuple(scores.shape)
as_list = scores.tolist()
row_as_list = scores[0].tolist()
assert shape_tuple == (2, 3)
assert as_list == [[60.0, 70.0, 80.0], [80.0, 90.0, 100.0]]
assert row_as_list == [60.0, 70.0, 80.0]
assert torch.equal(before, scores)

flat = scores.reshape(6)
assert flat.tolist() == [60.0, 70.0, 80.0, 80.0, 90.0, 100.0]
assert torch.equal(flat.reshape(2, 3), scores)
reshape_error = expected_error(lambda: scores.reshape(8))
item_error = expected_error(scores.item)
three_axes = torch.arange(24).reshape(2, 4, 3)
assert tuple(three_axes[0].shape) == (4, 3)
scalar_vector_matrix_ndim = [
    torch.tensor(5.0).ndim,
    torch.tensor([5.0]).ndim,
    scores.ndim,
]
assert scalar_vector_matrix_ndim == [0, 1, 2]

# 練習直接替換教材輸入中的第一列最後一格，重新執行同一內嵌程式。
practice_block = block.replace("[[60.0, 70.0, 80.0]", "[[60.0, 70.0, 110.0]", 1)
assert practice_block != block
practice, practice_stdout = execute(practice_block)
assert practice.mean(dim=1).tolist() == [80.0, 90.0]
assert practice.mean(dim=0).tolist() == [70.0, 80.0, 105.0]

record = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": COMMAND,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "torch_num_threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
    },
    "source_snapshot": str(SNAPSHOT.relative_to(ROOT)),
    "source_sha256": sha256(SNAPSHOT),
    "live_section_matches_snapshot": True,
    "figure_sha256": sha256(FIGURE),
    "probe_sha256": sha256(Path(__file__)),
    "inline_block_sha256": hashlib.sha256(block.encode()).hexdigest(),
    "inline_block": block,
    "inline_stdout": stdout,
    "input": scores.tolist(),
    "dtype": str(scores.dtype),
    "shape": list(scores.shape),
    "first_row": scores[0].tolist(),
    "second_student_third_score": scores[1, 2].item(),
    "mean_dim_1": scores.mean(dim=1).tolist(),
    "mean_dim_0": scores.mean(dim=0).tolist(),
    "mean_dim_minus_1": scores.mean(dim=-1).tolist(),
    "mean_shapes": {"dim_1": list(scores.mean(dim=1).shape), "dim_0": list(scores.mean(dim=0).shape)},
    "shape_tuple_repr": repr(shape_tuple),
    "tolist": as_list,
    "row_tolist": row_as_list,
    "conversions_preserve_original": torch.equal(before, scores),
    "scalar_vector_matrix_ndim": scalar_vector_matrix_ndim,
    "three_axis_input_shape": list(three_axes.shape),
    "three_axis_first_index_shape": list(three_axes[0].shape),
    "reshape_6": flat.tolist(),
    "reshape_roundtrip_equal": torch.equal(flat.reshape(2, 3), scores),
    "reshape_8_error": reshape_error,
    "item_multiple_elements_error": item_error,
    "practice_input": practice.tolist(),
    "practice_stdout": practice_stdout,
    "practice_mean_dim_1": practice.mean(dim=1).tolist(),
    "practice_mean_dim_0": practice.mean(dim=0).tolist(),
    "assertions": "all passed",
    "scope": "一次離線 CPU、非空 float32 成績示例；無訓練、GPU、模型能力或跨平台實測。",
}
json.dump(record, sys.stdout, ensure_ascii=False, indent=2)
print()
