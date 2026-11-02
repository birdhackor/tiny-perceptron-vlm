"""Execute the exact frozen 14.19 block and its stated exercises on CPU."""

from contextlib import redirect_stdout
from datetime import UTC, datetime
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import platform
import re

import torch


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix10.json"
COMMAND = "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-14-19-v1/verify_lesson.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(code, filename):
    namespace = {"__name__": "__frozen_lesson_review__"}
    stdout = StringIO()
    with redirect_stdout(stdout):
        exec(compile(code, filename, "exec"), namespace)
    return namespace, stdout.getvalue()


def close(actual, expected, tolerance=1e-12):
    assert abs(actual - expected) <= tolerance, (actual, expected)


torch.set_num_threads(1)
freeze = json.loads(FREEZE.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "14.19")
snapshot = ROOT / page["snapshot"]
assert sha(snapshot) == page["source_sha256"]
for path, digest in page["figures_sha256"].items():
    assert sha(ROOT / path) == digest
blocks = re.findall(r"```python\n(.*?)```", snapshot.read_text(), re.S)
assert len(blocks) == 1
code = blocks[0]
ns, stdout = run(code, page["snapshot"] + ":14-60")
expected_stdout = (
    "F5／A1 原時／A1 移時的 t [5.0, 5.2, 9.2]\n"
    "原時／移時對 F5 的點積 [2.9877, 1.0123]\n"
    "移時只改時間對 True\n"
    "序號版兩個分數相同 True\n"
    "A1 可讀 F5／F15 True True\n"
    "A1 可讀下一個聲音塊 False\n"
    "只改高度後三對 [[-0.5878, -0.809], [0.7071, 0.7071], [1.0, 0.0]]\n"
)
assert stdout == expected_stdout, stdout

actual_scores = [float(ns[name] @ ns["key"]) for name in ["q_a", "q_b"]]
scalar_scores = [2 + math.cos(delta * math.pi / 4) for delta in [0.2, 4.2]]
for actual, expected in zip(actual_scores, scalar_scores, strict=True):
    close(actual, expected)
assert len(ns["tags"]) == 24
assert ns["tags"][6] == ns["tags"][16] == "frame"
assert ns["tags"][20] == ns["tags"][21] == "audio"
mask = ns["causal_mask"]
assert mask.shape == (24, 24) and mask.dtype == torch.bool
assert torch.equal(torch.nonzero(mask[20]).flatten(), torch.arange(21))
assert ns["index_scores"][0] == ns["index_scores"][1]
assert torch.equal(ns["q_a"][2:], ns["q_b"][2:])
assert torch.equal(ns["moved_h"][:2], ns["q_a"][:2])
assert torch.equal(ns["moved_h"][4:], ns["q_a"][4:])

# Execute exactly the exercise's replacement expression in the lesson namespace.
with redirect_stdout(StringIO()):
    exec("moved_w = axis_rope(content, coordinates(1300.0, w=1.0))", ns)
moved_w = ns["moved_w"]
assert torch.equal(moved_w[:4], ns["q_a"][:4])
expected_w = [math.sqrt(0.5), math.sqrt(0.5)]
for actual, expected in zip(moved_w[4:].tolist(), expected_w, strict=True):
    close(actual, expected)

# Re-execute the full embedded block after its one stated unit change.
assert code.count("unit_ms = 250.0") == 1
half_ns, half_stdout = run(code.replace("unit_ms = 250.0", "unit_ms = 500.0"),
                           page["snapshot"] + ":unit_ms=500 exercise")
half_scores = [float(half_ns[name] @ half_ns["key"]) for name in ["q_a", "q_b"]]
for actual, delta in zip(half_scores, [0.1, 2.1], strict=True):
    close(actual, 2 + math.cos(delta * math.pi / 4))
assert [1250 / 500, 1300 / 500, 2300 / 500] == [2.5, 2.6, 4.6]
assert [round(s, 4) for s in half_scores] == [2.9969, 1.9215]
assert not torch.equal(half_ns["q_a"], ns["q_a"])

# Confirm the explicit six-feature guard on the supported even input shape.
try:
    ns["axis_rope"](torch.ones(4, dtype=torch.float64), ns["coordinates"](1300.0))
except ValueError as error:
    assert str(error) == "每一對都要有軸分配"
else:
    raise AssertionError("four features must fail the three-pair mapping")

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": COMMAND,
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "torch_git": torch.version.git_version, "device": "cpu", "dtype": "float64",
                    "threads": str(torch.get_num_threads()), "platform": platform.platform()},
    "input": {"freeze": str(FREEZE.relative_to(ROOT)), "freeze_sha256": sha(FREEZE),
              "snapshot": page["snapshot"], "source_sha256": sha(snapshot),
              "embedded_block_sha256": hashlib.sha256(code.encode()).hexdigest(),
              "embedded_block_lines": "14-60", "content": [1, 0, 1, 0, 1, 0],
              "time_ms": [1250, 1300, 2300], "unit_ms": 250.0,
              "pair_axis": [0, 1, 2], "radians_per_unit": math.pi / 4},
    "stdout": stdout,
    "observations": {"time_coordinates": [5.0, 5.2, 9.2], "raw_dot_products": actual_scores,
                     "independent_scalar_dot_products": scalar_scores,
                     "qa": ns["q_a"].tolist(), "qb": ns["q_b"].tolist(),
                     "sequence_length": len(ns["tags"]), "F5_index": 6, "F15_index": 16,
                     "A1_index": 20, "next_audio_index": 21,
                     "mask_shape": list(mask.shape), "A1_visible_index_interval": [0, 20],
                     "index_only_scores": ns["index_scores"],
                     "moved_h": ns["moved_h"].reshape(3, 2).tolist()},
    "exercise_w": {"expression": "coordinates(1300.0, w=1.0)",
                   "observed_pairs": moved_w.reshape(3, 2).round(decimals=4).tolist(),
                   "first_two_pairs_exactly_unchanged": True},
    "exercise_unit_500": {"time_coordinates": [2.5, 2.6, 4.6], "stdout": half_stdout,
                          "raw_dot_products": half_scores,
                          "rounded_dot_products": [round(s, 4) for s in half_scores]},
    "comparison": {"float_tolerance": "absolute 1e-12 against independent scalar cosine",
                   "printed_numbers": "exact stated four-decimal values; -0.809 equals -0.8090",
                   "unchanged_pairs_and_masks": "exact torch.equal / Python bool equality"},
    "result": "All exact frozen-block outputs, index/mask checks, scalar calculations and stated exercises agree.",
    "scope": "Deterministic CPU mechanism demonstration only; no optimizer, training, GPU or quality inference.",
}
with (HERE / "execution.json").open("x") as output:
    output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
