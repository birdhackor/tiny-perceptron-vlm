"""17.4 獨立技術核對；直接執行 freeze 中的教材程式及所指定練習。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from tiny_perceptron.quantization import pack_int4, quantize_symmetric, unpack_int4


ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "17.4")
snapshot = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(snapshot).hexdigest() == page["source_sha256"]
text = snapshot.decode("utf-8")
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
assert len(blocks) == 1
code = blocks[0]
assignment = "x = torch.tensor([-1.0, 0.0, 1.0, 2.0])"
assert code.count(assignment) == 1


def run_block(replacement=None):
    actual = code if replacement is None else code.replace(assignment, replacement)
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(actual, page["snapshot"] + ":python-block", "exec"), namespace)
    result = {
        "input": namespace["x"].tolist(),
        "input_dtype": str(namespace["x"].dtype),
        "low": namespace["low"],
        "high": namespace["high"],
        "scale": namespace["scale"],
        "zero": namespace["zero"],
        "q": namespace["q"].tolist(),
        "q_dtype": str(namespace["q"].dtype),
        "q_element_bytes": namespace["q"].element_size(),
        "q_total_bytes": namespace["q"].numel() * namespace["q"].element_size(),
        "restored": namespace["restored"].tolist(),
        "restored_equals_input": torch.equal(namespace["restored"], namespace["x"]),
        "maximum_absolute_error": (namespace["x"] - namespace["restored"]).abs().max().item(),
        "stdout": output.getvalue(),
    }
    return result


baseline = run_block()
assert baseline["scale"] == 0.2 and baseline["zero"] == 5
assert baseline["q"] == [0, 5, 10, 15]
assert baseline["restored"] == [-1.0, 0.0, 1.0, 2.0]
assert baseline["restored_equals_input"] and baseline["q_total_bytes"] == 4
exercise = run_block("x = torch.tensor([-2.0, 0.0, 1.0])")
assert exercise["scale"] == 0.2 and exercise["zero"] == 10
assert exercise["q"] == [0, 10, 15]
assert exercise["restored_equals_input"]
off_grid = run_block("x = torch.tensor([-1.0, 0.0, 0.13, 2.0])")
assert off_grid["q"] == [0, 5, 6, 15]
assert abs(off_grid["maximum_absolute_error"] - 0.07) < 1e-6
positive = run_block("x = torch.tensor([1.0, 2.0])")
negative = run_block("x = torch.tensor([-2.0, -1.0])")
assert positive["low"] == 0 and positive["zero"] == 0
assert negative["high"] == 0 and negative["zero"] == 15
try:
    run_block("x = torch.zeros(4)")
except ZeroDivisionError as error:
    all_zero = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("教材明示的零寬度問題未出現")

qrange = {}
for bits, maximum in [(4, 7), (8, 127)]:
    integers, scale = quantize_symmetric(torch.tensor([-1.0, 0.0, 1.0]), bits=bits)
    assert integers.tolist() == [-maximum, 0, maximum]
    qrange[str(bits)] = {"codes": integers.tolist(), "dtype": str(integers.dtype), "scale": scale.item()}
signed_codes = torch.arange(-8, 8, dtype=torch.int8)
packed = pack_int4(signed_codes)
assert torch.equal(unpack_int4(packed, signed_codes.shape), signed_codes)
rejected = {}
for value in [-9, 8]:
    try:
        pack_int4(torch.tensor([value], dtype=torch.int8))
    except ValueError as error:
        rejected[str(value)] = str(error)
    else:
        raise AssertionError("pack_int4 接受範圍外值")
uint8 = torch.iinfo(torch.uint8)
assert (uint8.min, uint8.max, uint8.bits) == (0, 255, 8)
assert len(range(16)) == 2**4 == 16
assert (10 - 5) * 0.2 == 1

current = (ROOT / "course/chapters/17.md").read_text()
context = {}
for lesson in ["17.2", "17.3", "17.4", "17.5", "17.8"]:
    entry = next(p for p in freeze["pages"] if p["page_id"] == lesson)
    section = re.search(r"^## " + re.escape(lesson) + r" .*?(?=^## |\Z)", current, re.M | re.S).group()
    raw = section.encode("utf-8")
    assert raw == (ROOT / entry["snapshot"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["source_sha256"]
    context[lesson] = {"path": entry["snapshot"], "sha256": entry["source_sha256"], "current_equal": True}

result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_17_4-v1/verify.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "torch_git_version": torch.version.git_version, "device": "cpu", "platform": platform.platform()},
    "source_freeze": {"path": str(FREEZE.relative_to(ROOT)), "sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    "source": context,
    "python_block_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "baseline": baseline,
    "exercise": exercise,
    "off_grid": off_grid,
    "include_zero_examples": {"positive": positive, "negative": negative},
    "all_zero": all_zero,
    "symmetric_ranges": qrange,
    "pack_int4": {"input": signed_codes.tolist(), "unsigned_nibbles": (signed_codes.to(torch.int16) + 8).tolist(), "packed": packed.tolist(), "round_trip_equal": True, "rejected": rejected},
    "uint8": {"min": uint8.min, "max": uint8.max, "bits": uint8.bits},
    "result": "All assertions passed. These are deterministic CPU mechanism checks; no training or quality/speed claim is tested.",
}
target = Path(__file__).with_name("verification.json")
with target.open("x", encoding="utf-8") as output:
    json.dump(result, output, ensure_ascii=False, indent=2)
    output.write("\n")
print(json.dumps({"result": result["result"], "output": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, ensure_ascii=False))
