"""核對凍結程式、透明算例及本專案量化碼與打包範圍。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.quantization import pack_int4, quantize_symmetric, unpack_int4

torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")
snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/17.4.md"
raw = snapshot.read_bytes()
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
initial = "x = torch.tensor([-1.0, 0.0, 1.0, 2.0])"
assert code.count(initial) == 1


def evaluate(values):
    namespace = {}
    output = io.StringIO()
    replaced = code.replace(initial, f"x = torch.tensor({values!r})")
    with contextlib.redirect_stdout(output):
        exec(compile(replaced, str(snapshot), "exec"), namespace)
    x, q, restored = (namespace[key] for key in ("x", "q", "restored"))
    return {
        "input": values,
        "low": namespace["low"],
        "high": namespace["high"],
        "scale": namespace["scale"],
        "zero": namespace["zero"],
        "codes": q.tolist(),
        "restored": restored.tolist(),
        "difference": (x - restored).tolist(),
        "equal": torch.equal(x, restored),
        "dtype": str(q.dtype),
        "element_bytes": q.element_size(),
        "storage_bytes": q.numel() * q.element_size(),
        "stdout": output.getvalue(),
    }


original = evaluate([-1.0, 0.0, 1.0, 2.0])
exercise = evaluate([-2.0, 0.0, 1.0])
off_grid = evaluate([-1.0, 0.0, 1.0, 2.0, 0.7])
positive = evaluate([1.0, 2.0])
negative = evaluate([-2.0, -1.0])
assert original["codes"] == [0, 5, 10, 15]
assert original["equal"] and original["scale"] == 0.2 and original["zero"] == 5
assert original["element_bytes"] == 1 and original["storage_bytes"] == 4
assert exercise["codes"] == [0, 10, 15]
assert exercise["equal"] and exercise["scale"] == 0.2 and exercise["zero"] == 10
assert not off_grid["equal"]
assert positive["low"] == 0.0 and positive["zero"] == 0
assert negative["high"] == 0.0 and negative["zero"] == 15
try:
    evaluate([0.0, 0.0])
except ZeroDivisionError as error:
    zero_case = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("全零輸入沒有出現預期的除零錯誤")

symmetric = []
for bits in (4, 8):
    q, scale = quantize_symmetric(torch.tensor([-1.0, 0.0, 1.0]), bits=bits)
    maximum = 2 ** (bits - 1) - 1
    assert q.tolist() == [-maximum, 0, maximum]
    symmetric.append({"bits": bits, "maximum": maximum, "codes": q.tolist(), "scale": scale.item()})

all_codes = torch.arange(-8, 8, dtype=torch.int8)
packed = pack_int4(all_codes)
unpacked = unpack_int4(packed, all_codes.shape)
assert torch.equal(unpacked, all_codes)
assert packed.numel() * packed.element_size() == 8
assert (packed & 15).tolist() == list(range(0, 16, 2))
assert (packed >> 4).tolist() == list(range(1, 16, 2))
rejected = []
for value in (-9, 8):
    try:
        pack_int4(torch.tensor([value], dtype=torch.int8))
    except ValueError as error:
        rejected.append({"value": value, "exception": type(error).__name__, "message": str(error)})
assert len(rejected) == 2

step = Fraction(3, 15)
grid = [(i - 5) * step for i in range(16)]
exercise_grid = [(i - 10) * step for i in range(16)]
assert step == Fraction(1, 5)
assert grid[0] == -1 and grid[1] == Fraction(-4, 5) and grid[5] == 0 and grid[10] == 1 and grid[15] == 2
assert exercise_grid[0] == -2 and exercise_grid[10] == 0 and exercise_grid[15] == 1

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-4/verify_claims.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "default_dtype": str(torch.get_default_dtype())},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "repository_code_sha256": hashlib.sha256((ROOT / "tiny_perceptron/quantization.py").read_bytes()).hexdigest(),
    "original": original,
    "exercise": exercise,
    "off_grid": off_grid,
    "positive": positive,
    "negative": negative,
    "all_zero": zero_case,
    "symmetric": symmetric,
    "packing": {"input": all_codes.tolist(), "offset_codes": (all_codes.to(torch.int16) + 8).tolist(), "packed": packed.tolist(), "unpacked": unpacked.tolist(), "bytes": packed.numel() * packed.element_size(), "rejected": rejected},
    "uint8": {"min": torch.iinfo(torch.uint8).min, "max": torch.iinfo(torch.uint8).max, "element_bytes": torch.zeros((), dtype=torch.uint8).element_size()},
    "exact_arithmetic": {"code_count": 2**4, "intervals": 15, "step": str(step), "grid": [str(v) for v in grid], "exercise_grid": [str(v) for v in exercise_grid], "code_ten": str((10-5)*step)},
    "result": "所有斷言通過；全零輸入出現正文指出的除零問題。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
