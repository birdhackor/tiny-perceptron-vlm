"""Bounded CPU checks for the frozen W.6 embedded code, exercises and figure."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import torch


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4/sections/W.6.md"
FIGURE = ROOT / "course/figures/rewrite-W-6-gradient-step.svg"
EXPECTED_SOURCE_SHA = "66fe3ce97fa572b26840ace318944312cc04a8cbb67813d6ba6f36c2fc8545ee"
EXPECTED_FIGURE_SHA = "e81b9f6de2af7b7d9d8bc9cbe32528794d207d1817014c46624d1619820ca942"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(observed, expected):
    assert math.isclose(observed, expected, rel_tol=0, abs_tol=2e-6), (observed, expected)


def run_embedded(code, expected_w, expected_loss):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, str(SNAPSHOT), "exec"), namespace)
    tensor = namespace["w"]
    assert tensor.device.type == "cpu"
    output = [[float(x) for x in line.split()] for line in stdout.getvalue().splitlines()]
    assert len(output) == 2
    assert output[0] == [4.0, -4.0]
    near(output[1][0], expected_w)
    near(output[1][1], expected_loss)
    assert tensor.grad.item() == -4.0
    return {
        "executed_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": stdout.getvalue(),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "w_after": tensor.item(),
        "stored_grad_after_update": tensor.grad.item(),
        "w_requires_grad_after_update": tensor.requires_grad,
        "w_is_leaf_after_update": tensor.is_leaf,
        "expected_w": expected_w,
        "expected_loss": expected_loss,
        "absolute_tolerance": 2e-6,
    }


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert digest(SNAPSHOT) == EXPECTED_SOURCE_SHA
    assert digest(FIGURE) == EXPECTED_FIGURE_SHA
    blocks = re.findall(r"```python\n(.*?)```", SNAPSHOT.read_text(), re.S)
    assert len(blocks) == 1
    code = blocks[0]
    original = run_embedded(code, 1.4, 2.56)
    exercise_results = {}
    for learning_rate, expected_w, expected_loss in [("0.6", 3.4, 0.16), ("1.1", 5.4, 5.76)]:
        replacement = f"w -= {learning_rate} * w.grad"
        assert code.count("w -= 0.1 * w.grad") == 1
        modified = code.replace("w -= 0.1 * w.grad", replacement)
        exercise_results[learning_rate] = run_embedded(modified, expected_w, expected_loss)

    # Separate observational probe; the exact embedded program above is unchanged.
    w = torch.tensor(1.0, requires_grad=True)
    initial_loss = (w - 3).square()
    initial_loss.backward()
    probe = {"w_after_backward_before_update": w.item(), "grad_after_backward": w.grad.item()}
    assert probe == {"w_after_backward_before_update": 1.0, "grad_after_backward": -4.0}
    with torch.no_grad():
        w -= 0.1 * w.grad
        untracked = (w - 3).square()
    probe["loss_inside_no_grad_requires_grad"] = untracked.requires_grad
    probe["w_after_no_grad_requires_grad"] = w.requires_grad
    assert not untracked.requires_grad and w.requires_grad and w.is_leaf
    (w - 3).square().backward()
    probe["grad_after_new_backward_without_clear"] = w.grad.item()
    near(w.grad.item(), -7.2)
    w.grad = None
    (w - 3).square().backward()
    probe["grad_after_clear_and_new_backward"] = w.grad.item()
    near(w.grad.item(), -3.2)
    probe["w_after_repeated_backwards"] = w.item()
    near(w.item(), 1.4)

    # Check every sampled SVG curve coordinate against one common linear axis map.
    svg = ET.fromstring(FIGURE.read_text())
    ns = {"s": "http://www.w3.org/2000/svg"}
    curve = next(p for p in svg.findall("s:path", ns) if p.attrib.get("d", "").startswith("M75.00 124.00"))
    points = [(float(x), float(y)) for x, y in re.findall(r"[ML]([\d.]+) ([\d.]+)", curve.attrib["d"])]
    residuals = []
    for x, y in points:
        position = (x - 75) / 123
        residual = abs(y - (430 - 34 * (position - 3) ** 2))
        residuals.append(residual)
        assert residual <= 0.014
    expected_circles = [(198, 294), (247.2, 342.96), (444, 430)]
    actual_circles = [(float(c.attrib["cx"]), float(c.attrib["cy"])) for c in svg.findall("s:circle", ns)]
    assert actual_circles == expected_circles
    for position, loss, (x, y) in zip([1, 1.4, 3], [4, 2.56, 0], actual_circles, strict=True):
        assert math.isclose(x, 75 + 123 * position, abs_tol=1e-10)
        assert math.isclose(y, 430 - 34 * loss, abs_tol=1e-10)

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-6-v1/verify_w6.py",
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "torch.float32",
            "num_threads": str(torch.get_num_threads()),
            "num_interop_threads": str(torch.get_num_interop_threads()),
        },
        "source_sha256": digest(SNAPSHOT),
        "figure_sha256": digest(FIGURE),
        "original_embedded_code": code,
        "original": original,
        "exercises": exercise_results,
        "separate_probe": probe,
        "figure_geometry": {
            "x_map": "x_px = 75 + 123*w",
            "y_map": "y_px = 430 - 34*L",
            "curve_points_checked": len(points),
            "max_curve_residual_px": max(residuals),
            "rounding_tolerance_px": 0.014,
            "circles": actual_circles,
            "arrow": "(208,299) to (235.2,330.96): right and downward in SVG coordinates; connects the update, not a tangent",
        },
        "result": "PASS: exact embedded code, both one-line exercise variants, separate backward/no_grad/accumulation probe, and all sampled figure points matched expectations",
        "scope": "Deterministic scalar CPU checks; no random data, datasets, training loop, quality measurement, GPU, or optimizer benchmark.",
    }
    output = ARTIFACT_DIR / "execution.json"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
