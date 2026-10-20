"""獨立核對對稱量化的算例、張量形狀與刻度儲存量。"""

import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.quantization import quantize_symmetric


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_quantize(rows, per_row):
    maxima = [max(abs(x) for x in row) for row in rows]
    scales = [value / 7 for value in maxima] if per_row else [max(maxima) / 7] * len(rows)
    codes = [[round(value / scale) for value in row] for row, scale in zip(rows, scales)]
    errors = [
        sum(abs(value - code * scale) for value, code in zip(row, integers)) / len(row)
        for row, integers, scale in zip(rows, codes, scales)
    ]
    return codes, scales, errors


def main():
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/17.5.md"
    implementation = ROOT / "tiny_perceptron/quantization.py"
    text = frozen.read_text()
    fences = re.findall(r"```python\n(.*?)\n```", text, re.S)
    shown_function = next(node for node in ast.parse(fences[1]).body if isinstance(node, ast.FunctionDef))
    actual_function = next(
        node for node in ast.parse(implementation.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "quantize_symmetric"
    )
    assert ast.dump(shown_function) == ast.dump(actual_function)
    rows = [list(map(Fraction, ("0.01", "0.03", "0.05", "0.08"))), list(map(Fraction, (10, 30, 50, 80)))]
    weight = torch.tensor([[float(x) for x in row] for row in rows], dtype=torch.float32)
    output = {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": "torch.float32"},
        "input_sha256": {str(frozen.relative_to(ROOT)): digest(frozen), str(implementation.relative_to(ROOT)): digest(implementation)},
        "shown_function_ast_matches_repository": True,
        "cases": [],
    }
    for per_row in (False, True):
        codes, scales, errors = exact_quantize(rows, per_row)
        q, scale = quantize_symmetric(weight, bits=4, per_channel=per_row)
        restored = q.float() * scale
        observed = (weight - restored).abs().mean(-1)
        assert q.tolist() == codes
        assert torch.allclose(observed, torch.tensor([float(x) for x in errors]), rtol=0, atol=1e-5)
        output["cases"].append({
            "per_row": per_row,
            "codes": q.tolist(),
            "exact_scales_per_row": [str(x) for x in scales],
            "exact_mae": [str(x) for x in errors],
            "actual_scales": scale.tolist(),
            "actual_scale_shape": list(scale.shape),
            "actual_mae": observed.tolist(),
            "restored": restored.tolist(),
            "mean_denominator": weight.shape[-1],
            "scale_count": scale.numel(),
            "scale_bytes": scale.numel() * scale.element_size(),
        })
    equal_rows = weight[1:].repeat(2, 1)
    q_global, s_global = quantize_symmetric(equal_rows, bits=4, per_channel=False)
    q_row, s_row = quantize_symmetric(equal_rows, bits=4, per_channel=True)
    assert torch.equal(q_global, q_row)
    assert torch.equal(q_global.float() * s_global, q_row.float() * s_row)
    output["equal_rows_exercise"] = {
        "codes": q_global.tolist(),
        "mae_global": (equal_rows - q_global.float() * s_global).abs().mean(-1).tolist(),
        "mae_per_row": (equal_rows - q_row.float() * s_row).abs().mean(-1).tolist(),
        "scale_count_global": s_global.numel(), "scale_count_per_row": s_row.numel(),
    }
    output["zero_rows"] = []
    for bits in (4, 8):
        for per_row in (False, True):
            q, scale = quantize_symmetric(torch.zeros((2, 4), dtype=torch.float32), bits, per_row)
            assert (scale > 0).all() and torch.isfinite(scale).all()
            assert not q.any()
            assert torch.allclose(scale, torch.full_like(scale, 1e-8 / (2 ** (bits - 1) - 1)), rtol=1e-6)
            output["zero_rows"].append({"bits": bits, "per_row": per_row, "scale": scale.tolist(), "codes": q.tolist()})
    output["code_endpoints"] = []
    for bits in (4, 8):
        q, scale = quantize_symmetric(torch.tensor([[-1.0, 0.0, 1.0]]), bits, True)
        maximum = 2 ** (bits - 1) - 1
        assert q.tolist() == [[-maximum, 0, maximum]] and q.dtype == torch.int8
        output["code_endpoints"].append({"bits": bits, "maximum": maximum, "codes": q.tolist(), "dtype": str(q.dtype)})
    output["invalid_bits"] = []
    for bits in (0, 2, 16):
        try:
            quantize_symmetric(weight, bits)
        except ValueError as error:
            output["invalid_bits"].append({"bits": bits, "type": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError("不支援的位元數應拒絕")
    output["fp32_element_bytes"] = torch.tensor([0.0], dtype=torch.float32).element_size()
    assert output["fp32_element_bytes"] == 4
    output["per_weight_storage"] = {
        "fp32_original_bits_per_weight": 32,
        "packed_4bit_code_plus_fp32_scale_bits_per_weight": 4 + 32,
        "int8_container_plus_fp32_scale_bits_per_weight": 8 + 32,
        "excludes": "張量物件開銷、對齊與其他格式資訊",
    }
    output["round_ties"] = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]).round().tolist()
    output["uniform_group_scale_counts"] = {str(group_size): 128 // group_size for group_size in (128, 64, 32, 1)}
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
