"""用精確分數核對截斷算例，並記錄 CPU 上的實際 tensor 結果。"""

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.quantization import quantize_symmetric


def check_case(extreme, clipped, expected_codes, expected_scale):
    original = [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(extreme)]
    exact_restored = [value * expected_scale for value in expected_codes]
    exact_errors = [abs(value - restored) for value, restored in zip(original, exact_restored)]
    expected_metrics = [sum(exact_errors[:3]) / 3, exact_errors[-1], sum(exact_errors) / 4]
    x = torch.tensor([float(value) for value in original], dtype=torch.float32, device="cpu")
    source = x.clamp(-1, 1) if clipped else x
    codes, scale = quantize_symmetric(source, bits=4)
    restored = codes.float() * scale
    error = (x - restored).abs()
    metrics = [error[:3].mean().item(), error[-1].item(), error.mean().item()]
    assert codes.tolist() == expected_codes, "整數碼與精確算例不符"
    assert abs(scale.item() - float(expected_scale)) < 1e-6, "刻度與精確算例不符"
    for actual, expected in zip(metrics, expected_metrics):
        assert abs(actual - float(expected)) < 2e-6, "平均誤差與精確分數不符"
    return {
        "extreme": extreme,
        "clipped": clipped,
        "input": x.tolist(),
        "source": source.tolist(),
        "dtype": str(x.dtype),
        "codes": codes.tolist(),
        "code_dtype": str(codes.dtype),
        "scale": scale.item(),
        "scale_shape": list(scale.shape),
        "restored": restored.tolist(),
        "absolute_errors_against_original": error.tolist(),
        "metrics_small_mae_extreme_error_all_mae": metrics,
        "error_denominators": [3, 1, 4],
        "rounded_metrics": [round(value, 4) for value in metrics],
        "mae_against_clipped_source": (source - restored).abs().mean().item(),
        "exact_scale": str(expected_scale),
        "exact_restored": [str(value) for value in exact_restored],
        "exact_absolute_errors": [str(value) for value in exact_errors],
        "exact_metrics": [str(value) for value in expected_metrics],
        "first_three": error[:3].tolist(),
        "last": error[-1].item(),
    }


cases = [
    check_case(100, False, [0, 0, 0, 7], Fraction(100, 7)),
    check_case(100, True, [1, 1, 2, 7], Fraction(1, 7)),
    check_case(2, False, [0, 1, 1, 7], Fraction(2, 7)),
    check_case(2, True, [1, 1, 2, 7], Fraction(1, 7)),
]
assert cases[1]["restored"][:3] == cases[3]["restored"][:3], "截斷後的小值應保持相同"
assert cases[1]["source"] == cases[3]["source"], "截斷後的輸入應保持相同"
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-13/verify_quantization.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "implementation_sha256": hashlib.sha256((ROOT / "tiny_perceptron/quantization.py").read_bytes()).hexdigest(),
    "cases": cases,
    "result": "全部精確算例、索引與輸入切換核對通過",
}, ensure_ascii=False, indent=2))
