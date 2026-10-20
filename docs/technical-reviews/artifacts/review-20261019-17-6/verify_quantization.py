"""核對凍結算例、練習與有理數參考值，保存本次實際執行結果。"""

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.quantization import pack_int4, quantize_symmetric


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_reference(values, bits):
    maximum = 2 ** (bits - 1) - 1
    scale = max(abs(v) for v in values) / maximum
    codes = [max(-maximum, min(maximum, round(v / scale))) for v in values]
    restored = [code * scale for code in codes]
    errors = [abs(v - r) for v, r in zip(values, restored)]
    return {
        "scale": str(scale),
        "codes": codes,
        "MAE": str(sum(errors) / len(errors)),
        "max_error": str(max(errors)),
        "max_error_as_scale_ratio": str(max(errors) / scale),
    }


def check_dataset(weight, rational_values, label):
    result = {"name": label, "count": weight.numel(), "input": weight.tolist(), "results": []}
    for bits in (8, 4):
        q, scale = quantize_symmetric(weight, bits)
        restored = q.float() * scale
        error = (weight - restored).abs()
        reference = exact_reference(rational_values, bits)
        exact_scale = Fraction(reference["scale"])
        differences = []
        for value, actual, expected in zip(rational_values, q.tolist(), reference["codes"]):
            if abs(value / exact_scale - actual) > Fraction(1, 2):
                raise AssertionError("實際量化碼不是有理數參考值的最近整數")
            if actual != expected:
                differences.append({"input": str(value), "actual_code": actual, "exact_ties_even_code": expected})
        if abs(error.mean().item() - float(Fraction(reference["MAE"]))) > 1e-7:
            raise AssertionError("平均誤差超出 FP32 核對容忍差")
        if abs(error.max().item() - float(Fraction(reference["max_error"]))) > 1e-7:
            raise AssertionError("最大誤差超出 FP32 核對容忍差")
        result["results"].append({
            "bits": bits,
            "input_dtype": str(weight.dtype),
            "device": str(weight.device),
            "q_dtype": str(q.dtype),
            "q_shape": list(q.shape),
            "q": q.tolist(),
            "scale": scale.item(),
            "scale_numel": scale.numel(),
            "scale_bytes": scale.numel() * scale.element_size(),
            "restored": restored.tolist(),
            "absolute_errors": error.tolist(),
            "MAE": error.mean().item(),
            "max_error": error.max().item(),
            "rounded_MAE": round(error.mean().item(), 6),
            "rounded_max_error": round(error.max().item(), 6),
            "max_error_minus_half_scale": error.max().item() - scale.item() / 2,
            "zero_and_endpoints_error": error[[0, weight.numel() // 2, -1]].tolist(),
            "ideal_code_bytes": (weight.numel() * bits + 7) // 8,
            "container_bytes": q.numel() * q.element_size(),
            "element_size": q.element_size(),
            "rational_reference": reference,
            "rational_tie_code_differences": differences,
        })
    return result


def main():
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    run = Path(sys.argv[1]).resolve()
    if not run.is_relative_to(ROOT / "outputs/reviewer-tools/runs"):
        raise ValueError("只接收本次工具的執行目錄")
    frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/17.6.md"
    expected_hash = "f14f8c78948d0edbf759a222cf46683d12d46ad365b4a424f2aac2487b0be14a"
    if digest(frozen) != expected_hash or digest(run / "section.md") != expected_hash:
        raise AssertionError("執行原文與指定凍結原文不同")
    copied = {}
    for name in ("stdout.txt", "stderr.txt", "execution.json", "environment.json", "extraction.json"):
        path = run / name
        copied[name] = {"sha256": digest(path), "content": path.read_text(encoding="utf-8")}
    bundle = {
        "invocation": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/17.md#17.6" --execute',
        "original_directory": str(run.relative_to(ROOT)),
        "copy_policy": "Original execution output and metadata are preserved verbatim as JSON strings.",
        "files": copied,
    }
    (DIRECTORY / "section-run.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    main_result = check_dataset(torch.linspace(-1, 1, 33), [Fraction(i, 16) for i in range(-16, 17)], "linspace")
    exercise = check_dataset(torch.arange(-7, 8).float() / 7, [Fraction(i, 7) for i in range(-7, 8)], "four_bit_grid")
    codes, _ = quantize_symmetric(torch.linspace(-1, 1, 33), 4)
    packed = pack_int4(codes)
    weight = torch.tensor([1 / 16, 1.0])
    q, scale = quantize_symmetric(weight, 4)
    restored = q.float() * scale
    inputs = torch.tensor([[0.0, 1.0], [1.0, -1 / 32]])
    scores = inputs @ weight
    quantized_scores = inputs @ restored
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-6/verify_quantization.py " + str(run.relative_to(ROOT)),
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "platform": platform.platform(),
            "device": "cpu",
            "default_dtype": str(torch.get_default_dtype()),
        },
        "source_sha256": expected_hash,
        "implementation_path": "tiny_perceptron/quantization.py",
        "implementation_sha256": digest(ROOT / "tiny_perceptron/quantization.py"),
        "datasets": [main_result, exercise],
        "packing": {"bits": 132, "quotient_139_divided_by_8": 139 // 8, "bytes": packed.numel() * packed.element_size(), "unused_bits": 8 * packed.numel() - 132},
        "prediction_counterexample": {
            "definition": "兩個輸出 token 的 logits 為 (0, x dot weight)，取較大的 logit；兩份資料都以原權重的預測為正解。",
            "weight": weight.tolist(),
            "restored_weight": restored.tolist(),
            "inputs": inputs.tolist(),
            "original_scores": scores.tolist(),
            "quantized_scores": quantized_scores.tolist(),
            "original_predictions": (scores > 0).tolist(),
            "quantized_predictions": (quantized_scores > 0).tolist(),
            "same_weight_MAE": (weight - restored).abs().mean().item(),
        },
        "result": "實際碼都是有理數參考值的最近整數之一；FP32 半格決策差異另列，MAE 與最大誤差的 FP32 差距均小於 1e-7。",
    }
    (DIRECTORY / "verification-output.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
