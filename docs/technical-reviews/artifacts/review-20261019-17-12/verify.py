"""核對凍結算例、練習與原始來源；全文只留在記憶體。"""

import contextlib
import hashlib
import html
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "docs/course-review-20261019/freeze-01/sources/17.12.md"
EXPECTED_SHA = "81bb0e609e9a5cd572bb55e25b52521b36019f550058bdaefadf07e3de3be179"


def capture(code):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, str(FROZEN), "exec"), namespace)
    return namespace, output.getvalue()


def inspect_quantizer(namespace, label):
    inputs = namespace["evaluation"]
    result = {}
    for name, maximum in (
        ("fixed", namespace["fixed_max"]),
        ("dynamic", namespace["dynamic_max"]),
    ):
        scale = maximum / 7
        codes = (inputs / scale).round().clamp(-7, 7)
        restored = namespace["simulate"](inputs, maximum)
        expected = torch.tensor(
            [-1 / 7, 3 / 7, 1] if float(maximum) == 1 else [0, 3 / 7, 3]
        )
        torch.testing.assert_close(restored, expected, rtol=0, atol=1e-7)
        assert int((inputs.abs() > maximum).sum()) == (1 if float(maximum) == 1 else 0)
        probe = torch.tensor([-100.0, -3.0, -1.0, 0.0, 1.0, 3.0, 100.0])
        bounded = namespace["simulate"](probe, maximum)
        assert bool((bounded.abs() <= maximum).all())
        result[name] = {
            "maximum": float(maximum),
            "scale": float(scale),
            "codes": codes.tolist(),
            "restored": restored.tolist(),
            "printed_values": restored.round(decimals=4).tolist(),
            "negative_signbit": torch.signbit(restored).tolist(),
            "overflow": int((inputs.abs() > maximum).sum()),
            "endpoint_probe": bounded.tolist(),
        }
    if label == "original":
        dynamic = namespace["simulate"](inputs, namespace["dynamic_max"])
        assert float(dynamic[0]) == 0 and bool(torch.signbit(dynamic[0]))
    else:
        fixed = namespace["simulate"](inputs, namespace["fixed_max"])
        dynamic = namespace["simulate"](inputs, namespace["dynamic_max"])
        assert torch.equal(fixed, dynamic)
        assert namespace["calibration"].numel() == 5
    return result


def receipt(identifier, url, version, locator, excerpt):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    response = subprocess.run(command, check=True, capture_output=True)
    raw = response.stdout
    original = raw.decode("utf-8")
    if url.endswith(".py"):
        text = original
    else:
        text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", original, flags=re.S | re.I)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"\s+", " ", text)
    offset = text.find(excerpt)
    assert offset >= 0, f"原始來源找不到指定摘錄：{identifier}"
    return {
        "id": identifier,
        "url": url,
        "version": version,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "command_argv": command,
        "exit_code": response.returncode,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": locator,
        "excerpt": excerpt,
        "excerpt_character_offset_in_decoded_text": offset,
        "full_text_saved": False,
    }


def main():
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    raw = FROZEN.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA
    code = re.search(r"```python\n(.*?)```", raw.decode("utf-8"), re.S).group(1)
    original, original_stdout = capture(code)
    replacement = "calibration = torch.tensor([-0.5, 0.2, 0.8, 1.0, 3.0])"
    needle = "calibration = torch.tensor([-0.5, 0.2, 0.8, 1.0])"
    assert code.count(needle) == 1
    exercise, exercise_stdout = capture(code.replace(needle, replacement))
    inputs = [Fraction(-1, 5), Fraction(2, 5), Fraction(3)]
    exact = {}
    for name, maximum in (("fixed", Fraction(1)), ("dynamic", Fraction(3))):
        scale = maximum / 7
        codes = [max(-7, min(7, round(x / scale))) for x in inputs]
        exact[name] = {
            "scale": str(scale),
            "normalized": [str(x / scale) for x in inputs],
            "codes": codes,
            "restored": [str(x * scale) for x in codes],
        }
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-12/verify.py",
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": str(torch.get_default_dtype()),
        },
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "exact_arithmetic": exact,
        "original_stdout": original_stdout,
        "original": inspect_quantizer(original, "original"),
        "exercise_stdout": exercise_stdout,
        "exercise": inspect_quantizer(exercise, "exercise"),
        "scope": "只核對此 CPU float32 算例、邊界探針與指定練習；沒有模型訓練或效能測試。",
        "source_receipts": [
            receipt(
                "onnx_docs",
                "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html",
                "2026-10-20 讀取的未標版官方網站；以 response_sha256 識別實際內容",
                "Quantization Overview；Dynamic Quantization；Static Quantization",
                "These quantization parameters are written as constants to the quantized model and used for all inputs.",
            ),
            receipt(
                "pytorch_blog",
                "https://pytorch.org/blog/quantization-in-practice/",
                "發布 2022-02-08；頁面 dateModified 2024-11-15；本次內容以 response_sha256 識別",
                "Calibration；Post-Training Dynamic/Weight-only Quantization；Post-Training Static Quantization (PTQ)",
                "The process of choosing the input clipping range is known as calibration",
            ),
            receipt(
                "guo",
                "https://proceedings.mlr.press/v70/guo17a.html",
                "ICML 2017；PMLR 70:1321–1330；作者摘要頁",
                "Abstract 首句，僅核對 confidence calibration 的定義",
                "probability estimates representative of the true correctness likelihood",
            ),
            receipt(
                "sklearn",
                "https://scikit-learn.org/1.7/modules/cross_validation.html",
                "scikit-learn 1.7.2 documentation",
                "Cross-validation: evaluating estimator performance，調參造成測試資訊洩漏的段落",
                "there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally.",
            ),
            receipt(
                "onnx_code",
                "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/python/tools/quantization/calibrate.py",
                "ONNX Runtime v1.23.2 tag",
                "MinMaxCalibrater.collect_data；HistogramCollector.collect_absolute_value；HistogramCollector.compute_percentile",
                "data_arr_np = np.absolute(data_arr_np)",
            ),
        ],
        "result": "所有數值、超出項數、符號位元、邊界與練習斷言通過；五份來源以 curl 成功讀取。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
