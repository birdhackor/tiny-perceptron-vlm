"""僅補核已釋出的歷史 allclose 問題；不重跑單音、模型或 Notebook。"""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOTS = [
    (
        "original_before_fix",
        "docs/course-review-20261019/freeze-01/sources/12.2.md",
        "499c8e3011dcff0268b30d9cd4e9b68967a9f9cea8d4cf532e496fba35438290",
    ),
    (
        "historical_fix",
        "docs/course-review-20261019/freeze-05/sources/12.2.md",
        "05eb1ea9b9088d0f2496f59613bdbb461133c76e9a05819f509d1c6c63162945",
    ),
    (
        "current",
        "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14/sources/12.2.md",
        "5ed63e8c806b1082c3e8349b90bd736e768620a56d0a98d774ad9e39ea4d566e",
    ),
]


def explanation(text):
    paragraph = next(line for line in text.splitlines() if line.startswith("兩段各 16 點。"))
    return paragraph[paragraph.index("`torch.allclose"):paragraph.index("印 True。") + len("印 True。")]


def main():
    sources = []
    for stage, relative, expected in SNAPSHOTS:
        raw = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == expected
        sources.append({"stage": stage, "path": relative, "sha256": digest, "allclose_explanation": explanation(raw.decode())})
    assert "容許十萬分之一以內的浮點誤差" in sources[0]["allclose_explanation"]
    assert sources[1]["allclose_explanation"] == sources[2]["allclose_explanation"]
    assert "`rtol` 沒寫，用預設值 1e-5" in sources[2]["allclose_explanation"]
    assert "`-low` 那一點大小的 1e-5 倍" in sources[2]["allclose_explanation"]
    cases = []
    for sign in [1, -1]:
        for delta, expected in [(0.0000125, True), (0.00002, False)]:
            other = torch.tensor([sign * 0.5], dtype=torch.float64)
            value = torch.tensor([sign * (0.5 + delta)], dtype=torch.float64)
            residual = (value - other).abs()
            threshold = 1e-5 + 1e-5 * other.abs()
            formula = bool((residual <= threshold).all())
            default_rtol = torch.allclose(value, other, atol=1e-5)
            zero_rtol = torch.allclose(value, other, atol=1e-5, rtol=0)
            assert formula == default_rtol == expected
            assert zero_rtol is False
            cases.append({"input": value.item(), "other": other.item(), "nominal_delta": delta, "actual_abs_difference": residual.item(), "absolute_only_limit": 1e-5, "atol_plus_rtol_abs_other_limit": threshold.item(), "explicit_formula_result": formula, "allclose_default_rtol": default_rtol, "allclose_rtol_zero": zero_rtol})
    print(json.dumps({"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_p8_technical_audio122", "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "torch.float64"}, "source_explanation_comparison": sources, "historical_fix_explanation_identical_to_current": True, "threshold_cases": cases, "scope": "Four scalar-array pairs from the released historical allclose counterexamples, plus JSON/text/hash checks. No tone regeneration, real-audio resampling, neural forward, training, model or notebook execution.", "assertions_passed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
