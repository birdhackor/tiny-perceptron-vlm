"""核對均勻量化的數值、型別、邊界與二進位表示。"""

import json
import platform
import sys
from fractions import Fraction

import torch


torch.set_default_device("cpu")
x = torch.tensor([-0.7, 0.2, 0.7, 1.2])
results = {}
for scale, expected_codes, expected_values in [
    (0.5, [-1, 0, 1, 2], [-0.5, 0.0, 0.5, 1.0]),
    (0.25, [-3, 1, 3, 5], [-0.75, 0.25, 0.75, 1.25]),
]:
    divided = x / scale
    rounded = divided.round()
    clamped = rounded.clamp(-127, 127)
    q = clamped.to(torch.int8)
    restored = q.float() * scale
    errors = x - restored
    assert q.tolist() == expected_codes
    assert restored.tolist() == expected_values
    expected_error = 0.2 if scale == 0.5 else 0.05
    assert torch.allclose(errors.abs(), torch.full_like(x, expected_error), atol=1e-7, rtol=0)
    assert q.element_size() == 1
    assert divided.dtype == rounded.dtype == clamped.dtype == torch.float32
    assert restored.dtype == torch.float32
    results[str(scale)] = {
        "divided": divided.tolist(),
        "rounded": rounded.tolist(),
        "clamped": clamped.tolist(),
        "codes": q.tolist(),
        "restored": restored.tolist(),
        "signed_error": errors.tolist(),
        "rounded_error": errors.round(decimals=4).tolist(),
        "input_dtype": str(x.dtype),
        "code_dtype": str(q.dtype),
        "restored_dtype": str(restored.dtype),
        "code_element_bytes": q.element_size(),
        "code_payload_bytes": q.untyped_storage().nbytes(),
        "positive_code_count": len(range(1, 128)),
        "negative_code_count": len(range(-127, 0)),
        "restored_endpoints": [-127 * scale, 127 * scale],
    }

ties = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
assert ties.round().tolist() == [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]
int8 = torch.iinfo(torch.int8)
assert (int8.min, int8.max) == (-128, 127)
clipping_input = torch.tensor([-100.0, -63.75, -63.5, 63.5, 63.75, 100.0])
clipping_codes = (clipping_input / 0.5).round().clamp(-127, 127).to(torch.int8)
clipping_restored = clipping_codes.float() * 0.5
clipping_errors = clipping_input - clipping_restored
assert abs(clipping_errors[0].item()) > 0.25
assert abs(clipping_errors[-1].item()) > 0.25

# 用整數分子與分母核對界限，不把浮點近似當成恆等式。
exact_inputs = [Fraction(-7, 10), Fraction(1, 5), Fraction(7, 10), Fraction(6, 5)]
exact_examples = {}
for scale in [Fraction(1, 2), Fraction(1, 4)]:
    codes = [round(value / scale) for value in exact_inputs]
    restored = [code * scale for code in codes]
    errors = [value - value_hat for value, value_hat in zip(exact_inputs, restored)]
    assert all(abs(error) <= scale / 2 for error in errors)
    exact_examples[str(scale)] = {
        "inputs": [str(value) for value in exact_inputs],
        "codes": codes,
        "restored": [str(value) for value in restored],
        "signed_errors": [str(value) for value in errors],
        "half_scale": str(scale / 2),
        "positive_endpoint": str(127 * scale),
    }

p = torch.tensor(0.2, dtype=torch.float32)
nearest = Fraction.from_float(p.item())
lower = Fraction.from_float(torch.nextafter(p, torch.tensor(float("-inf"))).item())
upper = Fraction.from_float(torch.nextafter(p, torch.tensor(float("inf"))).item())
target = Fraction(1, 5)
assert abs(nearest - target) < abs(lower - target)
assert abs(nearest - target) < abs(upper - target)
assert nearest == Fraction(13421773, 67108864)
assert torch.equal(p.round(decimals=4), p)

report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-2/verify_quantization.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
        "default_dtype": str(torch.get_default_dtype()),
    },
    "examples": results,
    "ties": {"input": ties.tolist(), "rounded": ties.round().tolist()},
    "int8_range": {"min": int8.min, "max": int8.max, "bits": int8.bits},
    "clipping": {
        "input": clipping_input.tolist(),
        "codes": clipping_codes.tolist(),
        "restored": clipping_restored.tolist(),
        "signed_errors": clipping_errors.tolist(),
    },
    "exact_rational_examples": exact_examples,
    "error_bound_derivation": "對正刻度 s，最近整數 q 使 |x/s-q|≤1/2；未截斷時乘上 s 得 |x-sq|≤s/2。截斷後 q 可能不再是最近整數，因此不保證此界限。",
    "information_loss_derivation": "s=1/2 時，x=3/5 與 x=7/10 都取碼1；單憑碼與刻度不能辨認原值。",
    "binary_fraction_derivation": "有限二進位小數的最簡分母只能含質因數2；1/5 的分母含5，因此不能精確表示。",
    "fp32_point_two": {
        "python_repr": repr(p.item()),
        "stored_fraction": str(nearest),
        "lower_neighbor": str(lower),
        "upper_neighbor": str(upper),
        "distance_to_exact_point_two": str(nearest - target),
        "rounded_decimals_four_repr": repr(p.round(decimals=4).item()),
    },
    "assertions": "全部通過",
    "scope": "只核對指定的有限輸入、CPU 環境與有理數推導；不測模型品質、GPU 或效能。每碼1 byte僅指元素資料，不包括 Tensor 物件額外成本。",
}
print(json.dumps(report, ensure_ascii=False, indent=2))
