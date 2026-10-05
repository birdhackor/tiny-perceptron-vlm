"""Independent rational arithmetic, exercise, and dtype/storage checks; no model training."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
from tiny_perceptron.quantization import quantize_symmetric

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()

def tensor_record(t):
    # These are short CPU tensors, not neural weights or model artifacts.
    return {
        "values": t.tolist(),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": t.numel(),
        "element_size": t.element_size(),
        "storage_nbytes": t.untyped_storage().nbytes(),
        "raw_bytes_sha256": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(),
    }

rows = []
w = torch.linspace(-1, 1, 33)
assert w.tolist() == [i / 16 for i in range(-16, 17)]
for bits in (8, 4):
    maximum = 2 ** (bits - 1) - 1
    q, scale = quantize_symmetric(w, bits)
    restored = q.float() * scale
    error = (w - restored).abs()
    rational_q = [round(Fraction(i * maximum, 16)) for i in range(-16, 17)]
    rational_error = [abs(Fraction(i, 16) - Fraction(code, maximum)) for i, code in zip(range(-16, 17), rational_q)]
    rational_mae = sum(rational_error) / len(rational_error)
    rational_max = max(rational_error)
    tie_differences = []
    for i, actual, ideal in zip(range(-16, 17), q.tolist(), rational_q):
        exact_scaled = Fraction(i * maximum, 16)
        assert abs(Fraction(actual) - exact_scaled) <= Fraction(1, 2)
        if actual != ideal:
            assert exact_scaled.denominator == 2
            tie_differences.append({"index_i": i, "exact_scaled": str(exact_scaled),
                                    "float32_scaled": (w / scale)[i + 16].item(),
                                    "actual_code": actual, "exact_ties_even_code": ideal})
    assert abs(error.mean().item() - float(rational_mae)) < 1e-7
    assert abs(error.max().item() - float(rational_max)) < 1e-7
    assert q.dtype == torch.int8 and q.element_size() == 1
    assert q.numel() == 33 and q.untyped_storage().nbytes() == 33
    assert scale.dtype == torch.float32 and scale.numel() == 1 and scale.element_size() == 4
    rows.append({
        "bits": bits,
        "maximum": maximum,
        "w": tensor_record(w), "q": tensor_record(q), "scale": tensor_record(scale),
        "restored": tensor_record(restored), "abs_error": tensor_record(error),
        "mae": error.mean().item(), "max_abs_error": error.max().item(),
        "rational_mae": str(rational_mae), "rational_max_abs_error": str(rational_max),
        "fp32_half_grid_code_differences": tie_differences,
        "ideal_code_bytes": (33 * bits + 7) // 8,
        "ideal_code_bits": 33 * bits,
        "allocated_q_bytes": q.untyped_storage().nbytes(),
        "q_plus_scale_tensor_payload_bytes": q.untyped_storage().nbytes() + scale.untyped_storage().nbytes(),
    })
exercise_w = torch.arange(-7, 8).float() / 7
exercise_q, exercise_scale = quantize_symmetric(exercise_w, 4)
exercise_restored = exercise_q.float() * exercise_scale
exercise_error = (exercise_w - exercise_restored).abs()
assert exercise_q.tolist() == list(range(-7, 8))
assert exercise_error.max().item() < 1e-7
exercise = {"w": tensor_record(exercise_w), "q": tensor_record(exercise_q),
            "scale": tensor_record(exercise_scale), "restored": tensor_record(exercise_restored),
            "mae": exercise_error.mean().item(), "max_abs_error": exercise_error.max().item(),
            "denominator": exercise_w.numel()}
# Hold the calibration extrema at +/-1 so both bit settings use the same range.
counterexample_w = torch.tensor([-1.0, 1/7, 0.0, 1.0])
counterexample = []
for bits in (8, 4):
    q, scale = quantize_symmetric(counterexample_w, bits)
    restored = q.float() * scale
    errors = (counterexample_w - restored).abs()
    counterexample.append({"bits": bits, "w": tensor_record(counterexample_w),
                           "restored": tensor_record(restored), "abs_error": tensor_record(errors)})
assert counterexample[1]["abs_error"]["values"][1] == 0.0
assert counterexample[0]["abs_error"]["values"][1] > 0.001
assert counterexample[0]["abs_error"]["values"][0] == 0.0
assert counterexample[0]["abs_error"]["values"][2] == 0.0
assert counterexample[0]["abs_error"]["values"][3] == 0.0
out = {"scope": "Synthetic CPU arithmetic and storage payload only; no model evaluation, training, or serialization.",
       "tolerance": "absolute <1e-7 against exact rational arithmetic; integer codes/storage exact",
       "original_denominator": 33, "original": rows, "exercise": exercise, "counterexample": counterexample}
artifact = Path(__file__).resolve().parent
(artifact / "crosscheck-results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"original": [{k: r[k] for k in ['bits','mae','max_abs_error','rational_mae','rational_max_abs_error','ideal_code_bytes','allocated_q_bytes','q_plus_scale_tensor_payload_bytes']} for r in rows],
                  "exercise": {k: exercise[k] for k in ['mae','max_abs_error','denominator']},
                  "counterexample_errors": [r['abs_error']['values'] for r in counterexample],
                  "all_assertions_passed": True}, indent=2))
