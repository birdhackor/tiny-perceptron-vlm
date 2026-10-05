"""Short CPU checks for section 3.6, including its exact original fence."""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.attention import attention_mask, manual_attention

torch.set_num_threads(1)
torch.set_default_device("cpu")
OUT = Path(__file__).resolve().parent
original = (OUT / "original/fence-1.py").read_bytes()
namespace = {"__name__": "__main__"}
exec(compile(original, "original/fence-1.py", "exec"), namespace)
q, k, v, allowed, out, weights, changed, other = [namespace[name] for name in ("q", "k", "v", "allowed", "out", "weights", "changed", "other")]

mask_expected = torch.arange(4)[None, :] <= torch.arange(4)[:, None]
assert allowed.dtype == torch.bool and tuple(allowed.shape) == (1, 1, 4, 4)
assert torch.equal(allowed[0, 0], mask_expected)
assert torch.equal(attention_mask(torch.arange(4), torch.arange(4)), allowed)
assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (1, 1, 4, 2)
assert tuple(weights.shape) == (1, 1, 4, 4) and tuple(out.shape) == (1, 1, 4, 2)
expected_weights = mask_expected.double() / torch.arange(1, 5).double()[:, None]
expected_out = torch.tensor([[1, 10], [1.5, 15], [2, 20], [2.5, 25]], dtype=torch.float64)
assert torch.allclose(weights[0, 0].double(), expected_weights, atol=1e-6, rtol=1e-6)
assert torch.allclose(out[0, 0].double(), expected_out, atol=1e-6, rtol=1e-6)
assert torch.equal(weights[0, 0][~mask_expected], torch.zeros(6))
assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 4), atol=1e-6, rtol=1e-6)
assert torch.equal(out[:, :, :3], other[:, :, :3])
assert torch.equal(other[:, :, 3] - out[:, :, 3], torch.tensor([[[25.0, 25.0]]]))
assert torch.equal(v, torch.tensor([[[[1., 10.], [2., 20.], [3., 30.], [4., 40.]]]]))
assert v.data_ptr() != changed.data_ptr()
assert not any(t.requires_grad or t.grad_fn is not None for t in (q, k, v, out, weights))

# Execute precisely the proposed exercise edit, retaining its expected failing assertion.
exercise = original.replace(b".tril()", b"")
(OUT / "variant-without-tril.py").write_bytes(exercise)
exercise_namespace = {"__name__": "__main__"}
exercise_stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(exercise_stdout):
        exec(compile(exercise, "variant-without-tril.py", "exec"), exercise_namespace)
except AssertionError:
    expected_exercise_assertion_failed = True
else:
    raise AssertionError("Removing tril did not expose the future card")
(OUT / "variant-without-tril.stdout.txt").write_text(exercise_stdout.getvalue())
exercise_delta = exercise_namespace["other"] - exercise_namespace["out"]
assert torch.equal(exercise_delta, torch.full((1, 1, 4, 2), 25.0))

# A score of zero is not an exclusion: e^0 = 1, so all four candidates receive 1/4.
zero_filled_scores = (q @ k.transpose(-2, -1)).masked_fill(~allowed, 0.0)
wrong_weights = zero_filled_scores.softmax(-1)
assert torch.equal(wrong_weights, torch.full((1, 1, 4, 4), 0.25))
assert bool((wrong_weights[0, 0][~mask_expected] > 0).all())

# Explicit row/column alignment: permitting only diagonal key j returns V[j].
identity_out, identity_weights = manual_attention(q, k, v, torch.eye(4, dtype=torch.bool)[None, None])
assert torch.equal(identity_out, v)
assert torch.equal(identity_weights[0, 0], torch.eye(4))

# Nonzero, nonuniform scores avoid relying solely on the q=k=0 demonstration.
nonzero_q = torch.tensor([[[[1., -1.], [.5, 2.], [2., 1.], [-1., 1.]]]])
nonzero_k = torch.tensor([[[[.5, 1.], [2., 0.], [-1., 2.], [1., -.5]]]])
nonzero_out, nonzero_weights = manual_attention(nonzero_q, nonzero_k, v, allowed)
future_k = nonzero_k.clone()
future_k[:, :, 3] += torch.tensor([50., -60.])
nonzero_other, _ = manual_attention(nonzero_q, future_k, changed, allowed)
assert torch.equal(nonzero_out[:, :, :3], nonzero_other[:, :, :3])
assert not torch.allclose(nonzero_out[:, :, 3], nonzero_other[:, :, 3])
assert torch.equal(nonzero_weights[0, 0][~mask_expected], torch.zeros(6))
sdpa = F.scaled_dot_product_attention(nonzero_q, nonzero_k, v, attn_mask=allowed, dropout_p=0.0)
assert torch.allclose(nonzero_out, sdpa, rtol=1e-6, atol=1e-6)

# Low loss alone cannot certify causal access: deliberately copy the unseen targets.
tokens = list("貓看狗。")
pairs = list(zip(tokens[:-1], tokens[1:], strict=True))
assert pairs == [("貓", "看"), ("看", "狗"), ("狗", "。")]
targets = torch.tensor([1, 2, 3])
copied_logits = F.one_hot(targets, num_classes=4).float() * 20.0
leaky_mean_loss = F.cross_entropy(copied_logits, targets).item()
assert leaky_mean_loss < 1e-6

result = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": str(q.device), "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads())},
    "original_fence_executed": True,
    "mask": allowed[0, 0].tolist(),
    "qkv_shape": list(q.shape),
    "weights_shape": list(weights.shape),
    "out_shape": list(out.shape),
    "weights": weights[0, 0].tolist(),
    "row_denominators": [1, 2, 3, 4],
    "weights_row_sums": weights.sum(-1).tolist(),
    "out": out[0, 0].tolist(),
    "changed_out": other[0, 0].tolist(),
    "future_delta": (other - out)[0, 0].tolist(),
    "clone_preserves_original": True,
    "maximum_numeric_error": max((weights[0, 0].double() - expected_weights).abs().max().item(), (out[0, 0].double() - expected_out).abs().max().item()),
    "numeric_tolerance": {"atol": 1e-6, "rtol": 1e-6},
    "exercise_expected_assertion_failed": expected_exercise_assertion_failed,
    "unmasked_delta": exercise_delta[0, 0].tolist(),
    "ordinary_zero_mask_future_weights": wrong_weights[0, 0][~mask_expected].tolist(),
    "diagonal_position_alignment_verified": True,
    "nonzero_future_KV_first_three_equal": True,
    "nonzero_future_KV_last_changes": True,
    "sdpa_maximum_error": (nonzero_out - sdpa).abs().max().item(),
    "has_grad_or_parameter_update": False,
    "shift_pairs": pairs,
    "deliberately_leaky_mean_loss": leaky_mean_loss,
    "leaky_loss_denominator": "three target positions; four character classes; mean reduction; synthetic logits copied from targets; zero training steps",
    "scope": "Short CPU mechanism checks; no training, model quality, tokenizer, cache, padded rows, document packing, or end-to-end decoder validation."
}
(OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
