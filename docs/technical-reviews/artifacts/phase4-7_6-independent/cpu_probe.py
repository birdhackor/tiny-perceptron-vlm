"""Bounded independent CPU checks for section 7.6, without optimizer/model data."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import torch
from torch.nn import functional as F
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum, masked_loss

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
output = Path(__file__).resolve().parent
environment = {"python": sys.version, "python_executable": sys.executable,
               "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
               "device": "cpu", "cuda_build": str(torch.version.cuda),
               "cuda_available": str(torch.cuda.is_available()), "cwd": str(Path.cwd()),
               "dtype": "float64", "threads": str(torch.get_num_threads()),
               "downloads": "none", "model_forward": "none", "optimizer_steps": "0"}
(output / "probe-environment.json").write_text(json.dumps(environment, indent=2) + "\n")

a = torch.tensor([[[0., 2., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
labels = torch.tensor([[1, 2]])
p1 = math.exp(2) / (math.exp(2) + 3)
per_token_expected = [math.log(math.exp(2) + 3) - 2, math.log(4)]
mean_expected = sum(per_token_expected) / 2
total, count = loss_sum(a, labels)
mean = masked_loss(a, labels)
assert count.item() == 2
assert math.isclose(total.item(), sum(per_token_expected), abs_tol=1e-12)
assert math.isclose(mean.item(), mean_expected, abs_tol=1e-12)
assert torch.allclose(mean, F.cross_entropy(a.reshape(-1, 4), labels.reshape(-1)), atol=1e-12, rtol=0)
mean.backward()
expected_gradient = a.detach().softmax(-1)
expected_gradient[0, 0, 1] -= 1
expected_gradient[0, 1, 2] -= 1
expected_gradient /= 2
assert torch.allclose(a.grad, expected_gradient, atol=1e-12, rtol=0)

extended = torch.tensor([[1, 2, -100, -100, -100]])
padding_logits = torch.tensor([[[123., -456., 77., 20.], [-3., 2., 9., -8.], [1000., 0., -1000., 4.]]])
b = torch.cat([a.detach(), torch.zeros(1, 3, 4)], dim=1).requires_grad_()
padded_loss = masked_loss(b, extended)
padded_loss.backward()
assert tuple(b.shape) == (1, 5, 4)
assert padded_loss.item() == mean.item()
assert torch.equal(b.grad[:, :2], a.grad)
assert torch.count_nonzero(b.grad[:, 2:]).item() == 0
arbitrary = torch.cat([a.detach(), padding_logits], dim=1).requires_grad_()
arbitrary_loss = masked_loss(arbitrary, extended)
arbitrary_loss.backward()
assert arbitrary_loss.item() == mean.item()
assert torch.equal(arbitrary.grad[:, :2], a.grad)
assert torch.count_nonzero(arbitrary.grad[:, 2:]).item() == 0

wrong = extended.clone()
wrong[0, 2] = 0
c = b.detach().clone().requires_grad_()
wrong_total, wrong_count = loss_sum(c, wrong)
wrong_loss = masked_loss(c, wrong)
wrong_loss.backward()
wrong_expected = (sum(per_token_expected) + math.log(4)) / 3
assert wrong_count.item() == 3
assert math.isclose(wrong_loss.item(), wrong_expected, abs_tol=1e-12)
assert not torch.allclose(mean, wrong_loss)
assert torch.count_nonzero(c.grad[0, 2]).item() == 4
assert torch.allclose(c.grad[:, :2], a.grad * (2 / 3), atol=1e-12, rtol=0)
assert torch.count_nonzero(c.grad[:, 3:]).item() == 0
restored = masked_loss(c.detach(), extended)
assert restored.item() == mean.item()

all_ignored = torch.full((1, 5), -100, dtype=torch.long)
try:
    masked_loss(b.detach(), all_ignored)
except ValueError as error:
    all_ignored_error = str(error)
else:
    raise AssertionError("all ignored must be rejected before division")
vanilla_mean = F.cross_entropy(b.detach().reshape(-1, 4), all_ignored.reshape(-1), ignore_index=-100)
assert torch.isnan(vanilla_mean)
vanilla_sum = F.cross_entropy(b.detach().reshape(-1, 4), all_ignored.reshape(-1), ignore_index=-100, reduction="sum")
assert vanilla_sum.item() == 0

# One row has two supervised targets; the other has three. All real context is
# valid for attention, including user positions ignored by the loss.
messages = [
    [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
    [{"role": "user", "content": "QQ"}, {"role": "assistant", "content": "BC"}],
]
examples = [render_chat(m) for m in messages]
inputs, y, valid = pad_batch(examples)
assert inputs.shape == y.shape == valid.shape == (2, 8)
assert valid.sum().item() == 14 and (y != -100).sum().item() == 5
assert torch.equal(inputs[0, 6:], torch.tensor([0, 0]))
assert torch.equal(y[0, 6:], torch.tensor([-100, -100]))
assert torch.equal(valid[0, 6:], torch.tensor([False, False]))
first = [(e[1] != -100).nonzero()[0].item() for e in examples]
assert first == [4, 5]
assert [examples[i][0][first[i]].item() for i in range(2)] == [4, 4]
assert [examples[i][1][first[i]].item() for i in range(2)] == [73, 74]
allowed = attention_mask(torch.arange(8), torch.arange(8), valid)
assert allowed[0, 0, 4, 2] and y[0, 2].item() == -100
assert not allowed[0, 0, :, 6:].any()

toy_scores = torch.zeros(2, 8, 264)
toy_scores[0, :, 73] = 2
batch_total, batch_count = loss_sum(toy_scores, y)
row_totals, row_counts = zip(*(loss_sum(toy_scores[i:i+1], y[i:i+1]) for i in range(2)))
weighted = sum(row_totals) / sum(row_counts)
unweighted_row_mean = torch.stack([t / n for t, n in zip(row_totals, row_counts)]).mean()
assert batch_count.item() == 5
assert torch.allclose(masked_loss(toy_scores, y), weighted, atol=1e-12, rtol=0)
assert not torch.allclose(weighted, unweighted_row_mean, atol=1e-12, rtol=0)

result = {
    "input_provenance": "literal tensors and two literal Q/A and QQ/BC messages in cpu_probe.py; no corpus, weights, model forward, downloads or optimizer",
    "axes": {"original": [1, 2, 4], "extended": [1, 5, 4], "names": ["batch", "position", "candidate"], "cat_dim": 1},
    "units": "dimensionless probability; loss is negative natural logarithm, nats per supervised target",
    "independent_math": {"p1": p1, "per_token_nll": per_token_expected, "sum_nll": sum(per_token_expected), "mean_nll": mean_expected, "wrong_pad_mean": wrong_expected, "rounding": {"p1": round(p1, 3), "loss1": round(per_token_expected[0], 3), "loss2": round(per_token_expected[1], 3), "mean": round(mean_expected, 3), "wrong_pad_mean": round(wrong_expected, 3)}},
    "original_observed": {"sum": total.item(), "count": count.item(), "mean": mean.item(), "gradient": a.grad.tolist()},
    "extended_observed": {"zero_pad_mean": padded_loss.item(), "arbitrary_finite_scores_mean": arbitrary_loss.item(), "ignored_gradient_nonzero_count": torch.count_nonzero(b.grad[:, 2:]).item(), "valid_gradient_equal": torch.equal(b.grad[:, :2], a.grad)},
    "pad_label_variation": {"labels": wrong.tolist(), "sum": wrong_total.item(), "count": wrong_count.item(), "mean": wrong_loss.item(), "new_gradient": c.grad[0, 2].tolist(), "old_gradient_scale": "2/3", "restored_mean": restored.item()},
    "all_ignored": {"repo_error": all_ignored_error, "torch_mean_is_nan": bool(torch.isnan(vanilla_mean)), "torch_sum": vanilla_sum.item(), "effective_count": 0},
    "batch_contract": {"messages": messages, "examples": [{"x": x.tolist(), "y": y.tolist()} for x, y in examples], "padded_x": inputs.tolist(), "padded_y": y.tolist(), "valid": valid.tolist(), "valid_count": valid.sum().item(), "supervised_count": batch_count.item(), "first_prediction_positions": first, "row_counts": [n.item() for n in row_counts], "weighted_token_mean": weighted.item(), "unweighted_example_mean": unweighted_row_mean.item(), "attention_user_key_allowed": bool(allowed[0, 0, 4, 2]), "padding_keys_allowed": bool(allowed[0, 0, :, 6:].any())},
    "assertions": "all passed", "tolerance": "float64 math and gradient absolute tolerance 1e-12; restored and padding invariance exact",
}
(output / "probe-results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
