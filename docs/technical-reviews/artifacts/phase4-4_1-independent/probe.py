"""Bounded CPU checks; mathematical features are unitless, axes are position/feature."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
from decimal import Decimal
import torch
from torch import nn
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.attention import attention_mask

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(42)
original = (OUT / "original/fence-1.py").read_bytes()
ns = {"__name__": "__main__"}
exec(compile(original, str(OUT / "original/fence-1.py"), "exec"), ns)
e, p, ids, positions, x = (ns[name] for name in ("e", "p", "ids", "positions", "x"))
expected = torch.tensor([[1., 0., 0.], [0.1, 1.2, 0.3], [1.2, 0.4, 0.6]])
error = (x.detach() - expected).abs().max().item()
assert error <= 1e-6
word_decimals = [["1", "0", "0"], ["0", "1", "0"], ["1", "0", "0"]]
pos_decimals = [["0", "0", "0"], ["0.1", "0.2", "0.3"], ["0.2", "0.4", "0.6"]]
decimal_sum = [[Decimal(a) + Decimal(b) for a, b in zip(erow, prow)] for erow, prow in zip(word_decimals, pos_decimals)]
decimal_error = max(abs(Decimal(float(x[i,j].detach())) - decimal_sum[i][j]) for i in range(3) for j in range(3))
assert decimal_error <= Decimal("0.000001")
assert positions.tolist() == [0, 1, 2] and positions.dtype == torch.int64
assert list(x.shape) == [3, 3] and torch.equal(e(ids)[0], e(ids)[2])
assert not torch.equal(x[0], x[2])
assert x.requires_grad and type(x.grad_fn).__name__ == "AddBackward0"
assert e.weight.grad is None and p.weight.grad is None
original_values = x.detach().tolist()

# Execute the specified exercise as an actual modified copy of the original fence.
zero_variant = original.replace(b"ids = torch.tensor", b"    p.weight.zero_()\nids = torch.tensor").replace(
    b"assert not torch.equal(x[0], x[2])", b"assert torch.equal(x[0], x[2])")
assert zero_variant != original
(OUT / "exercise-zero-position.py").write_bytes(zero_variant)
zero_ns = {"__name__": "__main__"}
exec(compile(zero_variant, str(OUT / "exercise-zero-position.py"), "exec"), zero_ns)
assert torch.equal(zero_ns["x"], zero_ns["e"](zero_ns["ids"]))
zero_values = zero_ns["x"].detach().tolist()

# Reorder position indices alone; token lookup is unchanged.
swapped_positions = torch.tensor([2, 1, 0])
swapped = e(ids) + p(swapped_positions)
assert torch.equal(swapped[0], x[2]) and torch.equal(swapped[2], x[0])

# Finite lookup includes row 4 but excludes both 5 and negative values.
assert torch.equal(p(torch.tensor([4]))[0], torch.tensor([0.4, 0.8, 1.2]))
bound_errors = {}
for index in (5, -1):
    try:
        p(torch.tensor([index]))
    except IndexError as exc:
        bound_errors[str(index)] = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError("Embedding accepted invalid index")

try:
    e(ids) + nn.Embedding(5, 2)(positions)
except RuntimeError as exc:
    width_error = {"type": type(exc).__name__, "message": str(exc)}
else:
    raise AssertionError("incompatible 3-feature and 2-feature tables accepted")

# A local backward diagnostic proves that hand assignment did not freeze the tables.
# This is not a language-model loss and makes no training-quality claim.
x.sum().backward()
assert torch.equal(e.weight.grad, torch.tensor([[0.,0.,0.],[2.,2.,2.],[1.,1.,1.],[0.,0.,0.]]))
assert torch.equal(p.weight.grad, torch.tensor([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.],[0.,0.,0.],[0.,0.,0.]]))
with torch.no_grad():
    x_no_grad = e(ids) + p(positions)
assert not x_no_grad.requires_grad and x_no_grad.grad_fn is None
assert torch.is_grad_enabled()

# No equally-spaced requirement: replace the position table with arbitrary vectors.
arbitrary = torch.tensor([[.7,-.2,.8], [0.,.6,-.4], [-.1,.9,.5], [2.,-3.,4.], [-2.,1.,.1]])
with torch.no_grad():
    p.weight.copy_(arbitrary)
arbitrary_x = e(ids) + p(positions)
assert torch.allclose(arbitrary_x - e(ids), arbitrary[positions], atol=1e-6, rtol=0)

# Check the current TinyLM entry to the first block against the lesson's table.
model = TinyLM(ModelConfig(vocab_size=4, width=3, max_length=5, layers=1, heads=1))
with torch.no_grad():
    model.embedding.weight.copy_(ns["e"].weight)
    model.position.weight.copy_(torch.tensor([[0.,0.,0.],[.1,.2,.3],[.2,.4,.6],[.3,.6,.9],[.4,.8,1.2]]))
captured = []
hook = model.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
result = model(ids[None])
hook.remove()
assert len(captured) == 1 and captured[0].shape == (1,3,3)
assert torch.allclose(captured[0][0], expected, atol=1e-6, rtol=0)
assert result["logits"].shape == (1,3,4)
length_error = None
try:
    model(torch.ones((1,6), dtype=torch.int64))
except ValueError as exc:
    length_error = {"type": type(exc).__name__, "message": str(exc)}
assert length_error is not None
allowed = attention_mask(torch.arange(3), torch.arange(3))[0,0]
assert allowed.tolist() == [[True,False,False],[True,True,False],[True,True,True]]

# Compare saved upstream docs/source bytes with the files shipped in this CPU wheel.
torch_root = Path(torch.__file__).resolve().parent
installed_comparison = {}
for saved, installed in (("pytorch-sparse.py", "nn/modules/sparse.py"), ("pytorch-torch-docs.py", "_torch_docs.py"), ("pytorch-tensor-docs.py", "_tensor_docs.py"), ("pytorch-grad-mode.py", "autograd/grad_mode.py")):
    a = (OUT / "sources" / saved).read_bytes()
    b = (torch_root / installed).read_bytes()
    installed_comparison[installed] = {"installed_path": str(torch_root / installed), "sha256": hashlib.sha256(b).hexdigest(), "same_as_official_commit": a == b}
    assert a == b
facts = {
    "python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
    "device": str(x.device), "cuda_build": str(torch.version.cuda), "cuda_available": torch.cuda.is_available(),
    "axes": ["position", "feature"], "units": "unitless learned feature values", "denominator": "none: elementwise sums, no reduction-based metric", "tolerance": "absolute <= 1e-6, rtol=0; indices, shape and equal assertions exact",
    "original_shape": list(x.shape), "original_values": original_values, "max_abs_error": error,
    "independent_decimal_sums": [[str(n) for n in row] for row in decimal_sum], "max_abs_error_from_exact_decimal": str(decimal_error),
    "same_token_features_equal": True, "same_token_position_sum_differs": True,
    "zero_exercise_values": zero_values, "swapped_position_values": swapped.detach().tolist(),
    "bounds_errors": bound_errors, "incompatible_width_error": width_error,
    "original_grad_fn": type(x.grad_fn).__name__, "after_assignment_requires_grad": [e.weight.requires_grad,p.weight.requires_grad],
    "backward_e_grad": e.weight.grad.tolist(), "backward_p_grad": p.weight.grad.tolist(),
    "unused_p_rows": [3,4], "no_grad_variant_requires_grad": x_no_grad.requires_grad,
    "arbitrary_position_values": arbitrary.tolist(), "arbitrary_output": arbitrary_x.detach().tolist(),
    "tiny_lm_first_block_shape": list(captured[0].shape), "tiny_lm_first_block_values": captured[0].tolist(), "length_error": length_error,
    "causal_allowed": allowed.tolist(), "installed_official_bytes": installed_comparison,
    "scope": "Original forward, specified zero-table exercise, bounded counterfactuals and one local backward diagnostic. No optimizer step, corpus, weights download, GPU or training run.",
}
(OUT / "probe-result.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(facts, ensure_ascii=False, indent=2))
