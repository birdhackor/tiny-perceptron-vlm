"""Small CPU probes of lookup, manual changes and autograd; no dataset or training loop."""
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
import inspect
import io
import json
import platform
import sys

import torch
from torch import nn

ART = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(42)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "platform": platform.platform(),
    "threads": str(torch.get_num_threads()),
}

def run_source(raw, name):
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(raw, name, "exec"), namespace)
    return namespace, stdout.getvalue()

raw = (ART / "original-run/fence-1.py").read_bytes()
original, original_stdout = run_source(raw, "course/chapters/01.md:211-original-fence")
scores = original["scores"]
inputs = original["inputs"]
output = original["output"]
expected_weights = torch.tensor([[0., 2., 0.], [0., 0., 0.], [0., 0., 0.]])
expected_output = torch.tensor([[0., 2., 0.], [0., 0., 0.]])
assert torch.equal(scores.weight.detach(), expected_weights)
assert torch.equal(output.detach(), expected_output)
assert list(output.shape) == [2, 3] and list(scores.weight.shape) == [3, 3]
assert scores.weight.numel() == 9
assert inputs.dtype == torch.int64
assert isinstance(scores.weight, nn.Parameter)
assert dict(scores.named_parameters())["weight"] is scores.weight
assert scores.weight.requires_grad and scores.weight.grad_fn is None and scores.weight.is_leaf
assert output.requires_grad and output.grad_fn is not None and scores.weight.grad is None
assert "Parameter containing" in original_stdout and "requires_grad=True" in original_stdout
weight_before_forward = scores.weight.detach().clone()
version_before_forward = scores.weight._version
repeated_output = scores(inputs)
assert torch.equal(scores.weight.detach(), weight_before_forward)
assert scores.weight._version == version_before_forward

exercise_raw = raw.replace(b"scores.weight[0, 1] = 2.0", b"scores.weight[0, 1] = 2.0\n    scores.weight[2, 0] = 3.0")
(ART / "exercise-source.py").write_bytes(exercise_raw)
exercise, exercise_stdout = run_source(exercise_raw, "exercise-source.py")
exercise_output = exercise["output"].detach()
assert torch.equal(exercise_output, torch.tensor([[0., 2., 0.], [3., 0., 0.]]))
assert torch.equal(exercise_output[0], output.detach()[0])
assert torch.equal(exercise["scores"].weight.detach()[1], scores.weight.detach()[1])

plain = torch.ones(3)
plain_pointer = plain.data_ptr()
zero_return = plain.zero_()
assert zero_return is plain and plain.data_ptr() == plain_pointer
assert torch.equal(plain, torch.zeros(3))

with torch.no_grad():
    no_grad_computation = scores.weight.square()
assert not no_grad_computation.requires_grad and no_grad_computation.grad_fn is None
assert scores.weight.requires_grad and scores.weight.is_leaf
outside_grad_computation = scores.weight.square()
assert outside_grad_computation.requires_grad and outside_grad_computation.grad_fn is not None
try:
    nn.Embedding(3, 3).weight.zero_()
except RuntimeError as exc:
    missing_no_grad_error = str(exc)
else:
    raise AssertionError("Expected leaf in-place modification error without no_grad")

# A single sum.backward probe establishes that backward fills gradients,
# while this default Embedding lookup and backward do not change the weights.
weight_before_backward = scores.weight.detach().clone()
output.sum().backward()
expected_grad = torch.tensor([[1., 1., 1.], [0., 0., 0.], [1., 1., 1.]])
assert torch.equal(scores.weight.grad, expected_grad)
assert torch.equal(scores.weight.detach(), weight_before_backward)

# Demonstrate that a score table has no positivity/normalization constraint.
signed = nn.Embedding(3, 3)
with torch.no_grad():
    signed.weight.copy_(torch.tensor([[-1., 2., 4.], [0., 0., 0.], [3., -2., 0.]]))
signed_output = signed(torch.tensor([0, 2]))
assert torch.equal(signed_output.detach(), torch.tensor([[-1., 2., 4.], [3., -2., 0.]]))
assert bool((signed_output < 0).any()) and signed_output.sum(dim=-1).tolist() == [5., 1.]

# Extend the toy vocabulary for the prose's '=' thought experiment.
# The original three-character vocabulary deliberately does not include '='.
char_to_id = {"貓": 0, "看": 1, "狗": 2, "=": 3}
larger = nn.Embedding(len(char_to_id), len(char_to_id))
with torch.no_grad():
    larger.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
contexts = ["顏色=", "形狀="]
last_ids = torch.tensor([char_to_id[text[-1]] for text in contexts])
context_outputs = larger(last_ids)
assert last_ids.tolist() == [3, 3]
assert torch.equal(context_outputs[0], context_outputs[1])
assert larger.weight.numel() == len(char_to_id) ** 2

# Confirm that consulted official Python implementation snapshots match runtime files.
import torch.nn.modules.sparse as sparse_module
import torch.nn.parameter as parameter_module
import torch.nn.functional as functional_module
import torch.autograd.grad_mode as grad_mode_module
runtime_source_matches = {}
for module, filename in [(sparse_module, "torch-sparse-installed-commit.py"),
                         (parameter_module, "torch-parameter-installed-commit.py"),
                         (functional_module, "torch-functional-installed-commit.py"),
                         (grad_mode_module, "torch-grad-mode-installed-commit.py")]:
    installed_raw = Path(inspect.getsourcefile(module)).read_bytes()
    official_raw = (ART / filename).read_bytes()
    runtime_source_matches[module.__name__] = {
        "installed_path": inspect.getsourcefile(module),
        "installed_sha256": sha256(installed_raw).hexdigest(),
        "official_sha256": sha256(official_raw).hexdigest(),
        "bytes_identical": installed_raw == official_raw,
    }

results = {
    "environment": environment,
    "original": {"fence_sha256": sha256(raw).hexdigest(), "stdout": original_stdout,
                 "weight": scores.weight.detach().tolist(), "output": output.detach().tolist(),
                 "weight_shape": list(scores.weight.shape), "output_shape": list(output.shape),
                 "weight_numel": scores.weight.numel(), "input_dtype": str(inputs.dtype),
                 "is_registered_parameter": True, "parameter_requires_grad": scores.weight.requires_grad,
                 "output_requires_grad": output.requires_grad, "output_grad_fn": type(output.grad_fn).__name__,
                 "before_backward_grad_was_none": True,
                 "lookup_preserved_weight_and_version": True},
    "exercise": {"source_sha256": sha256(exercise_raw).hexdigest(), "stdout": exercise_stdout,
                 "output": exercise_output.tolist(), "cat_row_unchanged": True, "unselected_row_unchanged": True},
    "zero_inplace": {"returned_same_object": True, "storage_pointer_unchanged": True, "values": plain.tolist()},
    "no_grad": {"inside_result_requires_grad": no_grad_computation.requires_grad,
                "inside_grad_fn": str(no_grad_computation.grad_fn),
                "outside_result_requires_grad": outside_grad_computation.requires_grad,
                "parameter_stays_leaf_and_requires_grad": True,
                "without_no_grad_error": missing_no_grad_error},
    "backward_only": {"gradient": scores.weight.grad.tolist(), "weights_unchanged": True,
                      "purpose": "one sum backward probe; no optimizer/update and no training loop"},
    "signed_logits": {"output": signed_output.detach().tolist(), "row_sums": signed_output.sum(-1).tolist()},
    "single_character_context": {"contexts": contexts, "last_ids": last_ids.tolist(),
                                 "output": context_outputs.detach().tolist(), "parameter_count": larger.weight.numel(),
                                 "scope": "expanded vocabulary includes '='; feeds only the last character ID"},
    "runtime_source_matches": runtime_source_matches,
    "all_assertions_passed": True,
}
(ART / "bounded-checks-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
(ART / "bounded-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
