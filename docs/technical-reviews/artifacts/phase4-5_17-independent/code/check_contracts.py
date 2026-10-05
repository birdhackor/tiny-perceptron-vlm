"""Bounded CPU contracts for section 5.17, not model training."""
from contextlib import nullcontext
import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "threads": str(torch.get_num_threads()),
    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
def fixed_layer():
    layer = nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2., -3.]]))
        layer.bias.copy_(torch.tensor([0.5]))
    return layer

def data(value):
    return None if value is None else value.detach().tolist()

contexts = {"default": nullcontext, "no_grad": torch.no_grad, "inference": torch.inference_mode}
matrix = []
for training, mode, parameter_grad, input_grad in itertools.product(
        [True, False], contexts, [True, False], [False, True]):
    layer = fixed_layer()
    layer.train(training)
    for p in layer.parameters():
        p.requires_grad_(parameter_grad)
    x = torch.ones(1, 2, requires_grad=input_grad)
    old_parameters = [p.detach().clone() for p in layer.parameters()]
    with contexts[mode]():
        y = layer(x)
        grad_enabled_inside = torch.is_grad_enabled()
    expected_grad = mode == "default" and (parameter_grad or input_grad)
    assert y.requires_grad == expected_grad
    assert (y.grad_fn is not None) == expected_grad
    assert y.shape == (1, 1) and y.item() == -0.5
    assert layer.training == training and torch.is_grad_enabled()
    assert x.requires_grad == input_grad
    assert all(p.requires_grad == parameter_grad for p in layer.parameters())
    error = None
    try:
        y.sum().backward()
    except RuntimeError as exception:
        error = str(exception)
        assert not expected_grad and "does not require grad" in error
    else:
        assert expected_grad
    if expected_grad and parameter_grad:
        assert torch.equal(layer.weight.grad, torch.ones(1, 2))
        assert torch.equal(layer.bias.grad, torch.ones(1))
    else:
        assert layer.weight.grad is None and layer.bias.grad is None
    if expected_grad and input_grad:
        assert torch.equal(x.grad, layer.weight)
    else:
        assert x.grad is None
    assert all(torch.equal(p, old) for p, old in zip(layer.parameters(), old_parameters))
    matrix.append({
        "training": training, "mode": mode, "parameter_requires_grad": parameter_grad,
        "input_requires_grad": input_grad, "grad_enabled_inside": grad_enabled_inside,
        "output": data(y), "output_shape": list(y.shape), "output_requires_grad": y.requires_grad,
        "grad_fn": None if y.grad_fn is None else type(y.grad_fn).__name__,
        "is_inference_tensor": y.is_inference(), "weight_grad": data(layer.weight.grad),
        "bias_grad": data(layer.bias.grad), "input_grad": data(x.grad),
        "backward_error": error, "parameters_unchanged": True,
    })

# An eval Linear has the exact same numerical forward as a train Linear.
layer = fixed_layer()
x = torch.ones(1, 2)
train_output = layer.train()(x)
eval_output = layer.eval()(x)
assert torch.equal(train_output, eval_output)

# Dropout mode affects masking, independently of the grad context.
dropout_checks = []
for training, mode in itertools.product([True, False], contexts):
    dropout = nn.Dropout(p=0.5).train(training)
    x = torch.ones(16, requires_grad=True)
    torch.manual_seed(123)
    with contexts[mode]():
        y = dropout(x)
    assert dropout.training == training
    if training:
        assert set(y.detach().tolist()) == {0.0, 2.0}
        assert y.requires_grad == (mode == "default")
    else:
        assert torch.equal(y, x) and y is x
        # Identity reuses an existing leaf: it creates no new graph edge.
        assert y.requires_grad and y.grad_fn is None
    dropout_checks.append({"training": training, "mode": mode,
        "values": data(y), "output_requires_grad": y.requires_grad,
        "is_original_input": y is x, "is_inference_tensor": y.is_inference()})

# An upstream trainable generator can differentiate through a frozen reference.
generator_checks = []
for mode in contexts:
    generator = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        generator.weight.copy_(torch.eye(2))
    reference = fixed_layer().requires_grad_(False).eval()
    h = generator(torch.ones(1, 2))
    with contexts[mode]():
        y = reference(h)
    error = None
    try:
        y.sum().backward()
    except RuntimeError as exception:
        error = str(exception)
        assert mode != "default"
    if mode == "default":
        assert torch.equal(generator.weight.grad, torch.tensor([[2., 2.], [-3., -3.]]))
    else:
        assert generator.weight.grad is None and not y.requires_grad
    assert all(p.grad is None for p in reference.parameters())
    generator_checks.append({"mode": mode, "output_requires_grad": y.requires_grad,
        "generator_weight_grad": data(generator.weight.grad),
        "reference_grads": [data(p.grad) for p in reference.parameters()], "backward_error": error})

# no_grad scope restores the previous flag and has the documented factory exception.
with torch.no_grad():
    factory_leaf = torch.ones(1, requires_grad=True)
    factory_parameter = nn.Parameter(torch.ones(1))
    normal_result = factory_leaf * 2
    with torch.enable_grad():
        enabled_result = factory_leaf * 3
    assert not torch.is_grad_enabled()
assert torch.is_grad_enabled()
assert factory_leaf.requires_grad and factory_parameter.requires_grad
assert not normal_result.requires_grad and normal_result.grad_fn is None
assert enabled_result.requires_grad and enabled_result.grad_fn is not None

# Pre-existing .grad is not cleared by requires_grad_(False), eval or no_grad.
layer = fixed_layer()
layer(torch.ones(1, 2)).sum().backward()
old_grad = layer.weight.grad.clone()
layer.requires_grad_(False).eval()
with torch.no_grad():
    layer(torch.ones(1, 2))
assert torch.equal(layer.weight.grad, old_grad)

# Inference-created tensors have additional restrictions unlike no_grad tensors.
with torch.inference_mode():
    inference_tensor = torch.ones(1)
inference_errors = {}
for name, operation in {
    "requires_grad_outside": lambda: inference_tensor.requires_grad_(True),
    "version_counter": lambda: inference_tensor._version,
    "saved_for_backward": lambda: (inference_tensor * torch.ones(1, requires_grad=True)).sum().backward(),
}.items():
    try:
        operation()
    except RuntimeError as exception:
        inference_errors[name] = str(exception)
    else:
        raise AssertionError(name)
with torch.no_grad():
    ordinary_tensor = torch.ones(1)
trainable = torch.ones(1, requires_grad=True)
(ordinary_tensor * trainable).sum().backward()
assert torch.equal(trainable.grad, torch.ones(1))

# Check the AD distinction explicitly, using a single scalar dual tensor.
forward_ad = {}
for mode in ["no_grad", "inference"]:
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(2.), torch.tensor(1.))
        with contexts[mode]():
            result = dual * 3
        primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
        forward_ad[mode] = {"primal": primal.item(), "tangent": None if tangent is None else tangent.item()}
assert forward_ad == {"no_grad": {"primal": 6., "tangent": 3.}, "inference": {"primal": 6., "tangent": None}}

result = {"environment": environment, "matrix": matrix, "matrix_case_count": len(matrix),
    "linear_train_eval_equal": True, "dropout": dropout_checks,
    "generator": generator_checks, "factory_exception": True,
    "enable_grad_nested": True, "stale_gradient_preserved": True,
    "inference_errors": inference_errors, "no_grad_tensor_reusable": True,
    "forward_ad": forward_ad, "all_assertions_passed": True,
    "scope": "24 Linear forward/backward cases, 6 dropout cases and 3 generator cases; no optimizer or training, no data/model downloads."}
(BASE / "contracts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
