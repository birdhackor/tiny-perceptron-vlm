"""Independent scalar/tensor checks; no LM training, checkpoints or downloads."""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import loss_sum, masked_loss
import torch.optim.sgd as sgd_module
import torch.optim.adam as adam_module
import torch.optim.optimizer as optimizer_module

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version, "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
    "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
    "threads": str(torch.get_num_threads()), "cwd": str(Path.cwd()),
}
installed = {}
for module in [sgd_module, adam_module, optimizer_module]:
    path = Path(inspect.getsourcefile(module))
    raw = path.read_bytes()
    destination = BASE / "sources" / ("installed-" + path.name)
    destination.write_bytes(raw)
    installed[module.__name__] = {"input_path": str(path), "snapshot": str(destination.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()}
environment["installed_source_inputs"] = installed
(BASE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

raw = (BASE / "inputs" / "fence-1.py").read_bytes()
variation = raw.replace(b"torch.tensor(1.0)", b"torch.tensor(4.0)")
assert variation != raw and raw.count(b"torch.tensor(1.0)") == 1
(BASE / "code" / "exercise-original-single-change.py").write_bytes(variation)
buffer = io.StringIO()
namespace = {}
with contextlib.redirect_stdout(buffer):
    exec(compile(variation, "5.14:original-fence:start4", "exec"), namespace)
assert namespace["w"].item() == 4.0
(BASE / "exercise-stdout.txt").write_text(buffer.getvalue())

results = {"scope": "bounded deterministic scalar/tensor probes; no language-model performance measured", "exercise_stdout": buffer.getvalue()}
numeric = []
for start in [1.0, 4.0]:
    for lr in [0.01, 0.1, 2.0, 0.5]:
        initial = torch.tensor(start, dtype=torch.float32)
        update = initial - lr * 2 * (initial - 3)
        actual = {"start": start, "lr": lr, "initial_loss": (initial - 3).square().item(), "new": update.item(), "loss": (update - 3).square().item()}
        expected_new = start - lr * 2 * (start - 3)
        expected_loss = (expected_new - 3) ** 2
        assert math.isclose(actual["new"], expected_new, abs_tol=1e-6)
        assert math.isclose(actual["loss"], expected_loss, abs_tol=2e-6)
        actual["loss_ratio"] = actual["loss"] / actual["initial_loss"]
        assert math.isclose(actual["loss_ratio"], (1 - 2 * lr) ** 2, abs_tol=2e-6)
        numeric.append(actual)
results["numeric"] = numeric

parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
opt = torch.optim.SGD([parameter], lr=0.1)
(parameter - 3).square().backward()
assert parameter.grad.item() == -4 and parameter.item() == 1
before_step = parameter.item()
opt.step()
assert math.isclose(parameter.item(), 1.4, abs_tol=1e-14)
results["backward_vs_step"] = {"before_step": before_step, "after_step": parameter.item(), "gradient": parameter.grad.item()}

trajectories = []
for lr in [0.01, 0.5, 0.75, 1.0, 2.0]:
    w = 1.0
    positions, losses = [w], [(w - 3) ** 2]
    for _ in range(8):
        w -= lr * 2 * (w - 3)
        positions.append(w)
        losses.append((w - 3) ** 2)
    for t, loss in enumerate(losses):
        assert math.isclose(loss, 4 * ((1 - 2 * lr) ** 2) ** t, rel_tol=1e-12, abs_tol=1e-12)
    trajectories.append({"lr": lr, "positions": positions, "losses": losses})
assert trajectories[2]["positions"][1] > 3 and trajectories[2]["losses"][1] < trajectories[2]["losses"][0]
assert trajectories[3]["losses"] == [4.0] * 9
results["eight_step_scalar_trajectories"] = trajectories

warm_parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
warm_optimizer = torch.optim.SGD([warm_parameter], lr=0.1, momentum=0.9)
warm_parameter.grad = torch.ones_like(warm_parameter)
warm_optimizer.step()
warm_state = copy.deepcopy(warm_optimizer.state_dict())
history_results = []
for restore_history in [False, True]:
    p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([p], lr=0.1, momentum=0.9)
    if restore_history:
        optimizer.load_state_dict(copy.deepcopy(warm_state))
    (p - 3).square().backward()
    optimizer.step()
    history_results.append({"restored_momentum_history": restore_history, "new": p.item(), "momentum": optimizer.state[p]["momentum_buffer"].item()})
assert math.isclose(history_results[0]["new"], 1.4, abs_tol=1e-14)
assert math.isclose(history_results[1]["new"], 1.31, abs_tol=1e-14)
results["history_control"] = history_results

def pilot(lr):
    p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = torch.optim.AdamW([p], lr=lr, weight_decay=0.0, foreach=False)
    optimizer.load_state_dict(copy.deepcopy(adam_state))
    for group in optimizer.param_groups:
        group["lr"] = lr
    sequence, start_state = [2.0, 3.0, 2.5, 3.5], copy.deepcopy(optimizer.state[p])
    pre_losses = []
    for target in sequence:
        optimizer.zero_grad()
        loss = (p - target).square()
        pre_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return {"lr": lr, "initial_parameter": 1.0, "initial_history": {k: v.item() for k, v in start_state.items()}, "data_order": sequence, "updates": 4, "preupdate_losses": pre_losses, "final_parameter": p.item(), "fixed_probe_loss": (p - 3).square().item()}

warm_adam_parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
warm_adam = torch.optim.AdamW([warm_adam_parameter], lr=0.001, weight_decay=0.0, foreach=False)
warm_adam_parameter.grad = torch.ones_like(warm_adam_parameter)
warm_adam.step()
adam_state = copy.deepcopy(warm_adam.state_dict())
pilots = [pilot(lr) for lr in [1e-4, 1e-3, 1e-2]]
assert len({json.dumps(x["initial_history"], sort_keys=True) for x in pilots}) == 1
assert len({x["final_parameter"] for x in pilots}) == 3
assert pilots[1] == pilot(1e-3)
results["four_update_fixed_scalar_pilots"] = pilots

tiny = torch.tensor(1.0, dtype=torch.float64)
tiny_after = tiny - 1e-7 * 2 * (tiny - 3)
rounded_identical = f"{tiny.item():.4f}" == f"{tiny_after.item():.4f}"
assert tiny_after.item() != tiny.item() and rounded_identical
results["display_precision"] = {"before": tiny.item(), "after": tiny_after.item(), "same_four_decimal_display": rounded_identical}

logits = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
labels = torch.tensor([[1, -100]])
total, count = loss_sum(logits, labels)
loss = masked_loss(logits, labels)
loss.backward()
assert count.item() == 1 and math.isclose(loss.item(), math.log(3), abs_tol=1e-14)
assert torch.equal(logits.grad[0, 1], torch.zeros(3, dtype=torch.float64))
try:
    masked_loss(logits, torch.full((1, 2), -100))
except ValueError as exc:
    empty_error = str(exc)
else:
    raise AssertionError("empty targets must be rejected")
results["target_denominator"] = {"effective_count": count.item(), "loss": loss.item(), "ignored_gradient": logits.grad[0, 1].tolist(), "all_ignored_error": empty_error}

stages = [torch.tensor(1.0), torch.tensor(0.0)]
inf = stages[0] / stages[1]
nan = stages[1] / stages[1]
assert all(torch.isfinite(x) for x in stages)
assert torch.isinf(inf) and torch.isnan(nan)
results["nonfinite_denominator_probe"] = {"inputs_finite": True, "first_nonfinite_operation": "division by zero", "one_div_zero": "inf", "zero_div_zero": "nan"}

torch.manual_seed(42)
a = torch.randn(8)
torch.manual_seed(42)
b = torch.randn(8)
c = torch.randn(8)
assert torch.equal(a, b) and not torch.equal(b, c)
results["rng_reset"] = {"reseeded_sequence_identical": torch.equal(a, b), "unreset_sequence_identical": torch.equal(b, c)}
results["assertions"] = "all passed"
(BASE / "bounded-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
