"""Bounded CPU checks of the current lesson's original code and API contracts."""
import contextlib
import io
import json
import platform
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
ROOT = Path(__file__).resolve().parent
original = (ROOT / "original/fence-1.py").read_text()

def run_text(code):
    ns = {"__name__": "__main__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "current-1.13-fence-1", "exec"), ns)
    return {"stdout": buffer.getvalue(), "first": ns["first"].item(),
            "second": ns["second"].item(), "final_grad": ns["w"].grad.item(),
            "w": ns["w"].item(), "shape": list(ns["w"].shape),
            "dtype": str(ns["w"].dtype), "device": str(ns["w"].device)}

results = {"environment": {"python": platform.python_version(),
            "torch": torch.__version__, "torch_git_version": torch.version.git_version,
            "device": "cpu", "dtype": "torch.float32", "threads": torch.get_num_threads()},
           "original": run_text(original),
           "exercise_remove_clear": run_text(original.replace("w.grad = None\n", "# w.grad = None\n"))}
assert results["original"]["stdout"] == "4.0 8.0 4.0 2.0\n"
assert results["exercise_remove_clear"]["stdout"] == "4.0 8.0 12.0 2.0\n"

w = nn.Parameter(torch.tensor(2.0))
assert w.is_leaf and w.requires_grad
loss = w.square()
loss.backward()
try:
    loss.backward()
except RuntimeError as exc:
    results["same_loss_default"] = {"error_type": type(exc).__name__, "message": str(exc)}
    assert "backward through the graph a second time" in str(exc)
else:
    raise AssertionError("Reusing square's freed saved tensor unexpectedly succeeded")

w = nn.Parameter(torch.tensor(2.0))
loss = w.square()
loss.backward(retain_graph=True)
first = w.grad.clone()
loss.backward()
assert first.item() == 4 and w.grad.item() == 8
results["same_loss_retain_graph"] = {"first": first.item(), "second": w.grad.item(), "w": w.item()}

w = nn.Parameter(torch.tensor(2.0))
w.square().backward()
clone = w.grad.clone()
alias = w.grad
w.square().backward()
assert clone.item() == 4 and alias.item() == 8
results["clone_vs_alias"] = {"clone": clone.item(), "alias": alias.item(),
                             "storage_is_distinct": clone.data_ptr() != w.grad.data_ptr()}

w = nn.Parameter(torch.tensor(2.0))
optimizer = torch.optim.SGD([w], lr=0.1)
w.square().backward()
optimizer.zero_grad(set_to_none=True)
assert w.grad is None and w.item() == 2
w.square().backward()
grad_after_none = w.grad.item()
optimizer.zero_grad(set_to_none=False)
zero_grad = w.grad.item()
assert zero_grad == 0 and w.item() == 2
w.square().backward()
optimizer.step()
assert abs(w.item() - 1.6) < 1e-6 and w.grad.item() == 4
results["optimizer"] = {"zero_grad_none_w": 2.0, "after_none_backward": grad_after_none,
                        "after_zero": zero_grad, "step_w": w.item(),
                        "step_grad_remains": w.grad.item(), "lr": 0.1, "tolerance": 1e-6}

w = nn.Parameter(torch.tensor(-1.5))
w.square().backward()
negative_first = w.grad.item()
w.square().backward()
assert negative_first == -3 and w.grad.item() == -6 and w.item() == -1.5
results["negative_parameter"] = {"w": w.item(), "first": negative_first, "second": w.grad.item()}

w = nn.Parameter(torch.tensor(2.0))
w.square().backward()
(3 * w.square()).backward()
assert w.grad.item() == 16
results["different_losses"] = {"w": w.item(), "new_gradients": [4.0, 12.0], "sum": w.grad.item()}

w = nn.Parameter(torch.tensor(2.0))
targets = torch.tensor([0.0, 1.0])
for target in targets:
    ((w - target).square() / len(targets)).backward()
accumulated = w.grad.item()
w.grad = None
((w - targets).square().mean()).backward()
assert accumulated == w.grad.item() == 3
results["two_microbatches"] = {"samples": len(targets), "microbatch_samples": 1,
                              "reduction": "sum two scalar losses each divided by 2 equals full-batch mean",
                              "accumulated": accumulated, "full_batch_mean": w.grad.item(), "w": w.item()}
print(json.dumps(results, indent=2))
