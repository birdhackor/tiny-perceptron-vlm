"""Short CPU checks of 5.4 original code, its exercise, and optimizer equivalence."""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import platform
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import torch
import torch.optim.adam as adam_module
import torch.optim.optimizer as optimizer_module

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
getcontext().prec = 45
D = Decimal
env = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
       "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda),
       "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()),
       "platform": platform.platform(), "cwd": str(Path.cwd()), "default_dtype": str(torch.get_default_dtype()),
       "offline_flags": {key: os.environ.get(key, "<unset>") for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}}

def tensor(x):
    return torch.tensor(x, dtype=torch.float64)

def close(actual, expected, atol=1e-14, rtol=1e-12):
    torch.testing.assert_close(actual, tensor(expected), atol=atol, rtol=rtol)

def serial(ns):
    return {key: {"values": ns[key].tolist(), "dtype": str(ns[key].dtype), "shape": list(ns[key].shape),
                  "requires_grad": ns[key].requires_grad, "grad_is_none": ns[key].grad is None}
            for key in ("g", "m", "v", "m_hat", "v_hat", "update")}

raw = (OUT / "original/fence-1.py").read_bytes()
original_namespace = {"__name__": "__main__"}
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exec(compile(raw, "original/fence-1.py", "exec"), original_namespace)
assert original_namespace["g"].device.type == "cpu"
assert not any(original_namespace[k].requires_grad for k in ("g", "m", "v", "m_hat", "v_hat", "update"))
results = {"original_fence_sha256": hashlib.sha256(raw).hexdigest(), "original_stdout": buffer.getvalue(),
           "original": serial(original_namespace), "variants": {}}

for name, replacement in (("negative_second_gradient", "[1.0, -100.0]"), ("zero_gradients", "[0.0, 0.0]")):
    code = raw.replace(b"[1.0, 100.0]", replacement.encode())
    (OUT / f"{name}.py").write_bytes(code)
    ns = {"__name__": "__main__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, f"{name}.py", "exec"), ns)
    results["variants"][name] = {"stdout": buffer.getvalue(), "values": serial(ns)}
    if name == "negative_second_gradient":
        torch.testing.assert_close(ns["v"], original_namespace["v"], atol=0, rtol=0)
        torch.testing.assert_close(ns["m"], original_namespace["m"] * torch.tensor([1., -1.]), atol=0, rtol=0)
        torch.testing.assert_close(ns["update"], original_namespace["update"] * torch.tensor([1., -1.]), atol=0, rtol=0)
        p = torch.zeros(2)
        p -= ns["update"]
        assert p[1] > 0
        results["variants"][name]["parameter_after_manual_subtraction"] = p.tolist()
    else:
        assert torch.isfinite(ns["update"]).all() and torch.equal(ns["update"], torch.zeros(2))
        without_epsilon = .001 * ns["m_hat"] / ns["v_hat"].sqrt()
        assert torch.isnan(without_epsilon).all()
        results["variants"][name]["without_epsilon_is_nan"] = True

# Exact independent decimal arithmetic, including the placement of epsilon.
gs = [D(1), D(100)]
m = [D('.1') * g for g in gs]
v = [D('.001') * g * g for g in gs]
mh = [x / D('.1') for x in m]
vh = [x / D('.001') for x in v]
updates = [D('.001') * a / (b.sqrt() + D('1e-8')) for a, b in zip(mh, vh)]
sgd = [D('.001') * g for g in gs]
results["decimal_derivation"] = {"sgd": list(map(str, sgd)), "m": list(map(str, m)), "v": list(map(str, v)),
                                 "m_hat": list(map(str, mh)), "v_hat": list(map(str, vh)), "update": list(map(str, updates)),
                                 "epsilon": str(D('1e-8')), "epsilon_reciprocal": str(1 / D('1e-8'))}
for key, expected in (("m", m), ("v", v), ("m_hat", mh), ("v_hat", vh), ("update", updates)):
    close(original_namespace[key].double(), [float(x) for x in expected], atol=1e-6, rtol=2e-7)

def manual_step(g, m, v, t):
    m = .9 * m + .1 * g
    v = .999 * v + .001 * g.square()
    mh = m / (1 - .9 ** t)
    vh = v / (1 - .999 ** t)
    update = .001 * mh / (vh.sqrt() + 1e-8)
    return m, v, mh, vh, update

api_cases = {}
for name, gvals in (("original_first_step", [1., 100.]), ("exercise_first_step", [1., -100.]),
                    ("tiny_gradients_epsilon_location", [1e-10, -2e-10]), ("zero_gradients", [0., 0.])):
    g = tensor(gvals)
    m, v, mh, vh, delta = manual_step(g, torch.zeros_like(g), torch.zeros_like(g), 1)
    p = torch.nn.Parameter(torch.zeros_like(g))
    optimizer = torch.optim.Adam([p], lr=.001, betas=(.9, .999), eps=1e-8, foreach=False, fused=False, weight_decay=0)
    p.grad = g.clone()
    before = p.detach().clone()
    optimizer.step()
    observed = before - p.detach()
    close(observed, delta.tolist())
    close(optimizer.state[p]["exp_avg"], m.tolist())
    close(optimizer.state[p]["exp_avg_sq"], v.tolist())
    assert optimizer.state[p]["step"].item() == 1
    api_cases[name] = {"gradient": g.tolist(), "manual_update": delta.tolist(), "torch_Adam_update": observed.tolist(),
                       "max_abs_difference": (observed - delta).abs().max().item(),
                       "state": {key: value.tolist() for key, value in optimizer.state[p].items()}}
    if name == "tiny_gradients_epsilon_location":
        wrong = .001 * mh / (vh + 1e-8).sqrt()
        assert not torch.allclose(delta, wrong, atol=1e-12, rtol=1e-6)
        api_cases[name]["epsilon_inside_sqrt_incorrect_variant"] = wrong.tolist()
results["optimizer_api_cases"] = api_cases

# Different coordinate histories and exact resume; no training loop/data involved.
grads = [tensor([1., 100.]), tensor([1., -1.])]
p = torch.nn.Parameter(tensor([0., 0.]))
opt = torch.optim.Adam([p], lr=.001, foreach=False, fused=False)
p.grad = grads[0].clone()
opt.step()
saved_p = p.detach().clone()
saved_state = copy.deepcopy(opt.state_dict())
p.grad = grads[1].clone()
before = p.detach().clone()
opt.step()
delta2 = before - p.detach()
m2, v2, mh2, vh2, manual2 = manual_step(grads[1], tensor([.1, 10.]), tensor([.001, 10.]), 2)
close(delta2, manual2.tolist())
assert abs(delta2[0] - delta2[1]) > 1e-4
assert grads[1][1] < 0 and mh2[1] > 0 and delta2[1] > 0
resumed = torch.nn.Parameter(saved_p.clone())
resumed_opt = torch.optim.Adam([resumed], lr=.001, foreach=False, fused=False)
resumed_opt.load_state_dict(copy.deepcopy(saved_state))
resumed.grad = grads[1].clone()
resumed_opt.step()
close(resumed.detach(), p.detach().tolist(), atol=0, rtol=0)
restarted = torch.nn.Parameter(saved_p.clone())
restarted_opt = torch.optim.Adam([restarted], lr=.001, foreach=False, fused=False)
restarted.grad = grads[1].clone()
restarted_opt.step()
assert not torch.allclose(restarted, p, atol=1e-12, rtol=1e-10)
results["history_and_resume"] = {"gradient_sequence": [g.tolist() for g in grads], "m2": m2.tolist(), "v2": v2.tolist(),
                                 "m_hat2": mh2.tolist(), "v_hat2": vh2.tolist(), "update2": delta2.tolist(),
                                 "continued_parameter": p.detach().tolist(), "resumed_parameter": resumed.detach().tolist(),
                                 "parameter_with_optimizer_state_discarded": restarted.detach().tolist(),
                                 "saved_optimizer_keys": list(saved_state["state"][0].keys()),
                                 "scope": "The example's first step agrees with gradient sign; with history, m_hat may oppose current gradient. This two-step check does not claim convergence or training quality."}

env["local_official_source_identity"] = {}
for name, module in (("adam", adam_module), ("optimizer", optimizer_module)):
    installed = Path(inspect.getfile(module))
    downloaded = OUT / f"sources/torch-{name}-installed-commit.py"
    identity = {"installed_path": str(installed), "installed_sha256": hashlib.sha256(installed.read_bytes()).hexdigest(),
                "downloaded_sha256": hashlib.sha256(downloaded.read_bytes()).hexdigest(),
                "bytes_equal": installed.read_bytes() == downloaded.read_bytes()}
    assert identity["bytes_equal"]
    env["local_official_source_identity"][name] = identity
(OUT / "bounded-environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n")
(OUT / "bounded-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": "all_assertions_passed", "original_fences": 1, "original_variants": 2,
                  "optimizer_first_step_cases": 4, "history_steps": 2, "checkpoint": "in-memory optimizer state only", "device": "cpu"}))
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
