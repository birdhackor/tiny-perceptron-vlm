"""核對梯度、有效目標與原始量測紀錄；不重新訓練。"""

import ast
import hashlib
import json
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import IGNORE, pad_batch, render_chat
from tiny_perceptron.model import loss_sum

torch.set_num_threads(1)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-6/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "platform": platform.platform(),
        "torch_git_version": str(torch.version.git_version),
        "threads": str(torch.get_num_threads()),
    },
}

x = torch.tensor([1.0, 2.0, 3.0])
toy = {}
for sizes in ((1, 2), (2, 1)):
    whole = torch.tensor(1.0, requires_grad=True)
    correct = torch.tensor(1.0, requires_grad=True)
    wrong = torch.tensor(1.0, requires_grad=True)
    whole_loss = (whole * x).square().mean()
    whole_loss.backward()
    for part in x.split(sizes):
        ((correct * part).square().sum() / len(x)).backward()
        ((wrong * part).square().mean() / len(sizes)).backward()
    observed = [p.grad.item() for p in (whole, correct, wrong)]
    assert abs(observed[0] - 28 / 3) < 1e-6
    assert abs(observed[1] - 28 / 3) < 1e-6
    assert observed[2] == (7.5 if sizes == (1, 2) else 11.5)
    toy[str(sizes)] = {"whole_loss": whole_loss.item(), "gradients": observed, "rounded": [round(v, 4) for v in observed]}
result["toy"] = toy
result["slices"] = {"first": x[:1].tolist(), "rest": x[1:].tolist(), "exercise_first": x[:2].tolist(), "exercise_rest": x[2:].tolist()}
result["exact_arithmetic"] = {
    "whole_loss": str(Fraction(14, 3)),
    "whole_gradient": str(Fraction(28, 3)),
    "wrong_gradient_one_two": str(Fraction(2, 2) + Fraction(26, 4)),
    "wrong_gradient_two_one": str(Fraction(10, 4) + Fraction(18, 2)),
    "equal_losses_unequal_counts": [float((Fraction(1, 1) + Fraction(10, 10)) / 2), float(Fraction(11, 11))],
}

logits = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 7
labels = torch.tensor([[0, IGNORE, IGNORE], [1, 2, 0]])
full = logits.clone().requires_grad_()
micro = logits.clone().requires_grad_()
bad = logits.clone().requires_grad_()
total, count = loss_sum(full, labels)
(total / count).backward()
counts = []
for i in range(2):
    summed, n = loss_sum(micro[i : i + 1], labels[i : i + 1])
    counts.append(int(n))
    (summed / count).backward()
    summed, n = loss_sum(bad[i : i + 1], labels[i : i + 1])
    (summed / n / 2).backward()
expected = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), reduction="none", ignore_index=IGNORE)
assert int(count) == 4 and counts == [1, 3]
assert torch.allclose(total, expected.sum())
assert torch.allclose(full.grad, micro.grad, atol=1e-7, rtol=0)
assert not torch.allclose(full.grad, bad.grad)
result["loss_sum"] = {"labels": labels.tolist(), "logits": logits.tolist(), "sum": total.item(), "count": int(count), "micro_counts": counts, "whole_mean": (total / count).item(), "correct_gradient_max_error": (full.grad - micro.grad).abs().max().item(), "wrong_gradient_max_error": (full.grad - bad.grad).abs().max().item()}
inputs, targets = render_chat([{"role": "user", "content": "question"}, {"role": "assistant", "content": "ok"}])
batch_x, batch_y, valid = pad_batch([(inputs, targets), (inputs[:-1], targets[:-1])])
assert int((targets != IGNORE).sum()) == 3
assert batch_x[-1, -1] == 0 and batch_y[-1, -1] == IGNORE and not valid[-1, -1]
result["assistant_and_padding"] = {"assistant_targets_including_eos": int((targets != IGNORE).sum()), "padded_input": batch_x[-1, -1].item(), "padded_label": batch_y[-1, -1].item(), "padded_valid": valid[-1, -1].item()}

p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
p.grad = torch.tensor([3.0, 4.0])
norm = torch.nn.utils.clip_grad_norm_([p], 1.0)
assert norm.item() == 5 and torch.allclose(p.grad, torch.tensor([0.6, 0.8]), atol=1e-6)
result["clipping"] = {"input_gradient": [3, 4], "original_norm": norm.item(), "clipped_gradient": p.grad.tolist()}
steps = {}
for name in ("whole", "accumulated", "per_micro_step"):
    w = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([w], lr=0.1)
    opt.zero_grad(set_to_none=True)
    if name == "whole":
        ((w * x).square().sum() / 3).backward()
    else:
        for part in (x[:1], x[1:]):
            ((w * part).square().sum() / 3).backward()
            if name == "per_micro_step":
                opt.step()
                opt.zero_grad(set_to_none=True)
    if name != "per_micro_step":
        opt.step()
    steps[name] = w.item()
assert abs(steps["whole"] - steps["accumulated"]) < 1e-7
assert abs(steps["whole"] - steps["per_micro_step"]) > 0.01
result["optimizer_order"] = steps
bn_input = torch.tensor([[1.0], [2.0], [10.0], [20.0]])
bn = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=False)
joint = bn(bn_input)
split = torch.cat([bn(v) for v in bn_input.split(2)])
assert not torch.allclose(joint, split)
result["batch_statistics_counterexample"] = {"joint": joint.flatten().tolist(), "split": split.flatten().tolist()}

path = ROOT / "docs/course-experiments/results/efficiency.json"
raw = path.read_bytes()
d = json.loads(raw)
r = d["results"]
record = {k: d[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "status")}
record.update(source_path=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(raw).hexdigest(), runtime=r["runtime"], dataset=r["dataset"], model=r["models"]["mha"]["model"], accumulation=r["accumulation"])
assert r["accumulation"]["effective_token_counts"] == [7, 13]
assert r["accumulation"]["shared_denominator"] == 20
assert f'{r["accumulation"]["gradient_max_error"]:.5e}' == "2.08616e-07"
record["branches"] = {}
for name in ("ordinary", "accumulated"):
    b = r["update_variants"][name]
    train, test = b["training"], b["heldout"]["test"]
    assert train["steps"] == train["optimizer_updates"] == 40 and train["effective_tokens"] == 2328
    assert train["batch_size"] == 8 and train["skipped_updates"] == 0
    assert train["micro_batch_sizes"] == ([8] if name == "ordinary" else [3, 5])
    assert train["peak_memory_allocated_bytes"] - train["memory_allocated_before_bytes"] == train["peak_additional_allocated_bytes"]
    assert test["matches"] == sum(v["exact"] for v in test["samples"]) == 5
    assert all(v["exact"] == (v["generated"] == v["expected"]) for v in test["samples"])
    assert test["records"] == test["examples"] == len(test["samples"]) == 10
    assert abs(test["nll_sum"] / test["effective_tokens"] - test["nll"]) < 1e-12
    assert f'{test["nll"]:.5f}' == "0.41778"
    formatted = {
        "milliseconds": f'{train["warm_step_median_seconds"] * 1000:.3f}',
        "additional_MiB": f'{train["peak_additional_allocated_bytes"] / 2**20:.3f}',
        "peak_MiB": f'{train["peak_memory_allocated_bytes"] / 2**20:.3f}',
        "baseline_MiB": f'{train["memory_allocated_before_bytes"] / 2**20:.3f}',
    }
    expected = {"ordinary": ["14.070", "8.824", "75.988", "67.164"], "accumulated": ["24.064", "6.474", "74.724", "68.250"]}[name]
    assert list(formatted.values()) == expected
    record["branches"][name] = {"training": train, "test": test, "formatted": formatted}
error = r["update_variants"]["accumulated"]["weight_max_error_from_ordinary_after_updates"]
assert f"{error:.5e}" == "1.38730e-05"
record["weight_max_error"] = error
assert record["model"]["config"]["heads"] == 4
result["recorded_experiment"] = record

source = json.loads((Path(__file__).parent / "experiment-source-excerpts.json").read_text())
assert source["original_sha256"] == d["code_sha256"]["scripts/course_experiments/architecture.py"]
assert all(v["identical_in_current"] for v in source["functions"].values())
result["experiment_code_identity"] = {"recorded_source_sha256": source["original_sha256"], "available_commit": source["commit"], "identical_functions": list(source["functions"])}
result["status"] = "全部核對通過"
print(json.dumps(result, ensure_ascii=False, indent=2))
