"""G.2 reviewer CPU checks: exact frozen examples plus explicit boundary checks."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
SECTION = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/G.2.md"
pages = {p["page_id"]: p for p in json.loads(FREEZE.read_bytes())["pages"]}
torch.set_num_threads(1)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def block(lesson_id, index=0):
    page = pages[lesson_id]
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert digest(raw) == page["source_sha256"]
    code = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)[index]
    return code, {"lesson_id": lesson_id, "snapshot": page["snapshot"],
                  "source_sha256": digest(raw), "python_block_index": index,
                  "executed_code_sha256": digest(code.encode())}


def execute(lesson_id, index=0, dependencies=None):
    code, receipt = block(lesson_id, index)
    namespace = {"__name__": "__main__"}
    namespace.update(dependencies or {})
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, f"{receipt['snapshot']}:python-block-{index}", "exec"), namespace)
    receipt["dependency_names"] = sorted((dependencies or {}).keys())
    receipt["stdout"] = output.getvalue()
    receipt["result"] = "completed; all original asserts passed"
    return namespace, receipt


def main():
    raw = SECTION.read_bytes()
    assert digest(raw) == "4233bc2da53655266f21c583f1ebeadb5a090f4096b2a6ab61e2351669f8eb36"
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-g-2-v1/verify_g2_cpu.py",
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "torch_git_version": torch.version.git_version, "device": "cpu",
                        "platform": platform.platform(), "threads": str(torch.get_num_threads())},
        "source_sha256": digest(raw), "frozen_examples": [],
        "scope": "Deterministic CPU mechanism checks and tiny optimizer updates only. No formal training, GPU, quality, speed, or generalization claim.",
    }
    # Execute the actual first Python blocks from the linked frozen lessons.
    for lesson_id in ["1.5", "1.10", "5.4", "5.7", "7.21", "7.22", "9.18", "17.17"]:
        _, receipt = execute(lesson_id)
        result["frozen_examples"].append(receipt)

    # Inspect and execute the *in-lesson* render_chat definition, then compare
    # its actual X/Y with the lesson's first block that calls the module helper.
    from tiny_perceptron.data import IGNORE, ByteTokenizer

    inline_chat, receipt = execute("7.1", 1, {"torch": torch, "IGNORE": IGNORE, "ByteTokenizer": ByteTokenizer})
    result["frozen_examples"].append(receipt)
    helper_chat, receipt = execute("7.1")
    x, y = inline_chat["render_chat"](helper_chat["messages"], ByteTokenizer())
    assert torch.equal(x, helper_chat["x"]) and torch.equal(y, helper_chat["y"])
    assert y[y != -100].tolist() == [58, 2]
    receipt["inline_definition_matches_helper"] = True
    result["frozen_examples"].append(receipt)

    # Execute the full in-lesson LoRALinear, not only the imported module class.
    inline_lora, receipt = execute("8.8", 1, {"torch": torch, "nn": torch.nn})
    result["frozen_examples"].append(receipt)
    torch.manual_seed(0)
    layer = inline_lora["LoRALinear"](torch.nn.Linear(16, 12), rank=2, alpha=2)
    x = torch.randn(3, 16)
    assert torch.equal(layer(x), layer.base(x))
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 56
    layer(x).sum().backward()
    assert layer.a.grad.abs().max().item() == 0 and layer.b.grad.abs().max().item() > 0
    assert not layer.base.weight.requires_grad
    _, receipt = execute("8.8")
    receipt["inline_definition_shape_parameter_gradient_checks"] = "passed"
    result["frozen_examples"].append(receipt)

    # G.2's own numbers and literal inline API operations.
    scores = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    p = scores.softmax(-1)
    expected = [0.665, 0.245, 0.090]
    assert all(abs(a - b) <= 0.0005 for a, b in zip(p.tolist(), expected))
    ce = F.cross_entropy(scores[None], torch.tensor([2]))
    assert abs(ce.item() + math.log(p[2].item())) < 1e-14
    labels = torch.tensor([2, -100, 0])
    rows = scores.repeat(3, 1).requires_grad_()
    terms = F.cross_entropy(rows, labels, ignore_index=-100, reduction="none")
    total = F.cross_entropy(rows, labels, ignore_index=-100, reduction="sum")
    mean = F.cross_entropy(rows, labels, ignore_index=-100, reduction="mean")
    assert terms[1].item() == 0 and torch.allclose(total / 2, mean)
    mean.backward()
    assert torch.equal(rows.grad[1], torch.zeros(3, dtype=torch.float64))
    sequence = [0, 1, 2, 3]
    shifted_x, shifted_y = sequence[:-1], sequence[1:]
    assert sequence == [0, 1, 2, 3] and list(zip(shifted_x, shifted_y)) == [(0, 1), (1, 2), (2, 3)]
    nan_row = torch.full((3,), float("-inf")).softmax(-1)
    assert nan_row.isnan().all()
    result["g2_numeric_and_mask_checks"] = {"scores": scores.tolist(), "softmax": p.tolist(),
        "target_index": 2, "cross_entropy": ce.item(), "loss_terms": terms.detach().tolist(),
        "effective_targets": 2, "total_nll": total.item(), "mean_nll": mean.item(),
        "ignored_position_gradient": rows.grad[1].tolist(), "original_sequence": sequence,
        "shifted_pairs": list(zip(shifted_x, shifted_y)), "all_negative_infinity_softmax_is_nan": nan_row.isnan().tolist()}

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.square().sum().backward()
    gradient = parameter.grad.clone()
    optimizer.step()
    assert torch.allclose(parameter, torch.tensor([0.8, 1.6], dtype=torch.float64))
    optimizer.zero_grad()
    assert parameter.grad is None  # reset, not necessarily a tensor full of zeros
    adam_parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
    adam = torch.optim.Adam([adam_parameter], lr=0.001)
    steps = []
    for grad in [[1.0, 100.0], [1.0, -100.0]]:
        before = adam_parameter.detach().clone()
        adam_parameter.grad = torch.tensor(grad, dtype=torch.float64)
        adam.step()
        steps.append((before - adam_parameter.detach()).tolist())
    assert abs(steps[1][0] - steps[1][1]) > 0.0009
    state = adam.state[adam_parameter]
    result["g2_optimizer_checks"] = {"sgd_gradient": gradient.tolist(), "sgd_parameter_after_step": parameter.tolist(),
        "zero_grad_default_sets_none": parameter.grad is None, "adam_actual_updates": steps,
        "adam_state_keys": sorted(state), "m": state["exp_avg"].tolist(),
        "v": state["exp_avg_sq"].tolist(), "step": state["step"].item(),
        "saved_param_groups_include_lr": "lr" in adam.state_dict()["param_groups"][0]}

    # Boundary checks do not purport to be general theoretical proofs.
    z = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
    temperatures = {str(t): (z / t).softmax(-1).tolist() for t in [0.5, 1.0, 2.0]}
    assert all((z / t).argmax().item() == z.argmax().item() for t in [0.5, 1.0, 2.0])
    counts = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
    normalized = counts / counts.sum(-1, keepdim=True)
    smooth = counts + 1
    smooth = smooth / smooth.sum(-1, keepdim=True)
    assert normalized[0].isnan().all() and torch.allclose(smooth.sum(-1), torch.ones(2, dtype=torch.float64))
    branched = torch.tensor(3.0, requires_grad=True)
    (2 * branched + 3 * branched).backward()
    assert branched.grad.item() == 5
    result["boundary_checks"] = {"temperature_probabilities": temperatures,
        "positive_temperature_argmax_preserved": True,
        "unsmoothed_empty_bigram_row_is_undefined": normalized[0].isnan().tolist(),
        "add_one_smoothed_row_sums": smooth.sum(-1).tolist(),
        "branched_gradient_is_sum_of_path_contributions": branched.grad.item(),
        "interpretation": "Bigram normalization requires a nonzero total (linked 1.5 supplies smoothing). Backprop multiplies along a path and sums branches (linked 1.10 explicitly says so)."}
    result["result"] = "all checks passed"
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
