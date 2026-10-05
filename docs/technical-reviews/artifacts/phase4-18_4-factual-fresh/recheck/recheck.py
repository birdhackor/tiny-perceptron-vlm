"""Personal CPU recheck of corrected 18.4 candidate-score direction condition."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[4]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


q = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float64)
cases = []
for name, values in [
    ("original_hard", [1.0, 0.0, 0.0]),
    ("original_soft", [0.7, 0.2, 0.1]),
    ("exercise", [0.2, 0.7, 0.1]),
    ("old_rank_counterexample", [0.34, 0.35, 0.31]),
    ("just_below_threshold", [0.501, 0.399, 0.1]),
    ("at_threshold", [0.5, 0.4, 0.1]),
    ("just_above_threshold_without_highest_rank", [0.499, 0.401, 0.1]),
    ("above_threshold", [0.1, 0.6, 0.3]),
]:
    p = torch.tensor(values, dtype=torch.float64)
    assert abs(p.sum().item() - 1.0) < 1e-14
    z = q.log().detach().clone().requires_grad_()
    before = z.detach().clone()
    ce = -(p * z.log_softmax(0)).sum()
    ce.backward()
    assert torch.equal(before, z.detach()), "backward computes but does not update"
    gradient = q - p
    torch.testing.assert_close(z.grad, gradient, atol=1e-14, rtol=0)
    assert abs(ce.item() - (-sum(pi * math.log(qi) for pi, qi in zip(values, q.tolist())))) < 1e-14
    finite_difference = []
    for i in range(3):
        step = torch.zeros_like(z)
        step[i] = 1e-6
        plus = -(p * (before + step).log_softmax(0)).sum()
        minus = -(p * (before - step).log_softmax(0)).sum()
        finite_difference.append(((plus - minus) / 2e-6).item())
    torch.testing.assert_close(torch.tensor(finite_difference, dtype=torch.float64), gradient, atol=3e-10, rtol=0)
    after = before - 0.1 * z.grad
    delta_score3 = (after[1] - before[1]).item()
    if values[1] > 0.4:
        assert z.grad[1] < 0 and delta_score3 > 0
    elif values[1] < 0.4:
        assert z.grad[1] > 0 and delta_score3 < 0
    else:
        assert abs(z.grad[1].item()) < 1e-14 and abs(delta_score3) < 1e-14
    cases.append({
        "case": name, "teacher": values, "student": q.tolist(),
        "teacher_argmax_candidate": ["4", "3", "100"][p.argmax().item()],
        "ce_nats": ce.item(), "gradient": z.grad.tolist(),
        "finite_difference": finite_difference,
        "candidate3_score_delta_lr0_1": delta_score3,
        "conditional_teacher3_gt_student3": values[1] > 0.4,
    })
old_counter = next(c for c in cases if c["case"] == "old_rank_counterexample")
assert old_counter["teacher_argmax_candidate"] == "3" and old_counter["gradient"][1] > 0
rank_not_required = next(c for c in cases if c["case"] == "just_above_threshold_without_highest_rank")
assert rank_not_required["teacher_argmax_candidate"] == "4" and rank_not_required["gradient"][1] < 0
unchanged = []
for current, frozen in [
    (ROOT / "scripts/course_experiments/compression.py", BASE / "code/compression.py"),
    (ROOT / "tiny_perceptron/data.py", BASE / "code/data.py"),
    (ROOT / "tiny_perceptron/model.py", BASE / "code/model.py"),
    (ROOT / "docs/course-experiments/results/distillation.json", BASE / "sources/original-distillation-result.json"),
    (ROOT / "outputs/private-review-artifacts/original-distillation/sft-teacher-logits.pt", BASE / "sources/sft-teacher-logits.pt"),
    (ROOT / "outputs/private-review-artifacts/original-distillation/sft-hard-targets.json", BASE / "sources/sft-hard-targets.json"),
]:
    assert current.read_bytes() == frozen.read_bytes()
    unchanged.append({"current": str(current.relative_to(ROOT)), "formal_snapshot": str(frozen.relative_to(ROOT)), "sha256": sha(frozen)})
result = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "dtype": "float64"},
    "scope": "Single-position synthetic tensors only; no trained model execution or quality evaluation",
    "source_sha256": sha(HERE / "section.md"),
    "current_fence_sha256": sha(HERE / "fence-1.py"),
    "initial_fence_sha256": sha(BASE / "fence-1.py"),
    "candidate_axis": 0, "ce_unit": "nats", "positions": 1,
    "cases": cases, "unchanged_sources": unchanged,
    "conditional_proof": "For normalized p, derivative dCE/dz3=q3-p3=.4-p3; any positive direct score step along -gradient raises z3 iff p3>.4. Highest rank alone is neither necessary nor sufficient.",
    "all_assertions_passed": True,
}
(HERE / "recheck-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
