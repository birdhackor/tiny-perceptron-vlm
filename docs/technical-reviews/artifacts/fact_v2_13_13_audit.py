"""Run the exact 13.13 lesson and exercise, plus a small CPU clipping audit."""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from tiny_perceptron.posttraining import ppo_clipped_objective

OUT = Path("docs/technical-reviews/artifacts")
BODY = dict(sections(Path("course/chapters/13.md")))["13.13"]
CODE = re.findall(r"```python\n(.*?)```", BODY, re.S)[0]
ROWS = []

for clip_range, expected in (
    (0.2, [0.7, 1.0, 1.2, -0.8, -1.0, -1.3]),
    (0.1, [0.7, 1.0, 1.1, -0.9, -1.0, -1.3]),
):
    namespace = {}
    output = io.StringIO()
    executed_code = CODE.replace("clip_range=0.2", f"clip_range={clip_range}")
    with contextlib.redirect_stdout(output):
        exec(compile(executed_code, f"13.13-clip-{clip_range}", "exec"), namespace)
    terms = namespace["result"]
    ratio = namespace["ratio"]
    torch.testing.assert_close(terms["surrogate"], torch.tensor(expected), rtol=0, atol=2e-7)
    torch.testing.assert_close(ratio.grad, torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]), rtol=0, atol=2e-7)
    probabilities = namespace["new_log_probability"].exp()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    ROWS.append(
        {
            "clip_range": clip_range,
            "stdout": output.getvalue(),
            "dtype": str(ratio.dtype),
            "shape": list(ratio.shape),
            "input_ratio": ratio.detach().tolist(),
            "advantage": namespace["advantage"].tolist(),
            "old_probability": namespace["old_log_probability"].exp().tolist(),
            "new_probability": probabilities.detach().tolist(),
            "unclipped": terms["unclipped"].detach().tolist(),
            "clipped": terms["clipped"].detach().tolist(),
            "surrogate": terms["surrogate"].detach().tolist(),
            "sum_loss": namespace["loss"].item(),
            "mean_policy_loss": terms["policy_loss"].item(),
            "ratio_gradient_for_sum_loss": ratio.grad.tolist(),
            "denominator": "6 independent selected state-action samples; example sums, API policy_loss means over 6",
        }
    )

# Check the whole piecewise curve, with no nondifferentiable threshold in the gradient checks.
grid = torch.linspace(0.4, 1.6, 121, dtype=torch.float64)
for advantage in (1.0, -1.0):
    terms = ppo_clipped_objective(grid.log(), torch.zeros_like(grid), torch.full_like(grid, advantage))
    expected = torch.minimum(grid, torch.full_like(grid, 1.2)) if advantage > 0 else -torch.maximum(
        grid, torch.full_like(grid, 0.8)
    )
    torch.testing.assert_close(terms["surrogate"], expected, rtol=0, atol=5e-16)

# A plateau in one sample is not a hard bound on the shared policy parameter.
parameter = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
shared_ratios = torch.stack([parameter, parameter - 0.6])
shared_terms = ppo_clipped_objective(shared_ratios.log(), torch.zeros(2), torch.ones(2))
individual_grads = [
    torch.autograd.grad(-shared_terms["surrogate"][i], parameter, retain_graph=True)[0].item()
    for i in range(2)
]
shared_loss = -shared_terms["surrogate"].sum()
shared_loss.backward()
updated_parameter = parameter.detach() - 0.1 * parameter.grad
updated_probabilities = torch.stack([updated_parameter, updated_parameter - 0.6]) * 0.5
assert individual_grads == [0.0, -1.0]
assert updated_parameter.item() > 1.2
assert updated_probabilities[0].item() > 0.6
assert ((updated_probabilities > 0) & (updated_probabilities < 1)).all()

result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_13_audit.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "platform": platform.platform(),
    },
    "source_sha256": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
    "exact_lesson_code": CODE,
    "result": "Exact lesson and exercise passed all value/gradient checks; 242 curve values checked; shared-parameter counterexample passed.",
    "runs": ROWS,
    "curve_grid": {"points_per_advantage": 121, "range": [0.4, 1.6], "dtype": "torch.float64"},
    "shared_parameter_counterexample": {
        "construction": "Two distinct binary contexts with selected probabilities .5*theta and .5*(theta-.6); old .5 each.",
        "before_ratios": shared_ratios.detach().tolist(),
        "individual_gradients_for_negative_surrogate": individual_grads,
        "step_size": 0.1,
        "after_parameter": updated_parameter.item(),
        "after_selected_probabilities": updated_probabilities.tolist(),
        "scope": "A transparent shared scalar construction; not a network training or quality experiment.",
    },
}
(OUT / "fact_v2_13_13_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
