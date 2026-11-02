"""Independent review of frozen lesson 1.10; small scalar CPU examples only."""

import contextlib
import hashlib
import io
import json
import platform
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("cpu-check.json")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json"
page = next(p for p in json.loads(FREEZE.read_bytes())["pages"] if p["page_id"] == "1.10")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
code_blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(code_blocks) == 1
code = code_blocks[0]


def run_inline(source):
    capture = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(capture):
        exec(compile(source, str(ROOT / page["snapshot"]), "exec"), namespace)
    w, u, loss = (namespace[k] for k in ("w", "u", "loss"))
    detached = u.detach()
    return {
        "stdout": capture.getvalue(),
        "w": w.item(),
        "u": u.item(),
        "loss": loss.item(),
        "gradient": w.grad.item(),
        "dtype": str(w.dtype),
        "device": str(w.device),
        "w_is_leaf": w.is_leaf,
        "w_requires_grad": w.requires_grad,
        "u_requires_grad": u.requires_grad,
        "u_grad_fn": type(u.grad_fn).__name__,
        "loss_grad_fn": type(loss.grad_fn).__name__,
        "detached_requires_grad": detached.requires_grad,
        "detached_grad_fn": detached.grad_fn,
        "detach_shares_storage": detached.data_ptr() == u.data_ptr(),
    }


base = run_inline(code)
assert (base["w"], base["u"], base["loss"], base["gradient"]) == (3, 6, 36, 24)
exercise_code = code.replace("u = 2 * w", "u = 3 * w").replace(
    'print("第一段 du/dw", 2,', 'print("第一段 du/dw", 3,'
).replace("assert w.grad.item() == 24", "assert w.grad.item() == 54")
assert exercise_code != code
exercise = run_inline(exercise_code)
assert (exercise["w"], exercise["u"], exercise["loss"], exercise["gradient"]) == (3, 9, 81, 54)

# Decimal is an exact finite-increment audit, separate from autograd.
w_exact, step = Decimal("3"), Decimal("0.001")
u_exact = 2 * w_exact
loss_exact = u_exact**2
new_u = 2 * (w_exact + step)
delta = new_u**2 - loss_exact
linear_delta = (2 * u_exact) * (2 * step)
assert delta == Decimal("0.024004")
assert linear_delta == Decimal("0.024")

# The optional note concerns summing distinct paths. This is a reviewer example.
branch_w = torch.tensor(3.0, requires_grad=True)
branch_loss = (2 * branch_w).square() + (3 * branch_w).square()
branch_loss.backward()
assert branch_w.grad.item() == 24 + 54 == 78

notebook_path = ROOT / page["notebook"]
notebook_raw = notebook_path.read_bytes()
assert hashlib.sha256(notebook_raw).hexdigest() == page["notebook_sha256"]
notebook = json.loads(notebook_raw)
matched_cells = [i for i, cell in enumerate(notebook["cells"]) if
                 cell["cell_type"] == "code" and "".join(cell["source"]).strip() == code.strip()]
assert matched_cells == [5]

result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_p5_technical_1_10",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "CPU", "platform": platform.platform()},
    "frozen_source": {"path": page["snapshot"], "sha256": page["source_sha256"]},
    "inline_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "base_inline_exact_execution": base,
    "exercise_modified_inline_execution": exercise,
    "finite_increment_decimal": {
        "step": str(step), "u_increment": str(new_u - u_exact),
        "exact_loss_increment": str(delta), "first_order_loss_increment": str(linear_delta),
        "exact_forward_difference_quotient": str(delta / step),
        "derivative_at_3": "24", "first_order_absolute_error": str(delta - linear_delta),
        "interpretation": "0.024 is explicitly approximate; 0.024004 is the exact finite increment.",
    },
    "reviewer_two_path_example": {"formula": "L=(2*w)^2+(3*w)^2 at w=3",
                                  "path_contributions": [24, 54],
                                  "observed_gradient": branch_w.grad.item()},
    "notebook": {"path": page["notebook"], "sha256": page["notebook_sha256"],
                 "exact_lesson_code_cell_indices": matched_cells,
                 "scope": "Compared the full notebook code cells; exact lesson code executed. Colab bootstrap not run."},
    "result": "all scalar CPU checks passed; no model, training, optimizer update, network weights or final-family inference",
}
with OUT.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
