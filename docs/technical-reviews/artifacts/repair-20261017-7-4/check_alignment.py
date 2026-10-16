"""7.4 independent bounded CPU verification; no training or weight updates."""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


ROOT = Path(__file__).resolve().parents[4]
torch.set_num_threads(1)
tok = ByteTokenizer()
raw = (ROOT / "course/chapters/07.md").read_text(encoding="utf-8")
body = re.search(r"^## 7\.4 .*?(?=^## |\Z)", raw, re.M | re.S)[0]
freeze = json.loads((ROOT / "docs/course-repair-20261017/freeze-02.json").read_text())
entry = next(item for item in freeze["pages"] if item["page_id"] == "7.4")
body_hash = hashlib.sha256(body.encode()).hexdigest()
assert body_hash == entry["source_sha256"]
assert body == Path("/workspace/work/textbook-repair-20261017/input-freeze02/pages/7.4.md").read_text()

code = re.search(r"```python\n(.*?)```", body, re.S)[1]
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(code, "course/chapters/07.md#7.4", "exec"), {})

cases = []
for question, answer, position, answer_id in [("Q", "A", 4, 73), ("QQ", "A", 5, 73), ("QQ", "B", 5, 74)]:
    x, y = render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    indices = (y != -100).nonzero()
    first = indices[0].item()
    assert (first, x[first].item(), y[first].item()) == (position, 4, answer_id)
    assert indices.shape == (2, 1)
    assert y[-1].item() == 2
    cases.append({"question": question, "answer": answer, "x": x.tolist(), "y": y.tolist(),
                  "dtype": str(x.dtype), "nonzero_shape": list(indices.shape), "first": first,
                  "first_input": x[first].item(), "first_target": y[first].item(),
                  "effective_target_count": int((y != -100).sum())})

x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
full_ids = [1, 3, 89, 2, 4, 73, 2]
raw_targets = [-100, -100, -100, -100, -100, 73, 2]
assert x.tolist() == full_ids[:-1] and y.tolist() == raw_targets[1:]
unshifted_first = next(i for i, value in enumerate(raw_targets[:-1]) if value != -100)
assert unshifted_first == 5 and x[unshifted_first].item() == 73
wrong_same_count = torch.tensor([-100, -100, -100, 73, 2, -100])
assert int((wrong_same_count != -100).sum()) == int((y != -100).sum()) == 2
assert (wrong_same_count != -100).nonzero()[0].item() == 3

torch.manual_seed(74)
model = TinyLM(ModelConfig(width=8, heads=1, layers=1, max_length=16)).eval()
with torch.no_grad():
    logits = model(x[None])["logits"]
    changed_x = x.clone()
    changed_x[5] = tok.encode("B")[0]
    changed_logits = model(changed_x[None])["logits"]
    prefix_difference = float((logits[:, :5] - changed_logits[:, :5]).abs().max())
    assert prefix_difference <= 1e-7
    changed_question = x.clone()
    changed_question[2] = tok.encode("R")[0]
    question_logits = model(changed_question[None])["logits"]
    visible_question_difference = float((logits[:, 4] - question_logits[:, 4]).abs().max())
    assert visible_question_difference > 1e-8
    actual = masked_loss(logits, y[None])
    selected = F.cross_entropy(logits[0, [4, 5]], torch.tensor([73, 2]))
    assert torch.allclose(actual, selected, atol=1e-7, rtol=1e-7)

toy_logits = torch.zeros(1, 6, tok.vocab_size, dtype=torch.float64)
toy_logits[0, 4, 73] = 12
toy_logits[0, 5, 2] = 12
toy_logits.requires_grad_()
toy_loss = masked_loss(toy_logits, y[None])
toy_loss.backward()
ignored_grad_max = float(toy_logits.grad[0, :4].abs().max())
active_grad_abs_sum = float(toy_logits.grad[0, 4:].abs().sum())
assert ignored_grad_max == 0.0 and active_grad_abs_sum > 0.0
double_shifted = F.cross_entropy(toy_logits.detach()[:, :-1].reshape(-1, 264), y[None, 1:].reshape(-1), ignore_index=-100)
assert float(double_shifted) > float(toy_loss.detach()) + 1

print(json.dumps({
    "command": "uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-7-4/check_alignment.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": "1"},
    "scope": "Exact lesson code, three deterministic data cases, one randomly initialized TinyLM forward comparison, toy-logit loss/backward; no optimizer, no model training, no historical experiment reproduction.",
    "source_sha256": body_hash,
    "figure_sha256": hashlib.sha256((ROOT / "course/figures/rewrite-07-04-answer-alignment.svg").read_bytes()).hexdigest(),
    "byte_codes": {"Q": list(b"Q"), "A": list(b"A"), "B": list(b"B")},
    "lesson_code_stdout": capture.getvalue(),
    "cases": cases,
    "alignment": {"full_ids": full_ids, "raw_targets": raw_targets,
                  "unshifted_mask_first": unshifted_first, "input_at_unshifted_first": x[unshifted_first].item(),
                  "wrong_same_count_labels": wrong_same_count.tolist(), "wrong_same_count": 2, "wrong_first": 3},
    "model": {"seed": 74, "width": 8, "layers": 1, "heads": 1, "logit_shape": list(logits.shape),
              "causal_prefix_max_abs_difference": prefix_difference,
              "visible_question_max_abs_difference_at_assistant": visible_question_difference,
              "masked_loss": float(actual), "selected_positions_loss": float(selected)},
    "toy_loss": {"dtype": "torch.float64", "valid_positions": [4, 5], "count": 2,
                 "aligned_loss": float(toy_loss.detach()), "double_shifted_loss": float(double_shifted),
                 "ignored_grad_max": ignored_grad_max, "active_grad_abs_sum": active_grad_abs_sum},
    "result": "all assertions passed"
}, ensure_ascii=False, indent=2))
