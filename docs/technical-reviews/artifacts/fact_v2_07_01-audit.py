"""Execute section 7.1 verbatim, its exercise, and bounded format/loss checks."""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.check_technical_reviews import sections
from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import IGNORE, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_07_01-"
body = dict(sections(ROOT / "course/chapters/07.md"))["7.1"]
code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
(ART / (PREFIX + "original-code.txt")).write_bytes(code.encode())


def run_snippet(snippet):
    namespace = {}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(snippet, "course/chapters/07.md#7.1", "exec"), namespace)
    return namespace, buffer.getvalue()


original, original_stdout = run_snippet(code)
x, y = original["x"], original["y"]
expected_x = [1, 3, 57, 51, 57, 69, 71, 2, 4, 58]
expected_y = [-100] * 8 + [58, 2]
assert x.tolist() == expected_x
assert y.tolist() == expected_y
assert x.dtype == y.dtype == torch.int64
assert x.device.type == y.device.type == "cpu"
assert y[y != -100].tolist() == [58, 2]
(ART / (PREFIX + "original-stdout.txt")).write_text(original_stdout)

exercise_code = code.replace('"1+1=?"', '"2+1=?"')
(ART / (PREFIX + "exercise-code.txt")).write_bytes(exercise_code.encode())
exercise, exercise_stdout = run_snippet(exercise_code)
changed = (exercise["x"] != x).nonzero().flatten().tolist()
assert changed == [2]
assert exercise["x"][2].item() == 58
assert exercise["y"].tolist() == expected_y
assert exercise["x"].shape == x.shape
(ART / (PREFIX + "exercise-stdout.txt")).write_text(exercise_stdout)
corrected_x, corrected_y = render_chat(
    [{"role": "user", "content": "2+1=?"}, {"role": "assistant", "content": "3"}]
)
assert corrected_y[corrected_y != IGNORE].tolist() == [59, 2]
assert corrected_x[-1].item() == 59

tok = ByteTokenizer()
encoding = {}
for text in ["1+1=?", "2", "cat", "貓", "🙂", "<assistant>"]:
    ids = tok.encode(text)
    assert len(ids) == len(text.encode("utf-8"))
    assert all(8 <= token <= 263 for token in ids)
    assert tok.decode(ids) == text
    encoding[text] = {"bytes": list(text.encode("utf-8")), "ids": ids}
assert len(tok.encode("貓")) == 3
assert len(tok.encode("cat")) == 3
assert len(tok.encode("🙂")) == 4
assert 4 not in tok.encode("<assistant>")

literal_x, literal_y = render_chat(
    [{"role": "user", "content": "<assistant>"}, {"role": "assistant", "content": "2"}]
)
assert literal_x.tolist().count(4) == 1
assert literal_x.tolist().index(4) == 14
assert literal_y[literal_y != IGNORE].tolist() == [58, 2]
multi_messages = [
    {"role": "user", "content": "A"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content": "C"},
    {"role": "assistant", "content": "D"},
]
multi_x, multi_y = render_chat(multi_messages)
assert multi_x.tolist() == [1, 3, 73, 2, 4, 74, 2, 3, 75, 2, 4, 76]
assert multi_y.tolist() == [-100, -100, -100, -100, 74, 2, -100, -100, -100, -100, 76, 2]
assert multi_y[multi_y != IGNORE].tolist() == [74, 2, 76, 2]

torch.manual_seed(1701)
logits = torch.randn(1, len(x), 264, dtype=torch.float64, requires_grad=True)
loss = masked_loss(logits, y[None])
selected = F.cross_entropy(logits[0, y != IGNORE], y[y != IGNORE])
assert torch.allclose(loss, selected, atol=1e-12, rtol=0)
loss.backward()
assert logits.grad[0, :8].abs().max().item() == 0
assert logits.grad[0, 8:].abs().max().item() > 0
positions = torch.arange(len(x))
allowed = attention_mask(positions, positions)[0, 0]
assert allowed[8, 2:7].all().item()
assert not allowed[8, 9].item()

torch.manual_seed(1701)
model = TinyLM(ModelConfig(vocab_size=264, width=8, max_length=32)).double()
model_logits = model(x[None])["logits"]
model_loss = masked_loss(model_logits, y[None])
model_loss.backward()
question_grad = model.embedding.weight.grad[57].abs().max().item()
assert question_grad > 0
with torch.no_grad():
    changed_logits = model(exercise["x"][None])["logits"]
    question_effect = (changed_logits[0, 8] - model_logits[0, 8]).abs().max().item()
assert question_effect > 0

report = {
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": "cpu",
        "platform": platform.platform(),
    },
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "configuration": {"seed": 1701, "dtype": "torch.float64", "training_updates": 0},
    "original": {
        "stdout": original_stdout,
        "x": x.tolist(),
        "y": y.tolist(),
        "length": len(x),
        "dtype": str(x.dtype),
        "effective_indices": (y != IGNORE).nonzero().flatten().tolist(),
        "effective_targets": y[y != IGNORE].tolist(),
    },
    "exercise": {
        "stdout": exercise_stdout,
        "x": exercise["x"].tolist(),
        "y": exercise["y"].tolist(),
        "changed_input_indices": changed,
        "corrected_answer_targets": corrected_y[corrected_y != IGNORE].tolist(),
    },
    "encoding": encoding,
    "literal_role": {"x": literal_x.tolist(), "y": literal_y.tolist(), "assistant_indices": [14]},
    "multi_round": {"messages": multi_messages, "x": multi_x.tolist(), "y": multi_y.tolist()},
    "masked_loss": {
        "effective_target_count": 2,
        "observed": loss.item(),
        "selected_only_cross_entropy": selected.item(),
        "absolute_difference": abs(loss.item() - selected.item()),
        "ignored_logits_gradient_max": logits.grad[0, :8].abs().max().item(),
        "effective_logits_gradient_max": logits.grad[0, 8:].abs().max().item(),
        "question_positions_visible_to_answer_prediction": allowed[8, 2:7].tolist(),
        "future_answer_byte_visible_to_answer_prediction": allowed[8, 9].item(),
        "question_embedding_gradient_max": question_grad,
        "changed_question_effect_on_answer_logits_max": question_effect,
    },
    "result": "All verbatim, exercise, token-boundary, alignment and bounded CPU loss assertions passed.",
}
(ART / (PREFIX + "audit-output.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
