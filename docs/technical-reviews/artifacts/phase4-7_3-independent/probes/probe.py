"""Independent bounded CPU checks for 7.3; random model, gradients only, no step."""
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum, masked_loss
from tiny_perceptron.attention import attention_mask

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
OUT = Path(__file__).parent
tok = ByteTokenizer()
result = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "threads": str(torch.get_num_threads()), "cwd": str(Path.cwd()), "executable": sys.executable}, "scope": "Local text encoding, role/shift alignment, loss and backward only; no data/model downloads, existing model evaluation, optimizer step or training."}
result["contracts"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py"]}
result["examples"] = {}
for answer, expected_count in [("答", 4), ("答案", 7), ("OK", 3), ("", 1)]:
    messages = [{"role": "user", "content": "問"}, {"role": "assistant", "content": answer}]
    x, y = render_chat(messages)
    content = list(answer.encode("utf-8"))
    original_ids = [1, 3] + tok.encode("問") + [2, 4] + tok.encode(answer) + [2]
    targets = [-100] * 7 + tok.encode(answer) + [2]
    assert x.tolist() == original_ids[:-1]
    assert y.tolist() == targets[1:]
    valid = y != -100
    positions = valid.nonzero().flatten().tolist()
    assert valid.sum().item() == expected_count
    assert x[positions[0]].item() == tok.assistant_id
    assert y[positions[0]].item() == (tok.encode(answer)[0] if answer else tok.eos_id)
    assert x[2:5].tolist() == tok.encode("問")
    result["examples"][answer or "EMPTY"] = {"answer_utf8_bytes": content, "question_utf8_bytes": list("問".encode("utf-8")), "original_ids": original_ids, "original_target_indices": list(range(7, len(original_ids))), "x": x.tolist(), "y": y.tolist(), "effective_indices": positions, "effective_y": y[valid].tolist(), "effective_count": int(valid.sum()), "dtype_mask": str(valid.dtype), "dtype_sum": str(valid.sum().dtype)}

multi = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "R"}, {"role": "assistant", "content": "B"}]
mx, my = render_chat(multi)
assert my[my != -100].tolist() == tok.encode("A") + [2] + tok.encode("B") + [2]
result["multi_turn"] = {"messages": multi, "x": mx.tolist(), "y": my.tolist(), "effective_indices": (my != -100).nonzero().flatten().tolist(), "effective_y": my[my != -100].tolist(), "effective_count": int((my != -100).sum())}
result["errors"] = {}
for name, func in [("no_assistant", lambda: render_chat([{"role": "user", "content": "Q"}])), ("unknown_role", lambda: render_chat([{"role": "tool", "content": "A"}])), ("nonstring_content", lambda: render_chat([{"role": "assistant", "content": None}]))]:
    try:
        func()
    except ValueError as error:
        result["errors"][name] = {"type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError(name)

x, y = render_chat([{"role": "user", "content": "問"}, {"role": "assistant", "content": "答"}])
v = tok.vocab_size
uniform = torch.zeros((1, len(y), v), requires_grad=True)
total, count = loss_sum(uniform, y[None])
average = masked_loss(uniform, y[None])
assert count.item() == 4
assert abs(total.item() - 4 * math.log(v)) < 3e-6
assert abs(average.item() - math.log(v)) < 1e-6
average.backward()
assert torch.equal(uniform.grad[0, y == -100], torch.zeros_like(uniform.grad[0, y == -100]))
extra_logits = torch.cat([uniform.detach(), torch.randn(1, 3, v)], dim=1)
extra_y = torch.cat([y[None], torch.full((1, 3), -100)], dim=1)
assert torch.equal(masked_loss(extra_logits, extra_y), average.detach())
result["loss"] = {"shape": list(uniform.shape), "axes": ["batch", "input/target position", "vocabulary class"], "unit": "natural-log cross entropy per supervised target token", "denominator": count.item(), "sum": total.item(), "mean": average.item(), "expected_sum": 4 * math.log(v), "expected_mean": math.log(v), "padded_mean": masked_loss(extra_logits, extra_y).item(), "ignored_logits_gradient_max": uniform.grad[0, y == -100].abs().max().item(), "sum_tolerance": 3e-6, "mean_tolerance": 1e-6}
try:
    masked_loss(torch.zeros(1, 2, v), torch.full((1, 2), -100))
except ValueError as error:
    result["errors"]["all_ignored_loss"] = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("all_ignored_loss")
try:
    pad_batch([(x, y)], max_length=6)
except ValueError as error:
    result["errors"]["truncated_before_answer"] = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("truncated_before_answer")

torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8))
embedded = model.embedding(x[None])
embedded.retain_grad()
logits = model(embeddings=embedded)["logits"]
logits.retain_grad()
masked_loss(logits, y[None]).backward()
allowed = attention_mask(torch.arange(len(x)), torch.arange(len(x)))
qnorms = embedded.grad[0, 2:5].norm(dim=-1)
assert torch.all(qnorms > 0)
assert logits.grad[0, y == -100].abs().max().item() == 0
assert allowed[0, 0, 6, 2:5].all().item()
assert not allowed[0, 0, 6, 7:].any().item()
assert all(token >= 0 and token < v for token in x.tolist()) and -100 not in x.tolist()
try:
    model.embedding(torch.tensor([-100]))
except IndexError as error:
    result["errors"]["negative_input_id"] = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("negative_input_id")
result["context_and_gradients"] = {"seed": 42, "width": 8, "tied": str(model.config.tied), "no_optimizer_step": True, "ignored_direct_logits_grad_max": logits.grad[0, y == -100].abs().max().item(), "question_input_positions": [2, 3, 4], "question_input_grad_norms": qnorms.tolist(), "question_embedding_parameter_row_grad_norms": model.embedding.weight.grad[x[2:5]].norm(dim=-1).tolist(), "first_answer_predictor_index": 6, "allowed_question_keys": allowed[0, 0, 6, 2:5].tolist(), "future_keys_allowed": allowed[0, 0, 6, 7:].tolist()}
(OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
