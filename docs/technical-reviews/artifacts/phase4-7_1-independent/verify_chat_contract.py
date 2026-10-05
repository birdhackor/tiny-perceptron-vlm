"""Bounded CPU checks of current render_chat, exact byte/shift units and loss denominator."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE, SPECIALS, render_chat
from tiny_perceptron.model import masked_loss, loss_sum
from tiny_perceptron.attention import attention_mask

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")
tok = ByteTokenizer()
messages = [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}]
x, y = render_chat(messages, tok)
original = [1, 3] + [b + 8 for b in b"1+1=?"] + [2, 4] + [b + 8 for b in b"2"] + [2]
assert original == [1, 3, 57, 51, 57, 69, 71, 2, 4, 58, 2]
assert x.tolist() == original[:-1] == [1, 3, 57, 51, 57, 69, 71, 2, 4, 58]
assert y.tolist() == [-100] * 8 + [58, 2]
assert y[y != IGNORE].tolist() == [58, 2]
assert x.shape == y.shape == (10,) and x.dtype == y.dtype == torch.int64
assert x[8].item() == tok.assistant_id and y[8].item() == tok.encode("2")[0]
assert x[9].item() == 58 and y[9].item() == tok.eos_id
dx, dy = render_chat(messages)
assert torch.equal(dx, x) and torch.equal(dy, y)
assert tok.encode("貓🙂") == [b + 8 for b in "貓🙂".encode("utf-8")]
assert len(tok.encode("貓🙂")) == 7 and tok.decode(tok.encode("貓🙂")) == "貓🙂"
print("default_tokenizer_and_utf8_roundtrip", {"default_equals_explicit": True, "unicode_content": "貓🙂", "unicode_bytes_and_tokens": 7})
print("original_sequence", original)
print("per_position", [{"index": i, "input": a, "target": b, "active": b != IGNORE} for i, (a, b) in enumerate(zip(x.tolist(), y.tolist(), strict=True))])
print("units", {"ordinary_question_utf8_bytes": 5, "ordinary_answer_utf8_bytes": 1, "original_tokens": 11, "input_positions": 10, "target_positions": 10, "valid_targets": 2})

change = [{"role": "user", "content": "2+1=?"}, {"role": "assistant", "content": "2"}]
vx, vy = render_chat(change, tok)
different = (vx != x).nonzero().flatten().tolist()
assert different == [2] and vx[2].item() == 58
assert torch.equal(vy, y) and len(vx) == 10
correct = [{"role": "user", "content": "2+1=?"}, {"role": "assistant", "content": "3"}]
cx, cy = render_chat(correct, tok)
assert cy[cy != IGNORE].tolist() == [59, 2]
print("exercise_format_only", {"changed_input_indices": different, "changed_input_id": vx[2].item(), "valid_targets": vy[vy != IGNORE].tolist(), "arithmetic_truth": "2+1=3; fixed answer 2 isolates format only"})
print("corrected_arithmetic_targets", cy[cy != IGNORE].tolist())

literal = [{"role": "user", "content": "<assistant>"}, {"role": "assistant", "content": "2"}]
lx, ly = render_chat(literal, tok)
assert all(i >= 8 for i in tok.encode("<assistant>"))
assert (lx == tok.assistant_id).nonzero().flatten().tolist() == [14]
assert tok.decode(tok.encode("<assistant>")) == "<assistant>"
multi = messages + [{"role": "user", "content": "2+1=?"}, {"role": "assistant", "content": "3"}]
mx, my = render_chat(multi, tok)
assert my[my != IGNORE].tolist() == [58, 2, 59, 2]
assert (mx == tok.assistant_id).nonzero().flatten().tolist() == [8, 18]
print("literal_and_multiturn", {"literal_content_ids": tok.encode("<assistant>"), "literal_role_indices": (lx == 4).nonzero().flatten().tolist(), "multiturn_valid_targets": my[my != IGNORE].tolist(), "multiturn_assistant_indices": (mx == 4).nonzero().flatten().tolist()})
for bad in ([{"role": "other", "content": "x"}], [{"role": "assistant", "content": 2}], [{"role": "user", "content": "x"}]):
    try:
        render_chat(bad, tok)
    except ValueError as e:
        print("expected_contract_rejection", type(e).__name__, str(e))
    else:
        raise AssertionError("contract invalid input should be rejected")
sx, sy = render_chat([{"role": "system", "content": "x"}, *messages], tok)
assert sy[sy != IGNORE].tolist() == [58, 2] and tok.system_id in sx.tolist()
print("system_supported_but_not_needed_by_example", True)

positions = torch.arange(len(x))
allowed = attention_mask(positions, positions)
assert allowed.shape == (1, 1, 10, 10)
assert bool(allowed[0, 0, 8, 2:7].all())
assert bool(allowed[0, 0, 9, 2:7].all())
assert not bool(allowed[0, 0, 8, 9])
print("context_visibility", {"allowed_axes": ["batch=1", "heads=1", "query=10", "key=10"], "answer_queries": [8, 9], "question_keys": [2, 3, 4, 5, 6], "all_question_keys_visible": True, "future_key9_visible_to_query8": False, "loss_labels_passed_to_attention_mask": False})

# No model or optimiser: constructed scores test ignore_index and exact denominator only.
scores = torch.zeros((1, 10, tok.vocab_size), dtype=torch.float64)
total, count = loss_sum(scores, y[None])
mean = masked_loss(scores, y[None])
assert count.item() == 2
assert math.isclose(total.item(), 2 * math.log(264), rel_tol=0, abs_tol=1e-12)
assert math.isclose(mean.item(), math.log(264), rel_tol=0, abs_tol=1e-12)
modified = scores.clone()
modified[:, :8, :] = torch.arange(264, dtype=torch.float64) * 1000
assert torch.equal(masked_loss(modified, y[None]), mean)
print("constructed_logits_denominator", {"logits_axes": ["batch=1", "sequence=10", "class=264"], "sum_nll_nats": total.item(), "valid_target_count": count.item(), "mean_nats_per_active_target": mean.item(), "expected_ln_264": math.log(264), "ignored_logits_changed_loss": False, "tolerance_abs": 1e-12})

print("environment", json.dumps({"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "torch_cuda_build": str(torch.version.cuda), "device": "cpu", "threads": str(torch.get_num_threads()), "repo_inputs": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py")}}, ensure_ascii=False))
print("ALL_CHECKS_PASSED; no model instantiated, no training, no evaluation or model/data download")
