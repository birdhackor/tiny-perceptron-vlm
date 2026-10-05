"""Bounded CPU checks of 7.10; never trains or saves model weights."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss

torch.set_num_threads(1)
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()
messages = [{"role": "user", "content": "很長的問題"}, {"role": "assistant", "content": "是"}]
x, y = render_chat(messages)
first = int((y != -100).nonzero()[0])
assert len(messages[0]['content']) == 5
assert len(tok.encode(messages[0]['content'])) == 15
assert len(tok.encode(messages[1]['content'])) == 3
assert first == 18 and len(x) == len(y) == 22
assert x[first].item() == tok.assistant_id and y[first].item() == tok.encode('是')[0]
assert y[y != -100].tolist() == tok.encode('是') + [tok.eos_id]

def must_raise(call, expected):
    try:
        call()
    except ValueError as error:
        assert expected in str(error), (expected, str(error))
        return {"type": "ValueError", "message": str(error)}
    raise AssertionError('expected ValueError')

length_results = []
for length in (0, 1, 2, 18, 19, 20, 21, 22, 23):
    if length < 19:
        length_results.append({"max_length": length, "result": must_raise(lambda: pad_batch([(x, y)], max_length=length), '第 0 筆截斷後没有有效答案')})
        continue
    bx, by, valid = pad_batch([(x, y)], max_length=length)
    expected_count = min(length, 22) - 18
    assert bx.shape == by.shape == valid.shape == (1, min(length, 22))
    assert valid.all() and int((by != -100).sum()) == expected_count
    assert by[0][by[0] != -100].tolist() == (tok.encode('是') + [2])[:expected_count]
    # Logits use B,T,V axes; CE flattens only B,T, and never shifts labels again.
    logits = torch.zeros((1, bx.shape[1], 264), dtype=torch.float64, requires_grad=True)
    total, count = loss_sum(logits, by)
    loss = masked_loss(logits, by)
    assert int(count) == expected_count
    assert abs(float(total.detach()) - expected_count * math.log(264)) < 1e-12
    assert abs(float(loss.detach()) - math.log(264)) < 1e-12
    loss.backward()
    gradient = logits.grad
    assert torch.count_nonzero(gradient[by == -100]) == 0
    assert float(gradient[0, first, int(by[0, first])]) < 0
    target_derivative = (1 / 264 - 1) / expected_count
    assert abs(float(gradient[0, first, int(by[0, first])]) - target_derivative) < 1e-12
    length_results.append({"max_length": length, "shape_B_T": list(bx.shape), "valid_input_positions": int(valid.sum()), "supervised_targets": int(count), "target_ids": by[by != -100].tolist(), "EOS_target_retained": bool((by == 2).any()), "mean_token_nll_nats": float(loss.detach()), "ignored_logits_gradient_max": float(gradient[by == -100].abs().max())})

short = render_chat([{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}])
mixed = must_raise(lambda: pad_batch([short, (x, y)], max_length=6), '第 1 筆截斷後没有有效答案')
empty_batch = must_raise(lambda: pad_batch([]), 'batch 不能是空的')
no_assistant = must_raise(lambda: render_chat([{'role': 'user', 'content': 'Q'}]), '没有可監督')
all_ignored = must_raise(lambda: loss_sum(torch.zeros(1, 2, 264), torch.full((1, 2), -100)), '所有 labels 都被忽略')
zero_positions = must_raise(lambda: loss_sum(torch.zeros(1, 0, 264), torch.full((1, 0), -100)), '所有 labels 都被忽略')
raw_all_ignored_mean = F.cross_entropy(torch.zeros(2, 264), torch.full((2,), -100))
assert torch.isnan(raw_all_ignored_mean)

bx, by, valid = pad_batch([(x, y), short], max_length=19)
assert (by != -100).sum(dim=1).tolist() == [1, 2]
assert valid.sum(dim=1).tolist() == [19, 6]
joint = torch.zeros(2, 19, 264, dtype=torch.float64, requires_grad=True)
total, count = loss_sum(joint, by)
assert int(count) == 3
assert abs(float(masked_loss(joint, by).detach()) - math.log(264)) < 1e-12

long_pair = render_chat([{'role':'user','content':'Q' * 300}, {'role':'assistant','content':'A'}])
long_first = int((long_pair[1] != -100).nonzero()[0])
assert long_first == 303 and len(long_pair[0]) == 305
too_long_prompt = must_raise(lambda: pad_batch([long_pair], max_length=128), '没有有效答案')
bounded_model = TinyLM(ModelConfig(width=8, layers=1, max_length=128))
too_long_model = must_raise(lambda: bounded_model(long_pair[0][None]), '序列超過 max_length')
empty_answer = render_chat([{'role':'user','content':'Q'}, {'role':'assistant','content':''}])
assert empty_answer[1][empty_answer[1] != -100].tolist() == [2]

# Independent gradient check with an actual short randomly initialized model;
# backward only, no optimizer and no parameter update.
model = TinyLM(ModelConfig(width=8, layers=1, max_length=22))
before = {name: value.detach().clone() for name, value in model.named_parameters()}
bx, by, valid = pad_batch([(x, y)], max_length=19)
logits = model(bx, valid=valid)['logits']
logits.retain_grad()
loss = masked_loss(logits, by)
loss.backward()
assert torch.count_nonzero(logits.grad[by == -100]) == 0
question_id = tok.encode('很')[0]
question_gradient_norm = float(model.embedding.weight.grad[question_id].norm())
assert question_gradient_norm > 0
assert all(torch.equal(before[name], value) for name, value in model.named_parameters())

result = {"environment": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "torch_cuda_build": str(torch.version.cuda)}, "units": "sequence lengths count aligned X/Y positions; answer tokens count UTF-8 bytes plus EOS; nll uses natural logarithms", "initial": {"X": x.tolist(), "Y": y.tolist(), "first_supervised_index": first, "full_X_Y_length": len(x)}, "length_variations": length_results, "mixed_batch_rejection": mixed, "empty_batch": empty_batch, "no_assistant": no_assistant, "all_ignored": all_ignored, "zero_positions": zero_positions, "raw_torch_all_ignored_mean_is_nan": bool(torch.isnan(raw_all_ignored_mean)), "mixed_valid_denominator": {"per_row": [1, 2], "combined": int(count), "padded_B_T": [2, 19]}, "long_prompt": {"first": long_first, "X_Y_length": len(long_pair[0]), "pad_rejection": too_long_prompt, "model_rejection": too_long_model}, "empty_answer_EOS_only_count": 1, "backward_only": {"loss": float(loss.detach()), "ignored_logits_gradient_max": float(logits.grad[by == -100].abs().max()), "question_embedding_gradient_norm": question_gradient_norm, "all_parameters_unchanged": True, "optimizer_steps": 0}, "tolerance": "counts and IDs exact; uniform CE loss and target derivative absolute tolerance 1e-12 float64; ignored gradient exactly zero"}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
