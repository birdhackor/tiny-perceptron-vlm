"""10.8 independent bounded CPU checks; no saved model or historical experiment rerun."""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_perceptron.data import ByteTokenizer


torch.set_num_threads(1)
root = Path(__file__).resolve().parents[4]
source = root / "course/chapters/10.md"
body = re.search(r"^## 10\.8\b.*?(?=^## |\Z)", source.read_text(), re.M | re.S)[0]
snippet = re.search(r"```python\n(.*?)```", body, re.S)[1]
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(snippet, "course/chapters/10.md#10.8", "exec"), namespace)
lm, wrapped = namespace["lm"], namespace["multimodal"]
a, b = namespace["a"], namespace["b"]
assert torch.equal(a, b) and tuple(a.shape) == (1, 3, 264)
assert wrapped.language is lm
shared_parameter_objects = all(
    value is dict(wrapped.language.named_parameters())[name] for name, value in lm.named_parameters()
)
assert shared_parameter_objects

records = {
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": str(a.device),
        "dtype": str(a.dtype),
        "threads": str(torch.get_num_threads()),
    },
    "original_fenced_code_stdout": stdout.getvalue(),
    "original": {
        "input_ids": namespace["ids"].tolist(),
        "direct_input_shape": list(namespace["ids"][None].shape),
        "direct_output_shape": list(a.shape),
        "wrapped_output_shape": list(b.shape),
        "equal": torch.equal(a, b),
        "max_abs_difference": float((a - b).abs().max().detach()),
        "finite": bool(torch.isfinite(a).all()),
        "same_language_object": wrapped.language is lm,
        "same_parameter_objects": shared_parameter_objects,
        "all_language_modules_eval": all(not module.training for module in lm.modules()),
    },
    "vocabulary": {
        "model_vocab_size": lm.config.vocab_size,
        "tokenizer_vocab_size": ByteTokenizer.vocab_size,
        "image_id": ByteTokenizer.image_id,
        "audio_id": ByteTokenizer.audio_id,
    },
}

with torch.no_grad():
    exercise = torch.tensor([1, 40, 50])
    ex_a = lm(exercise[None])["logits"]
    ex_b = wrapped(exercise)["logits"]
    assert tuple(ex_a.shape) == (1, 3, 264) and torch.equal(ex_a, ex_b)
    records["exercise"] = {
        "input_ids": exercise.tolist(),
        "shape": list(ex_a.shape),
        "equal": torch.equal(ex_a, ex_b),
        "max_abs_difference": float((ex_a - ex_b).abs().max()),
    }
    lm.train()
    train_output = lm(namespace["ids"][None])["logits"]
    lm.eval()
    eval_output = lm(namespace["ids"][None])["logits"]
    assert torch.equal(train_output, eval_output)
    records["mode_comparison"] = {
        "same_parameter_state": True,
        "train_eval_equal": torch.equal(train_output, eval_output),
        "max_abs_difference": float((train_output - eval_output).abs().max()),
        "module_types": sorted({type(module).__name__ for module in lm.modules()}),
    }

records["missing_modalities"] = []
for marker in (ByteTokenizer.image_id, ByteTokenizer.audio_id):
    bad_ids = torch.tensor([1, marker, 4])
    try:
        wrapped(bad_ids)
    except ValueError as exc:
        assert str(exc) == "placeholder 缺少配對圖片／聲音"
        records["missing_modalities"].append({"input_ids": bad_ids.tolist(), "exception": type(exc).__name__, "message": str(exc)})
    else:
        raise AssertionError("Missing modality placeholder silently accepted")

# One small update to this newly initialized toy TinyLM, never a historical model.
# This witnesses possible logit change. It makes no capability or forgetting claim.
ids = namespace["ids"]
targets = torch.tensor([[20, 30, 2]])
before = lm(ids[None])["logits"].detach().clone()
before_params = {name: value.detach().clone() for name, value in lm.named_parameters()}
optimizer = torch.optim.SGD(lm.parameters(), lr=0.01)
optimizer.zero_grad(set_to_none=True)
scores = lm(ids[None])["logits"]
loss = F.cross_entropy(scores.reshape(-1, 264), targets.reshape(-1))
loss.backward()
nonzero_gradients = [name for name, value in lm.named_parameters() if value.grad is not None and bool(value.grad.ne(0).any())]
optimizer.step()
after = lm(ids[None])["logits"].detach()
after_wrapped = wrapped(ids)["logits"].detach()
changed_parameters = [name for name, value in lm.named_parameters() if not torch.equal(before_params[name], value)]
assert changed_parameters and not torch.equal(before, after) and torch.equal(after, after_wrapped)
records["one_update_witness"] = {
    "seed": 0,
    "optimizer": "SGD",
    "learning_rate": 0.01,
    "optimizer_updates": 1,
    "effective_target_tokens": 3,
    "samples": 1,
    "input_ids": ids.tolist(),
    "targets": targets.tolist(),
    "loss_before": float(loss.detach()),
    "nonzero_gradient_parameter_count": len(nonzero_gradients),
    "changed_parameter_count": len(changed_parameters),
    "max_abs_logit_change": float((before - after).abs().max()),
    "before_after_equal": torch.equal(before, after),
    "direct_wrapped_still_equal_after_update": torch.equal(after, after_wrapped),
    "scope": "New random toy instance, one CPU optimizer step; no trained-product or historical-result reproduction, no skill-retention conclusion.",
}
print(json.dumps(records, ensure_ascii=False, indent=2))
