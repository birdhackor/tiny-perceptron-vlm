"""核對 Block 的連線、設定分支與範例變化。"""

import ast
import hashlib
import json
import platform
import re
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.attention import CausalAttention, manual_attention
from tiny_perceptron.data import ByteTokenizer, SPECIALS
from tiny_perceptron.model import Block as LibraryBlock
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN, MoEFFN, RMSNorm


def read_class(path, class_name):
    for code in re.findall(r"```python\n(.*?)```", Path(path).read_text(), re.S):
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                namespace = dict(torch=torch, nn=nn, F=F, manual_attention=manual_attention,
                                 CausalAttention=CausalAttention, DenseFFN=DenseFFN)
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                return namespace[class_name]
    raise ValueError("找不到指定類別")


def maximum_error(a, b):
    return float((a - b).abs().max().detach())


def swapped(block, x):
    attended, _ = block.attention(block.norm2(x))
    u = x + attended
    return u + block.ffn(block.norm1(u))


torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
Block = read_class("docs/course-review-20261019/freeze-01/sources/4.5.md", "Block")
block = Block(ModelConfig(width=8))
x = torch.randn(2, 3, 8)
initial_weights = [block.norm1.weight.detach().tolist(), block.norm2.weight.detach().tolist()]
initial_biases = [block.norm1.bias.detach().tolist(), block.norm2.bias.detach().tolist()]
initial_swap_error = maximum_error(block(x), swapped(block, x))
with torch.no_grad():
    block.norm2.weight.fill_(2.0)
y = block(x)
library = LibraryBlock(ModelConfig(width=8))
loaded = library.load_state_dict(block.state_dict())
library_y, cache, auxiliary = library(x)
equal = torch.allclose(y, library_y)
swap_error = maximum_error(y, swapped(block, x))
assert equal and swap_error > 0 and initial_swap_error == 0
assert block.norm1.weight is not block.norm2.weight
assert block.norm1.bias is not block.norm2.bias
assert set(dict(block.named_children())) == {"norm1", "norm2", "attention", "ffn"}
assert not loaded.missing_keys and not loaded.unexpected_keys
shapes = []
for shape in [(2, 3, 8), (1, 3, 8), (2, 4, 8)]:
    sample = torch.randn(*shape)
    output = block(sample)
    assert output.shape == sample.shape
    shapes.append({"input": list(shape), "output": list(output.shape)})
try:
    block(torch.randn(2, 3, 16))
except RuntimeError as error:
    mismatch = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("特徵寬度不符時應拒絕輸入")
wide = Block(ModelConfig(width=16))
assert list(wide(torch.randn(2, 3, 16)).shape) == [2, 3, 16]

local_attention = read_class("course/chapters/03.md", "CausalAttention")
local_ffn = read_class("course/chapters/04.md", "DenseFFN")
component_checks = []
for heads in [1, 2, 4]:
    a = local_attention(8, heads)
    b = CausalAttention(8, heads)
    b.load_state_dict(a.state_dict())
    output = a(x)
    library_output, _ = b(x)
    assert torch.allclose(output, library_output)
    component_checks.append({"component": "CausalAttention", "heads": heads,
                             "maximum_error": maximum_error(output, library_output)})
a = local_ffn(8)
b = DenseFFN(8)
b.load_state_dict(a.state_dict())
assert torch.allclose(a(x), b(x))
component_checks.append({"component": "DenseFFN", "maximum_error": maximum_error(a(x), b(x))})
ffn_changed = x.clone()
ffn_changed[:, 1] += 2
assert torch.equal(b(x)[:, 0], b(ffn_changed)[:, 0])
future_changed = x.clone()
future_changed[:, 2] += 2
assert torch.equal(library(x)[0][:, :2], library(future_changed)[0][:, :2])

variants = []
for settings in [{}, {"norm": "rms"}, {"activation": "relu2"}, {"activation": "swiglu"},
                 {"kv_heads": 1}, {"backend": "sdpa"}, {"rotary": True},
                 {"experts": 4, "top_k": 2}]:
    model = LibraryBlock(ModelConfig(width=8, heads=2, **settings))
    output, new_cache, aux = model(x)
    attention_output, _ = model.attention(model.norm1(x))
    u = x + attention_output
    ffn_output = model.ffn(model.norm2(u))
    if isinstance(ffn_output, tuple):
        h, expected_aux, _ = ffn_output
    else:
        h, expected_aux = ffn_output, torch.zeros(())
    assert torch.allclose(output, u + h) and torch.allclose(aux, expected_aux)
    if settings.get("experts"):
        assert isinstance(model.ffn, MoEFFN) and aux.item() > 0
    else:
        assert isinstance(model.ffn, DenseFFN) and aux.item() == 0
    if settings.get("norm") == "rms":
        assert isinstance(model.norm1, RMSNorm) and isinstance(model.norm2, RMSNorm)
    variants.append({"settings": settings, "norm": type(model.norm1).__name__,
                     "ffn": type(model.ffn).__name__,
                     "activation": getattr(model.ffn, "activation", None),
                     "kv_heads": model.attention.kv_heads,
                     "backend": model.attention.backend, "rotary": model.attention.rotary,
                     "output_shape": list(output.shape), "cache_shapes": [list(t.shape) for t in new_cache],
                     "auxiliary": aux.item(), "maximum_error": maximum_error(output, u + h)})

seen = []
hook = library.attention.register_forward_pre_hook(lambda module, args: seen.append(args))
valid = torch.ones(2, 3, dtype=torch.bool)
positions = torch.arange(3)
segments = torch.zeros(2, 3, dtype=torch.long)
_, returned_cache, _ = library(x, valid, positions, segments, None)
assert all(actual is expected for actual, expected in zip(seen[-1][1:], (valid, positions, segments, None)))
_, prefix_cache, _ = library(x[:, :1])
_, continued_cache, _ = library(x[:, 1:], cache=prefix_cache)
assert seen[-1][-1] is prefix_cache
assert continued_cache[0].shape[2] == 3
hook.remove()

before_item = {name: p.detach().clone() for name, p in library.named_parameters()}
scalar = auxiliary.item()
assert isinstance(scalar, float) and scalar == 0
assert all(torch.equal(before_item[name], p) for name, p in library.named_parameters())
defaults = asdict(ModelConfig())
overridden = asdict(ModelConfig(width=8))
assert {k for k in defaults if defaults[k] != overridden[k]} == {"width"}
plain_model = TinyLM()
tied_model = TinyLM(ModelConfig(tied=True))
assert len(plain_model.blocks) == defaults["layers"]
assert list(plain_model.position.weight.shape) == [defaults["max_length"], defaults["width"]]
assert plain_model.output.out_features == defaults["vocab_size"]
assert tied_model.embedding.weight is tied_model.output.weight
assert ByteTokenizer.vocab_size == 256 + len(SPECIALS) == defaults["vocab_size"]

record = {
    "command": "PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-5/verify_block.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(x.dtype)},
    "seed": 42, "defaults": defaults, "overridden": overridden,
    "components": list(dict(block.named_children())),
    "independent_norm_parameters": True, "initial_weights": initial_weights, "initial_biases": initial_biases,
    "norm2_weights_after_fill": block.norm2.weight.detach().tolist(),
    "initial_swap_maximum_error": initial_swap_error, "changed_swap_maximum_error": swap_error,
    "library_allclose": equal, "library_maximum_error": maximum_error(y, library_y),
    "output_differs_from_input": not torch.equal(y, x),
    "shapes": shapes, "mismatched_width": mismatch, "matching_wide_shape": [2, 3, 16],
    "component_comparisons": component_checks,
    "ffn_position_independent": True, "block_future_positions_masked": True,
    "variants": variants, "argument_forwarding_identity": True,
    "continued_cache_shapes": [list(t.shape) for t in continued_cache],
    "item_value": scalar, "item_python_type": type(scalar).__name__, "item_parameters_unchanged": True,
    "tokenizer_vocabulary": {"bytes": 256, "specials": len(SPECIALS), "total": ByteTokenizer.vocab_size},
    "model_setting_semantics_checked": True,
    "source_hashes": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [
        "tiny_perceptron/model.py", "tiny_perceptron/modern.py", "tiny_perceptron/attention.py",
        "tiny_perceptron/data.py", "course/chapters/03.md", "course/chapters/04.md"]},
    "result": "所有斷言通過；未進行訓練或品質比較。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
