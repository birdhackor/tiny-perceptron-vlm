"""4.8 獨立技術核對；只做 CPU 結構／前向，沒有訓練更新。"""

from contextlib import redirect_stdout
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from tiny_perceptron.data import ByteTokenizer, SPECIALS
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


freeze_path = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads(freeze_path.read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "4.8")
snapshot = ROOT / page["snapshot"]
body = snapshot.read_text()
assert sha(snapshot) == page["source_sha256"]
current = (ROOT / page["source"]).read_text()
part = re.search(r"^## 4\.8 .*?(?=^## |\Z)", current, re.M | re.S).group()
assert part == body
code = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(code) == 1

torch.set_num_threads(1)
out = {
    "executed_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_4_8",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": "cpu",
        "parameter_dtype": "torch.float32",
        "input_dtype": "torch.int64",
        "threads": str(torch.get_num_threads()),
    },
    "source_snapshot": {"path": page["snapshot"], "sha256": sha(snapshot)},
    "inline_python_sha256": hashlib.sha256(code[0].encode()).hexdigest(),
    "training_updates": 0,
}

# 原樣執行本節內嵌版本，然後只改練習指定的 width=8。
for name, snippet in [("inline_exact", code[0]), ("exercise_width16", code[0].replace("width=8", "width=16"))]:
    stdout = io.StringIO()
    namespace = {}
    with redirect_stdout(stdout):
        exec(compile(snippet, str(snapshot) + ":" + name, "exec"), namespace)
    out[name] = stdout.getvalue().splitlines()

rows = []
for width in [8, 16]:
    for layers in [1, 2, 3]:
        model = TinyLM(ModelConfig(vocab_size=20, width=width, layers=layers))
        logits = model(torch.tensor([[1, 2]]))["logits"]
        fixed = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("blocks."))
        per_block = [sum(p.numel() for p in block.parameters()) for block in model.blocks]
        predicted_fixed = 2 * 20 * width + 128 * width + 2 * width
        predicted_block = 12 * width**2 + 9 * width
        actual = model.description()["parameters"]
        assert fixed == predicted_fixed and per_block == [predicted_block] * layers
        assert actual == fixed + sum(per_block)
        assert logits.shape == (1, 2, 20) and torch.isfinite(logits).all()
        rows.append({"width": width, "layers": layers, "fixed": fixed, "per_block": per_block,
                     "total": actual, "logits_shape": list(logits.shape), "finite": True})
out["count_rows"] = rows

model = TinyLM(ModelConfig(vocab_size=20, width=8, layers=3))
out["width8_block0_parameters"] = [
    {"name": name, "shape": list(p.shape), "numel": p.numel()}
    for name, p in model.blocks[0].named_parameters()
]
out["width8_fixed_parameters"] = [
    {"name": name, "shape": list(p.shape), "numel": p.numel()}
    for name, p in model.named_parameters() if not name.startswith("blocks.")
]
assert len({p.data_ptr() for b in model.blocks for p in b.parameters()}) == sum(
    len(list(b.parameters())) for b in model.blocks
)
assert model.output.weight.data_ptr() != model.embedding.weight.data_ptr()
assert model.output.bias is None
for block in model.blocks:
    assert all(getattr(block.attention, n).bias is None for n in ["q", "k", "v", "out"])
    assert block.ffn.up.bias is not None and block.ffn.down.bias is not None
    assert block.ffn.gate is None and block.ffn.activation == "gelu"
assert sum(p.numel() for p in torch.nn.GELU().parameters()) == 0

inputs, outputs, handles = [], [], []
for block in model.blocks:
    handles.append(block.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().clone())))
    handles.append(block.register_forward_hook(lambda _, args, result: outputs.append(result[0].detach().clone())))
with torch.no_grad():
    result = model(torch.tensor([[1, 2]]))["logits"]
for handle in handles:
    handle.remove()
assert all(torch.equal(inputs[i + 1], outputs[i]) for i in [0, 1])
assert all(not torch.equal(inputs[i], outputs[i]) for i in range(3))
out["serial_stack"] = {"blocks": len(model.blocks), "next_input_equals_previous_output": True,
                        "each_block_changed_representation": True, "block_parameters_independent": True}

grid = []
for width in [4, 8, 16, 64]:
    for vocab in [20, 264]:
        for layers in [1, 2, 3]:
            config = ModelConfig(vocab_size=vocab, width=width, layers=layers)
            actual = TinyLM(config).description()["parameters"]
            expected = 2 * vocab * width + config.max_length * width + 2 * width + layers * (12 * width**2 + 9 * width)
            assert actual == expected
            grid.append((width, vocab, layers, actual))
out["formula_grid"] = {"checked_configurations": len(grid), "all_exact": True}

foundation = TinyLM(ModelConfig(width=64, layers=2))
assert foundation.description()["parameters"] == 141568
out["foundation_structure"] = {"config": foundation.description()["config"], "fixed": 42112,
                               "per_block": 49728, "total": foundation.description()["parameters"]}
out["arithmetic"] = {"square_matrix_width_ratio": 16 * 16 / (8 * 8),
                      "fixed_width_ratio": 2720 / 1360, "block_width_ratio": 3216 / 840}
assert len(SPECIALS) == 8 and ByteTokenizer.vocab_size == 256 + len(SPECIALS)
out["byte_tokenizer"] = {"special_tokens": list(SPECIALS), "byte_values": 256,
                         "vocab_size": ByteTokenizer.vocab_size, "中文_utf8_bytes": list("中文".encode()),
                         "中文_ids": ByteTokenizer().encode("中文")}

record_path = ROOT / "docs/course-experiments/results/text_foundation.json"
record = json.loads(record_path.read_text())
training = record["results"]["training"]
assert training["parameters"] == 141568
out["historical_record"] = {
    "path": str(record_path.relative_to(ROOT)), "sha256": sha(record_path),
    "revision": record["revision"], "device": record["device"], "seed": record["seed"],
    "training_fields": {k: training[k] for k in ["parameters", "steps", "records", "effective_tokens", "checkpoint"]},
    "has_same_width64_depth1_comparator": False,
    "scaling_experiment_names": list(record["results"]["scaling"]),
    "scope": "只核作者保存的歷史紀錄與目前可重算的結構；此腳本未訓練、未重測歷史品質。",
}
out["repository_inputs"] = {p: sha(ROOT / p) for p in [
    "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
    "tiny_perceptron/data.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
]}
out["result"] = "全部結構／數字／內嵌程式斷言通過；無訓練或品質比較。"
print(json.dumps(out, ensure_ascii=False, indent=2))
