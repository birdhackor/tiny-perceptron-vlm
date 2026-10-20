"""核對章首導覽所指的程式、資料流與參數安排，不訓練模型。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.attention import manual_attention
from tiny_perceptron.data import shifted
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.modern import DenseFFN


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = Path(__file__).parent


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def section(path, identifier):
    text = (ROOT / path).read_text()
    match = re.search(r"^## " + re.escape(identifier) + r" .*$", text, re.MULTILINE)
    assert match is not None, "找不到指定標題"
    following = re.search(r"^## ", text[match.end():], re.MULTILINE)
    end = match.end() + following.start() if following else len(text)
    return text[match.start():end]


torch.set_num_threads(1)
torch.manual_seed(42)
environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "device": "cpu",
    "platform": platform.platform(),
    "dtype": "torch.float32",
}
freeze_source = "docs/course-review-20261019/freeze-01/sources/chapter-04.md"
assert digest(freeze_source) == "54cfbe71cacf6517e3c9f85b4228fcce892569afeba8d0d3460fb72ac44de5de"
freeze_manifest = "docs/course-review-20261019/freeze-04.json"
assert digest(freeze_manifest) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
freeze = json.loads((ROOT / freeze_manifest).read_text())
page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-04")
assert page["snapshot"] == freeze_source and page["figures_sha256"] == {}
current_introduction = (ROOT / "course/chapters/04.md").read_text().split("## ", 1)[0]
assert current_introduction.encode() == (ROOT / freeze_source).read_bytes()

paths = [
    "course/chapters/03.md",
    "course/chapters/04.md",
    "course/chapters/14.md",
    "course/glossary.md",
    "tiny_perceptron/model.py",
    "tiny_perceptron/modern.py",
    "tiny_perceptron/attention.py",
    "tiny_perceptron/data.py",
]
later_chapters = []
later_code_calls = {}
for number in range(5, 24):
    path = f"course/chapters/{number:02d}.md"
    if not (ROOT / path).exists():
        continue
    calls = 0
    for code in re.findall(r"```python\n(.*?)```", (ROOT / path).read_text(), re.DOTALL):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        calls += sum(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TinyLM"
            for node in ast.walk(tree)
        )
    if calls:
        later_chapters.append(path)
        later_code_calls[path] = calls
        paths.append(path)
assert len(later_chapters) >= 2, "後續教材未找到多處模型使用"

navigation = {
    "ffn": "DenseFFN" in section("course/chapters/04.md", "4.4"),
    "block": "class Block" in section("course/chapters/04.md", "4.5"),
    "complete_model": "class TinyLM" in section("course/chapters/04.md", "4.6"),
    "answer_alignment": "shifted([1, 2, 3, 4])" in section("course/chapters/04.md", "4.7"),
    "gradient": "loss.backward()" in section("course/chapters/04.md", "4.7"),
    "depth_after_alignment": (ROOT / "course/chapters/04.md").read_text().index("## 4.7 ")
    < (ROOT / "course/chapters/04.md").read_text().index("## 4.8 "),
    "looped_block": "looped = nn.ModuleList([shared, shared, shared])"
    in section("course/chapters/14.md", "14.11"),
    "glossary": (ROOT / "course/glossary.md").is_file(),
}
assert all(navigation.values()), "導覽承諾與教材位置不一致"
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
inspection = {
    "environment": environment,
    "git_head": revision,
    "file_sha256": {path: digest(path) for path in paths},
    "snapshot_sha256": digest(freeze_source),
    "freeze_manifest_sha256": digest(freeze_manifest),
    "introduction_matches_frozen_source": True,
    "figures": page["figures_sha256"],
    "navigation_checks": navigation,
    "later_chapters_using_TinyLM": later_chapters,
    "TinyLM_constructor_calls_in_python_blocks": later_code_calls,
}
(ARTIFACTS / "repository-inspection.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")

# 用同一批 key/value，改變 query，以檢查內容會改變讀取比例。
k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
allowed = torch.ones(1, 1, 1, 2, dtype=torch.bool)
first, first_weights = manual_attention(k[:, :, :1], k, v, allowed)
second, second_weights = manual_attention(k[:, :, 1:], k, v, allowed)
assert not torch.allclose(first_weights, second_weights), "改查詢後讀取比例未改變"
assert not torch.allclose(first, second), "改查詢後取回內容未改變"

config = ModelConfig(vocab_size=5, width=8, layers=2)
model = TinyLM(config)
assert config.experts == 0 and all(isinstance(block.ffn, DenseFFN) for block in model.blocks)
parameter_sets = [{parameter.data_ptr() for parameter in block.parameters()} for block in model.blocks]
assert parameter_sets[0].isdisjoint(parameter_sets[1]), "不同 block 意外共用參數儲存"
ids, labels = shifted([1, 2, 3, 4])
ids, labels = ids[None], labels[None]
entered = []
ffn_shapes = []
handles = [model.blocks[0].register_forward_pre_hook(lambda module, args: entered.append(args[0].detach().clone()))]
for block in model.blocks:
    handles.append(block.ffn.up.register_forward_hook(lambda module, args, output: ffn_shapes.append(list(output.shape))))
result = model(ids)
for handle in handles:
    handle.remove()
expected_embedding = model.embedding(ids) + model.position(torch.arange(ids.shape[1]))
torch.testing.assert_close(entered[0], expected_embedding)
assert ffn_shapes == [[1, 3, 32], [1, 3, 32]], "Dense FFN 沒有處理所有位置與完整中間寬度"
assert result["logits"].shape == (1, 3, 5), "輸出未給每個位置完整候選分數"
assert torch.isfinite(result["logits"]).all(), "候選分數包含非有限值"

ffn = model.blocks[0].ffn
features = torch.randn(1, 3, 8)
whole = ffn(features)
separate = torch.cat([ffn(features[:, position:position + 1]) for position in range(3)], dim=1)
torch.testing.assert_close(whole, separate, atol=1e-6, rtol=1e-5)
perturbed = features.clone()
perturbed[:, -1, 0] += 1
torch.testing.assert_close(ffn(perturbed)[:, :-1], whole[:, :-1], atol=0, rtol=0)

changed = ids.clone()
changed[:, -1] = 4
with torch.no_grad():
    changed_logits = model(changed)["logits"]
prefix_delta = (changed_logits[:, :-1] - result["logits"][:, :-1]).abs().max().item()
torch.testing.assert_close(changed_logits[:, :-1], result["logits"][:, :-1], atol=1e-6, rtol=1e-5)
assert not torch.allclose(changed_logits[:, -1], result["logits"][:, -1]), "更改末字後未觀察到末位輸出改變"

before = {name: value.detach().clone() for name, value in model.named_parameters()}
loss = masked_loss(result["logits"], labels)
loss.backward()
assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())

# 直接執行教材中比較堆疊與循環的短程式；沒有 optimizer 或訓練迴圈。
loop_source = re.findall(r"```python\n(.*?)```", section("course/chapters/14.md", "14.11"), re.DOTALL)[0]
loop_stdout = io.StringIO()
with contextlib.redirect_stdout(loop_stdout):
    loop_namespace = {}
    exec(compile(loop_source, "course/chapters/14.md", "exec"), loop_namespace)
stacked, looped = loop_namespace["stacked"], loop_namespace["looped"]
assert looped[0] is looped[1] is looped[2]
stacked_count = sum(parameter.numel() for parameter in stacked.parameters())
looped_count = sum(parameter.numel() for parameter in looped.parameters())
assert stacked_count == 3 * looped_count

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --offline python docs/technical-reviews/artifacts/review-20261019-chapter-04/verify_model.py",
    "environment": environment,
    "source_checks": {
        "path": (ARTIFACTS / "repository-inspection.json").relative_to(ROOT).as_posix(),
        "sha256": digest((ARTIFACTS / "repository-inspection.json").relative_to(ROOT)),
    },
    "content_based_attention": {"first_weights": first_weights.tolist(), "second_weights": second_weights.tolist()},
    "model_configuration": model.description()["config"],
    "inputs": ids.tolist(),
    "labels": labels.tolist(),
    "logits_shape": list(result["logits"].shape),
    "ffn_intermediate_shapes": ffn_shapes,
    "distinct_block_parameter_storage": True,
    "embedding_plus_position_verified": True,
    "pointwise_ffn_verified": True,
    "future_token_prefix_max_abs_difference": prefix_delta,
    "loss": loss.item(),
    "all_parameter_gradients_finite": True,
    "weights_unchanged_after_backward": True,
    "looped_example_stdout": loop_stdout.getvalue(),
    "shared_block_identity_verified": True,
    "stacked_parameters": stacked_count,
    "looped_parameters": looped_count,
    "result": "全部核對通過；沒有參數更新。",
}, ensure_ascii=False, indent=2))
