import ast
import hashlib
import inspect
import json
import platform
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from tiny_perceptron.attention import attention_mask
from tiny_perceptron.data import IGNORE, SPECIALS, ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum, masked_loss
from tiny_perceptron.multimodal import VisionEncoder, expand_modalities


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(path, selector):
    raw = path.read_bytes()
    headings = list(re.finditer(rb"(?m)^## (\S+) [^\n]+", raw))
    for index, heading in enumerate(headings):
        if heading[1].decode() == selector:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
            return raw[heading.start():end]
    raise ValueError("找不到指定的小節")


torch.manual_seed(1019)
torch.set_num_threads(1)
tok = ByteTokenizer()
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/10.7.md"
current = section(ROOT / "course/chapters/10.md", "10.7")
assert current == frozen.read_bytes()
fences = re.findall(r"```python\n(.*?)```", current.decode(), re.S)
excerpt = ast.parse(fences[1]).body[0]
actual = ast.parse(inspect.getsource(expand_modalities)).body[0]
assert ast.dump(excerpt, include_attributes=False) == ast.dump(actual, include_attributes=False)

embedding = nn.Embedding(264, 8)
with torch.no_grad():
    embedding.weight.copy_(torch.arange(264, dtype=torch.float32)[:, None].expand(264, 8))
ids = torch.tensor([1, 5, 4, 73, 2])
labels = torch.tensor([IGNORE, IGNORE, IGNORE, 73, 2])
observed = {}

for count in (3, 5):
    features = torch.arange(1000, 1000 + count, dtype=torch.float32)[:, None].expand(count, 8).clone()
    x, y = expand_modalities(ids, labels, embedding, {5: features}, {5})
    expected_inputs = [1] + list(range(1000, 1000 + count)) + [4, 73]
    expected_labels = [IGNORE] * (count + 1) + [73, 2]
    assert x.shape == (1, count + 3, 8)
    assert torch.equal(x[0], torch.tensor(expected_inputs)[:, None].expand(-1, 8))
    assert y.tolist() == [expected_labels]
    assert y.dtype == torch.long
    assert x.device == embedding.weight.device == y.device
    observed[str(count)] = {
        "expanded_length": len(ids) - 1 + count,
        "x_shape": list(x.shape),
        "input_row_values": x[0, :, 0].tolist(),
        "y": y.tolist(),
        "target_dtype": str(y.dtype),
        "valid_targets": y[y != IGNORE].tolist(),
        "assistant_index": expected_inputs.index(4),
    }

features = torch.arange(1000, 1003, dtype=torch.float32)[:, None].expand(3, 8).clone()
x, y = expand_modalities(ids, labels, embedding, {5: features}, {5})
full_ids = torch.tensor([1, 3, 5, 2, 4, 73, 2])
full_labels = torch.tensor([IGNORE] * 5 + [73, 2])
full_x, full_y = expand_modalities(full_ids, full_labels, embedding, {5: features}, {5})
assert full_x.shape == (1, 8, 8)
assert full_y.tolist() == [[IGNORE] * 6 + [73, 2]]
assert full_x[0, 6, 0].item() == 4 and full_y[0, 6].item() == 73
observed["complete_dialogue"] = {
    "ids": full_ids.tolist(), "labels": full_labels.tolist(),
    "x_shape": list(full_x.shape), "input_row_values": full_x[0, :, 0].tolist(),
    "y": full_y.tolist(), "extra_ignored_targets": int((full_y == IGNORE).sum() - (y == IGNORE).sum()),
}

chat_x, chat_y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
assert chat_x.tolist() == [1, 3, 89, 2, 4, 73]
assert chat_y.tolist() == [IGNORE] * 4 + [73, 2]
wrong_ids = chat_x.clone()
wrong_ids[2] = tok.image_id
wrong_x, wrong_y = expand_modalities(wrong_ids, chat_y, embedding, {5: features}, {5})
wrong_assistant = (wrong_x[0, :, 0] == tok.assistant_id).nonzero()[0].item()
assert wrong_y[0, wrong_assistant].item() == tok.eos_id
observed["already_shifted_dialogue"] = {
    "render_chat_x": chat_x.tolist(), "render_chat_y": chat_y.tolist(),
    "expanded_input_row_values": wrong_x[0, :, 0].tolist(), "expanded_y": wrong_y.tolist(),
    "assistant_index": wrong_assistant, "assistant_target": wrong_y[0, wrong_assistant].item(),
}

for short_ids, short_labels in ((ids, labels[:-1]), (ids[:-1], labels)):
    try:
        expand_modalities(short_ids, short_labels, embedding, {5: features}, {5})
    except ValueError as error:
        observed.setdefault("strict_length_errors", []).append(str(error))
    else:
        raise AssertionError("長度不符時應丟出 ValueError")

ordinary_positions = []
handle = embedding.register_forward_pre_hook(lambda module, inputs: ordinary_positions.extend(inputs[0].tolist()))
expand_modalities(ids, labels, embedding, {5: features}, {5})
handle.remove()
assert ordinary_positions == [1, 4, 73, 2]
observed["embedding_lookup_ids"] = ordinary_positions

random_features = torch.randn(3, 8)
assert random_features.shape == (3, 8)
encoder_features = VisionEncoder(width=8)(torch.zeros(1, 3, 16, 16))
assert encoder_features.shape == (1, 16, 8)
observed["tokenizer_and_encoder"] = {
    "specials": list(enumerate(SPECIALS)), "vocab_size": tok.vocab_size,
    "A_utf8_bytes": list("A".encode()), "A_ids": tok.encode("A"),
    "embedding_weight_shape": list(embedding.weight.shape),
    "random_features_shape": list(random_features.shape),
    "encoder_shape": list(encoder_features.shape),
    "marker_set_type": type({5}).__name__, "feature_mapping_type": type({5: features}).__name__,
    "repeated_ignore": [IGNORE] * 3,
}
assert tok.encode("A") == [73] and tok.vocab_size == 264
assert list("A".encode()) == [65]

model = TinyLM(ModelConfig(width=8, heads=1, layers=1, max_length=16))
positions_seen, masks_seen = [], []
pos_hook = model.position.register_forward_pre_hook(lambda module, inputs: positions_seen.append(inputs[0].tolist()))

def capture_mask(*args, **kwargs):
    allowed = attention_mask(*args, **kwargs)
    masks_seen.append(allowed[0, 0].tolist())
    return allowed

with patch("tiny_perceptron.attention.attention_mask", capture_mask):
    result = model(embeddings=x)
pos_hook.remove()
assert positions_seen == [list(range(6))]
assert masks_seen == [[[j <= i for j in range(6)] for i in range(6)]]
assert result["logits"].shape == (1, 6, 264)
observed["language_core"] = {
    "position_ids": positions_seen[0], "allowed_mask": masks_seen[0],
    "logits_shape": list(result["logits"].shape),
    "assistant_can_read_image_positions": masks_seen[0][4][1:4],
}

scores = torch.randn(1, 6, 264, requires_grad=True)
loss, count = loss_sum(scores, y)
expected_loss = F.cross_entropy(scores[0, 4:6], torch.tensor([73, 2]), reduction="sum")
assert count.item() == 2 and torch.equal(loss, expected_loss)
loss.backward()
ignored_grad_max = scores.grad[0, :4].abs().max().item()
assert ignored_grad_max == 0
changed_scores = scores.detach().clone()
changed_scores[0, :4] = torch.randn_like(changed_scores[0, :4]) * 100
changed_loss = masked_loss(changed_scores, y)
assert torch.equal(changed_loss, expected_loss.detach() / 2)
observed["ignored_targets"] = {
    "effective_target_count": count.item(), "loss_sum": loss.item(),
    "independent_two_target_sum": expected_loss.item(),
    "ignored_logits_gradient_max": ignored_grad_max,
    "masked_mean_after_ignored_scores_changed": changed_loss.item(),
}

image_values = features.clone().requires_grad_()
grad_x, grad_y = expand_modalities(ids, labels, model.embedding, {5: image_values}, {5})
grad_loss = masked_loss(model(embeddings=grad_x)["logits"], grad_y)
grad_loss.backward()
assert image_values.grad.norm().item() > 0
observed["image_context"] = {
    "feature_gradient_norm": image_values.grad.norm().item(),
    "interpretation": "此配置的圖片輸入可透過後方有效答案取得梯度；不代表訓練品質。",
}

receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
figure_record = next(record for record in receipt["records"] if record["page_id"] == "10.7")
png_digest = digest(ROOT / figure_record["png"]["path"])
assert png_digest == figure_record["png"]["sha256"]
observed["figure_receipt"] = figure_record
observed["figure_png_actual_sha256"] = png_digest
observed["source_integrity"] = {
    "frozen_sha256": digest(frozen), "current_section_matches_frozen": True,
    "excerpt_matches_repository_ast": True,
    "freeze_sha256": digest(ROOT / "docs/course-review-20261019/freeze-04.json"),
    "figure_sha256": digest(ROOT / "course/figures/rewrite-10-image-targets.svg"),
}
observed["environment"] = {
    "python": sys.version, "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform(),
    "seed": "1019", "threads": "1",
}
observed["repository_sha256"] = {name: digest(ROOT / name) for name in (
    "tiny_perceptron/data.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py",
    "tiny_perceptron/attention.py",
)}
observed["execution"] = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "result": "所有斷言通過；未進行訓練或參數更新。",
}
print(json.dumps(observed, ensure_ascii=False, indent=2))
