"""核對左側填充、位置索引、全遮罩列與忽略標籤的行為。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.attention import attention_mask, manual_attention
from tiny_perceptron.data import ByteTokenizer, CharTokenizer, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss

torch.set_num_threads(1)
torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=10, width=8))
ids = torch.tensor([[0, 0, 1, 2, 3]])
valid = ids != 0
positions = torch.tensor([[0, 0, 0, 1, 2]])
base = model(torch.tensor([[1, 2, 3]]))["logits"]
actual = model(ids, valid=valid, positions=positions)["logits"][:, 2:]
without_valid = model(ids, positions=positions)["logits"][:, 2:]
without_positions = model(ids, valid=valid)["logits"][:, 2:]

def difference(value):
    return (base - value).abs().max().item()

result = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "dtype": str(base.dtype),
        "platform": platform.platform(),
        "seed": "42",
        "threads": str(torch.get_num_threads()),
    },
    "configuration": model.description(),
    "ids": ids.tolist(),
    "valid": valid.tolist(),
    "positions": positions.tolist(),
    "shapes": {"base": list(base.shape), "actual": list(actual.shape)},
    "differences": {
        "masked_and_reindexed": difference(actual),
        "without_valid": difference(without_valid),
        "without_positions": difference(without_positions),
    },
    "allclose": {
        "masked_and_reindexed": torch.allclose(base, actual, atol=1e-6, rtol=0.0),
        "without_valid": torch.allclose(base, without_valid, atol=1e-6, rtol=0.0),
        "without_positions": torch.allclose(base, without_positions, atol=1e-6, rtol=0.0),
    },
}
assert result["allclose"] == {
    "masked_and_reindexed": True,
    "without_valid": False,
    "without_positions": False,
}

allowed = attention_mask(torch.arange(5), torch.arange(5), valid)
q = torch.randn(1, 1, 5, 8)
k = torch.randn_like(q)
v = torch.randn_like(q)
attended, weights = manual_attention(q, k, v, allowed)
raw = torch.full((1, 2), -torch.inf).softmax(-1)
result["attention"] = {
    "allowed": allowed.to(torch.int).tolist(),
    "empty_rows": (~allowed.any(-1)).tolist(),
    "all_negative_infinity_softmax_is_nan": bool(torch.isnan(raw).all()),
    "empty_output_is_zero": bool((attended[:, :, :2] == 0).all()),
    "empty_weights_are_zero": bool((weights[:, :, :2] == 0).all()),
    "all_output_is_finite": bool(attended.isfinite().all()),
}
assert result["attention"]["all_negative_infinity_softmax_is_nan"]
assert result["attention"]["empty_output_is_zero"]
assert result["attention"]["empty_weights_are_zero"]
assert result["attention"]["all_output_is_finite"]

valid_variants = []
for first, second in [(0, 0), (1, 2), (127, 127)]:
    trial = positions.clone()
    trial[0, :2] = torch.tensor([first, second])
    value = model(ids, valid=valid, positions=trial)["logits"][:, 2:]
    valid_variants.append({"pad_positions": [first, second], "max_difference": difference(value)})
    assert torch.allclose(base, value, atol=1e-6, rtol=0.0)
result["legal_pad_positions"] = valid_variants
result["invalid_pad_positions"] = []
for value in [-100, -1, 128]:
    trial = positions.clone()
    trial[0, :2] = value
    try:
        model(ids, valid=valid, positions=trial)
    except IndexError as error:
        result["invalid_pad_positions"].append({
            "pad_position": value,
            "exception": type(error).__name__,
            "message": str(error),
        })
    else:
        raise AssertionError("越界的位置索引應使位置表查詢失敗")

examples = [
    (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
    (torch.tensor([0, 5]), torch.tensor([5, 6])),
]
batch_ids, labels, batch_valid = pad_batch(examples)
logits = torch.arange(60, dtype=torch.float32).reshape(2, 3, 10) / 10
logits.requires_grad_()
loss = masked_loss(logits, labels)
selected_loss = F.cross_entropy(logits[labels != -100], labels[labels != -100])
loss.backward()
result["padding_and_loss"] = {
    "ids": batch_ids.tolist(),
    "labels": labels.tolist(),
    "valid": batch_valid.tolist(),
    "valid_from_ids_equal_returned_valid": bool(torch.equal(batch_ids != 0, batch_valid)),
    "effective_targets": int((labels != -100).sum()),
    "masked_mean": loss.item(),
    "selected_mean": selected_loss.item(),
    "ignored_logits_gradient_is_zero": bool((logits.grad[labels == -100] == 0).all()),
    "byte_pad_id": ByteTokenizer.pad_id,
    "char_unknown_token_id": CharTokenizer("甲").encode("乙"),
}
assert torch.allclose(loss, selected_loss, atol=1e-6, rtol=0.0)
assert result["padding_and_loss"]["ignored_logits_gradient_is_zero"]
assert not result["padding_and_loss"]["valid_from_ids_equal_returned_valid"]

result["repository_files"] = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/data.py"]
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
