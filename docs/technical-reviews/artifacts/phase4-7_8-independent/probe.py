"""Bounded CPU checks of section 7.8; no training, dataset or pretrained weights."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
torch.manual_seed(42)
assert not torch.cuda.is_available() and torch.version.cuda is None


def select(valid, logits):
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("Expected a Boolean [batch, sequence] mask")
    if valid.shape[1] == 0 or not valid.any(dim=1).all():
        raise ValueError("Every row must contain at least one true input position")
    positions = torch.arange(valid.shape[1])[None].expand_as(valid)
    last = positions.masked_fill(~valid, -1).amax(dim=-1)
    return last, logits[torch.arange(valid.shape[0]), last]


valid = torch.tensor([[True, True, False, False], [False, True, True, True]])
logits = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
original = logits.clone()
last, selected = select(valid, logits)
assert last.tolist() == [1, 3]
assert selected.tolist() == [[5, 6, 7, 8, 9], [35, 36, 37, 38, 39]]
assert selected.shape == (2, 5)
assert valid.sum(-1).sub(1).tolist() == [1, 2]
assert torch.equal(logits, original)
changed = valid.clone()
changed[0] = torch.tensor([False, False, True, True])
changed_last, changed_selected = select(changed, logits)
assert changed_last.tolist() == [3, 3]
assert changed_selected.tolist() == [[15, 16, 17, 18, 19], [35, 36, 37, 38, 39]]

all_masks = torch.tensor(list(itertools.product([False, True], repeat=4)), dtype=torch.bool)
nonempty = all_masks[all_masks.any(-1)]
all_logits = torch.arange(15 * 4 * 5, dtype=torch.float32).reshape(15, 4, 5)
all_last, all_selected = select(nonempty, all_logits)
python_last = [max(i for i, value in enumerate(row) if value) for row in nonempty.tolist()]
assert all_last.tolist() == python_last
for row, column in enumerate(python_last):
    assert all_selected[row].tolist() == [row * 20 + column * 5 + v for v in range(5)]
gather_index = all_last[:, None, None].expand(-1, 1, 5)
gathered = all_logits.gather(1, gather_index).squeeze(1)
assert gathered.shape == (15, 5) and torch.equal(all_selected, gathered)
holes = torch.tensor([[True, False, True, False]])
hole_last, hole_scores = select(holes, logits[:1])
assert hole_last.item() == 2 and holes.sum().item() - 1 == 1
assert hole_scores.tolist() == [[10, 11, 12, 13, 14]]

empty_mask = torch.zeros((1, 4), dtype=torch.bool)
raw_positions = torch.arange(4)[None].expand_as(empty_mask)
raw_empty_last = raw_positions.masked_fill(~empty_mask, -1).amax(-1)
assert raw_empty_last.item() == -1
assert torch.equal(logits[:1][torch.arange(1), raw_empty_last], logits[:1, -1])
try:
    logits[:1].gather(1, raw_empty_last[:, None, None].expand(-1, 1, 5))
except RuntimeError as error:
    gather_negative_error = str(error)
else:
    raise AssertionError("gather unexpectedly accepted negative sequence indices")
rejected = []
for label, mask, scores in (
    ("all_false", empty_mask, logits[:1]),
    ("zero_sequence_length", torch.zeros((1, 0), dtype=torch.bool), torch.zeros((1, 0, 5))),
):
    try:
        select(mask, scores)
    except ValueError as error:
        rejected.append({"case": label, "type": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("Expected empty-row rejection")

model = TinyLM(ModelConfig(vocab_size=10, width=8, max_length=16)).eval()
true_rows = [[1, 2], [3, 4, 5]]
layouts = {
    "right": [[1, 2, 0, 0], [3, 4, 5, 0]],
    "left": [[0, 0, 1, 2], [0, 3, 4, 5]],
    "holes": [[1, 0, 2, 0], [3, 0, 4, 5]],
}
model_checks = []
with torch.no_grad():
    base = torch.cat([model(torch.tensor([row]))["logits"][:, -1] for row in true_rows])
    for label, row_ids in layouts.items():
        ids = torch.tensor(row_ids)
        mask = ids != 0
        # Model position IDs count true content; last indices count physical columns.
        model_positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        output = model(ids, valid=mask, positions=model_positions)["logits"]
        indices, scores = select(mask, output)
        max_difference = (scores - base).abs().max().item()
        assert torch.allclose(scores, base, rtol=0, atol=1e-6)
        model_checks.append({"layout": label, "valid": mask.tolist(), "physical_last": indices.tolist(),
                             "model_position_ids": model_positions.tolist(), "logits_shape": list(output.shape),
                             "selected_shape": list(scores.shape), "base_max_abs_difference": max_difference})
    left_ids = torch.tensor(layouts["left"])
    left_valid = left_ids != 0
    left_positions = (left_valid.long().cumsum(-1) - 1).clamp_min(0)
    _, bad_valid_scores = select(left_valid, model(left_ids, positions=left_positions)["logits"])
    _, bad_positions_scores = select(left_valid, model(left_ids, valid=left_valid)["logits"])
    wrong_valid_difference = (bad_valid_scores - base).abs().max().item()
    wrong_positions_difference = (bad_positions_scores - base).abs().max().item()
    assert wrong_valid_difference > 1e-6 and wrong_positions_difference > 1e-6

results = {
    "environment": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_available": False,
                    "cuda_build": None, "threads": torch.get_num_threads()},
    "original": {"last": last.tolist(), "selected": selected.tolist(), "shape": list(selected.shape),
                 "effective_count_minus_one": valid.sum(-1).sub(1).tolist(), "logits_unchanged": True},
    "exercise": {"last": changed_last.tolist(), "selected": changed_selected.tolist()},
    "exhaustive_masks": {"total_binary_masks": 16, "nonempty_masks_checked": 15,
                         "all_last": all_last.tolist(), "python_oracle_equal": True,
                         "gather_index_shape": list(gather_index.shape), "gather_equal": True},
    "hole_case": {"last": hole_last.tolist(), "count_minus_one": 1, "selected": hole_scores.tolist()},
    "empty_case": {"raw_last": raw_empty_last.tolist(), "advanced_index_selects_final_column": True,
                   "gather_negative_error": gather_negative_error, "guard_rejections": rejected},
    "tiny_lm_synthetic_checks": model_checks,
    "incorrect_forward_inputs": {"wrong_valid_max_difference": wrong_valid_difference,
                                 "wrong_positions_max_difference": wrong_positions_difference},
    "scope": "Synthetic scores and one freshly initialized random TinyLM only; no backward/optimizer/training, no learned quality claim, no model or dataset download.",
    "repository_module_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                                  for name in ["tiny_perceptron/model.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py", "tiny_perceptron/data.py"]},
}
print(json.dumps(results, ensure_ascii=False, indent=2))
