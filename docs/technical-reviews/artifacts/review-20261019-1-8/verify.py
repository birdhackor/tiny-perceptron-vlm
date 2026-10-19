"""獨立核對負對數代價、單題與多題交叉熵，以及輸入分數的 API 行為。"""

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def main():
    torch.set_default_device("cpu")
    logits = torch.tensor([[0.0, 2.0, 0.0]])
    target = torch.tensor([1])
    probs = logits.softmax(dim=-1)
    expected_probs = [1 / (math.exp(2) + 2), math.exp(2) / (math.exp(2) + 2), 1 / (math.exp(2) + 2)]
    expected_loss = math.log(math.exp(2) + 2) - 2
    expected_cat_loss = math.log(math.exp(2) + 2)
    cat_target = torch.tensor([0])
    loss = F.cross_entropy(logits, target)
    manual = -probs[0, 1].log()
    cat_loss = F.cross_entropy(logits, cat_target)
    cat_manual = -probs[0, 0].log()
    probabilities_as_scores = F.cross_entropy(probs, target)
    repeated_softmax_manual = -probs.softmax(dim=-1)[0, 1].log()
    batch_logits = logits.repeat(2, 1)
    batch_targets = torch.tensor([1, 0])
    batch_losses = F.cross_entropy(batch_logits, batch_targets, reduction="none")
    batch_mean = F.cross_entropy(batch_logits, batch_targets)
    large_logits = torch.tensor([[10000.0, 10002.0, 10000.0]])
    stable_shifted = F.cross_entropy(large_logits, target)
    tiny_target = torch.tensor([[1000.0, -1000.0, 0.0]])
    extreme_loss = F.cross_entropy(tiny_target, target)
    extreme_naive = -tiny_target.softmax(dim=-1)[0, 1].log()
    p_values = [0.1, 0.5, 0.9]
    neglogs = [-math.log(p) for p in p_values]
    torch_neglogs = (-torch.tensor(p_values).log()).tolist()
    predictions = torch.tensor([[0.25, 0.5, 0.25], [0.05, 0.9, 0.05]])
    confidence_losses = (-predictions[:, 1].log()).tolist()
    bit_cost = expected_loss / math.log(2)
    bit_cost_direct = -math.log2(expected_probs[1])
    tokenization_example = {
        "same_text_split_a": {"targets": 10, "mean_nll": 0.6, "total_nll": 10 * 0.6},
        "same_text_split_b": {"targets": 5, "mean_nll": 1.0, "total_nll": 5 * 1.0},
    }
    assert F is torch.nn.functional
    assert list(logits.shape) == [1, 3] and list(target.shape) == [1]
    assert target.dtype == torch.int64 and logits.dtype == torch.float32
    assert torch.equal(probs, logits.softmax(dim=1))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
    assert torch.allclose(probs[0], torch.tensor(expected_probs), atol=1e-7, rtol=0)
    assert torch.allclose(loss, manual)
    assert abs(loss.item() - expected_loss) < 1e-7
    assert abs(cat_loss.item() - expected_cat_loss) < 2e-7
    assert torch.allclose(cat_loss, cat_manual)
    assert abs(cat_loss.item() - loss.item() - 2) < 2e-7
    assert torch.allclose(probabilities_as_scores, repeated_softmax_manual)
    assert abs(probabilities_as_scores.item() - loss.item()) > 0.1
    assert torch.allclose(batch_mean, batch_losses.mean())
    assert torch.allclose(loss, stable_shifted)
    assert extreme_loss.item() == 2000 and torch.isinf(extreme_naive)
    assert all(abs(a - b) < 1e-7 for a, b in zip(neglogs, torch_neglogs))
    assert predictions.argmax(dim=-1).tolist() == [1, 1]
    assert confidence_losses[1] < confidence_losses[0]
    assert abs(bit_cost - bit_cost_direct) < 1e-14
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-8/verify.py",
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "platform": platform.platform(),
            "device": str(logits.device),
            "logits_dtype": str(logits.dtype),
            "target_dtype": str(target.dtype),
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "p_values": p_values,
        "negative_natural_logs_float64": neglogs,
        "negative_natural_logs_torch_float32": torch_neglogs,
        "same_top1": {"argmax_ids": predictions.argmax(dim=-1).tolist(), "losses": confidence_losses},
        "shapes": {"logits": list(logits.shape), "target": list(target.shape)},
        "functional_alias_identity": F is torch.nn.functional,
        "probabilities_float32": probs.tolist(),
        "probabilities_float64_formula": expected_probs,
        "sum_along_last_axis": probs.sum(dim=-1).tolist(),
        "negative_axis_matches_positive": torch.equal(probs, logits.softmax(dim=1)),
        "target_1": {"manual": manual.item(), "cross_entropy": loss.item(), "float64_formula": expected_loss, "allclose_default": bool(torch.allclose(loss, manual))},
        "target_0": {"manual": cat_manual.item(), "cross_entropy": cat_loss.item(), "float64_formula": expected_cat_loss},
        "probabilities_misused_as_logits": {"cross_entropy": probabilities_as_scores.item(), "second_softmax_manual": repeated_softmax_manual.item()},
        "mean_reduction": {"individual_losses": batch_losses.tolist(), "default_mean": batch_mean.item(), "manual_mean": batch_losses.mean().item(), "effective_targets": 2},
        "stability": {"large_common_shift": stable_shifted.item(), "extreme_wrong_target_cross_entropy": extreme_loss.item(), "extreme_separate_softmax_log_is_infinite": bool(torch.isinf(extreme_naive))},
        "units": {"nats": expected_loss, "nats_div_ln2": bit_cost, "negative_log2_probability": bit_cost_direct},
        "tokenization_denominator_example": tokenization_example,
        "result": "所有斷言通過；只核對確定性算例與本環境的 API 行為。",
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2, allow_nan=False))
