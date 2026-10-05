"""Small CPU calculations for 13.2: no model training or existing-model evaluation."""
import hashlib
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch
from tiny_perceptron.alignment import sequence_log_probability

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")
tok = ByteTokenizer()

def chat(answer):
    return render_chat([{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": answer}])

def oracle(answer):
    # Independent construction from the UTF-8 bytes and role/boundary IDs.
    stream = [1, 3] + [value + 8 for value in b"1+1=?"] + [2, 4] + [value + 8 for value in answer.encode("utf-8")] + [2]
    return stream[:-1], [-100] * 8 + [value + 8 for value in answer.encode("utf-8")] + [2]

pairs = {}
for answer in ["2", "3", "4", "300", "四"]:
    x, y = chat(answer)
    expected_x, expected_y = oracle(answer)
    assert x.tolist() == expected_x and y.tolist() == expected_y
    assert len(x) == len(y)
    valid = torch.where(y != -100)[0].tolist()
    assert valid[0] == 8 and x[8].item() == 4 and y[-1].item() == 2
    for index in valid[:-1]:
        assert y[index].item() == x[index + 1].item()
    pairs[answer] = {"x": x.tolist(), "y": y.tolist(), "valid_indices": valid}

assert pairs["4"]["x"][-1] == 60
assert pairs["2"] == {"x": [1, 3, 57, 51, 57, 69, 71, 2, 4, 58],
                      "y": [-100] * 8 + [58, 2], "valid_indices": [8, 9]}
batch_x, batch_y, attention = pad_batch([chat("2"), chat("300")])
assert batch_x.shape == batch_y.shape == attention.shape == (2, 12)
assert batch_y[0, 10:].tolist() == [-100, -100]
assert attention[0].tolist() == [True] * 10 + [False, False]

def conditional_logits(inputs):
    # A declared probability table, not trained weights: p(3|assistant)=0.3;
    # p(EOS|2)=0.2 and p(EOS|3)=0.8. It exhibits dependence on the candidate prefix.
    result = torch.zeros((1, len(inputs), 264), dtype=torch.float64)
    for i, previous in enumerate(inputs.tolist()):
        if previous == 4:
            result[0, i, :] = -1000.0
            result[0, i, 58] = math.log(0.7)
            result[0, i, 59] = math.log(0.3)
        elif previous in (58, 59):
            probability = 0.2 if previous == 58 else 0.8
            result[0, i, :] = -1000.0
            result[0, i, 2] = math.log(probability)
            result[0, i, 0] = math.log(1 - probability)
    return result

chosen_x, chosen_y = chat("2")
rejected_x, rejected_y = chat("3")
rejected_score = sequence_log_probability(conditional_logits(rejected_x), rejected_y.unsqueeze(0)).item()
crossed_score = sequence_log_probability(conditional_logits(chosen_x), rejected_y.unsqueeze(0)).item()
assert abs(rejected_score - math.log(0.3 * 0.8)) < 1e-12
assert abs(crossed_score - math.log(0.3 * 0.2)) < 1e-12
assert abs((rejected_score - crossed_score) - math.log(4)) < 1e-12
ignored_changed = conditional_logits(rejected_x)
ignored_changed[:, :8, :] = torch.arange(264, dtype=torch.float64) * 30
assert sequence_log_probability(ignored_changed, rejected_y.unsqueeze(0)).item() == rejected_score
# An extra label shift assigns EOS to the assistant position: explicit counterexample.
extra_shift_labels = rejected_y[1:]
assert extra_shift_labels[8].item() == 2 and rejected_y[8].item() == 59

environment = {"python": sys.version, "python_executable": sys.executable,
               "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
               "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
               "device": "cpu", "num_threads": torch.get_num_threads()}
(BASE / "cpu-checks-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
result = {"pairs": pairs, "padding_shapes": list(batch_x.shape),
          "p_rejected_given_own_prefix": 0.3 * 0.8, "p_rejected_with_crossed_prefix": 0.3 * 0.2,
          "own_prefix_log_probability": rejected_score, "crossed_prefix_log_probability": crossed_score,
          "difference": rejected_score - crossed_score, "ignored_positions_invariant": True,
          "extra_shift_wrong_assistant_target": extra_shift_labels[8].item(),
          "scope": "Byte IDs, shift/mask contract and declared conditional table only; no fitted model or performance claim."}
(BASE / "cpu-checks-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
