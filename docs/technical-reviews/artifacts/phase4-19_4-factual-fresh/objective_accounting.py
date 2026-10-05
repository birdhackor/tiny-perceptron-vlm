"""Short CPU arithmetic for preference scoring and routing balance; no training."""
from pathlib import Path
import json
import math
import os
import sys
ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
import torch
from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.capstone import TOK, build_dataset, preference_pairs, prepare_batch
from tiny_perceptron.data import IGNORE
torch.set_num_threads(1)
assert torch.version.cuda is None
rows, _ = build_dataset()
row = next(row for row in rows["train"] if row["task"] == "style")
pair = preference_pairs([row])[0]
counts = {}
for field in ("chosen", "rejected"):
    batch, labels = prepare_batch([row], answers=[pair[field]])
    count = int((labels != IGNORE).sum())
    counts[field] = count
    logits = torch.zeros(*labels.shape, TOK.vocab_size)
    observed = sequence_log_probability(logits, labels).item()
    expected = -count * math.log(TOK.vocab_size)
    assert math.isclose(observed, expected, abs_tol=3e-5)
anchor_batch, anchor_labels = prepare_batch([row])
counts["supervised_replay_counted_targets"] = int((anchor_labels != IGNORE).sum())
assert len("，祝你愉快！".encode("utf-8")) == 18
assert counts == {"chosen": 10, "rejected": 28, "supervised_replay_counted_targets": 10}
logp = torch.tensor([-7.0, -9.0])
logr = torch.tensor([-8.0, -10.0])
reference_chosen = torch.tensor([-9.0, -8.0])
reference_rejected = torch.tensor([-10.0, -9.0])
observed = dpo_loss(logp, logr, reference_chosen, reference_rejected).item()
assert math.isclose(observed, math.log(2), abs_tol=1e-7)
n = 4
f_uniform = torch.ones(n) / n
p_uniform = torch.ones(n) / n
balanced = n * (f_uniform * p_uniform).sum().item()
concentrated = n * (torch.tensor([1., 0., 0., 0.]) * torch.tensor([1., 0., 0., 0.])).sum().item()
assert balanced == 1.0 and concentrated == 4.0
print(json.dumps({"preference_scored_target_lengths": counts, "dpo_formula_equal_policy_reference_margins_loss": observed, "four_expert_balance_surrogate": {"uniform": balanced, "fully_concentrated": concentrated, "coefficient": 0.01}, "interpretation": "The original step counter increments only supervised labels; DPO additionally scores chosen/rejected under both policy and reference. The surrogate encourages balance; this arithmetic does not measure training speed or guarantee balanced routing."}, indent=2))
