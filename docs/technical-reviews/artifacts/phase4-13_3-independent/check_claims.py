"""Independent bounded CPU calculations; no model loading, training, or downloading."""
import ast
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.data import render_chat, pad_batch

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None

def original_functions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

p = math.exp(2) / (math.exp(2) + 2)
single = math.log(p)
logits = torch.tensor([[[0., 2., 0.], [2., 0., 0.], [0., 0., 2.]]])
labels = torch.tensor([[-100, 0, 2]])
score = sequence_log_probability(logits, labels)
assert abs(score.item() - 2 * single) < 2e-7
assert abs(score.exp().item() - p**2) < 2e-7
assert tuple(score.shape) == (1,)
exercise = labels.clone(); exercise[0, 0] = 1
three = sequence_log_probability(logits, exercise)
assert abs(three.item() - 3 * single) < 3e-7
changed = logits.clone(); changed[0, 0] = torch.tensor([100., -100., 0.])
assert torch.equal(sequence_log_probability(changed, labels), score)
assert torch.allclose(sequence_log_probability(logits + 10, labels), score)
batch_logits = logits.repeat(2, 1, 1)
batch_labels = torch.cat((labels, exercise))
batch = sequence_log_probability(batch_logits, batch_labels)
assert torch.allclose(batch, torch.tensor([2 * single, 3 * single]))
zero_valid = None
try:
    sequence_log_probability(logits, torch.full_like(labels, -100))
except ValueError as e:
    zero_valid = str(e)
assert zero_valid is not None
answers = {}
for answer in ["4", "10"]:
    x, y = render_chat([{"role": "user", "content": "2+2=?"},
                        {"role": "assistant", "content": answer}])
    target = y[y != -100].tolist()
    assert target[-1] == 2
    assert len(target) == len(answer.encode("utf-8")) + 1
    answers[answer] = {"input": x.tolist(), "labels": y.tolist(),
                       "scored_target_ids": target, "target_count": len(target)}
x, y, mask = pad_batch([render_chat([{"role": "user", "content": "2+2=?"},
                                    {"role": "assistant", "content": a}]) for a in ["4", "10"]])
assert y[0, -1].item() == -100 and not mask[0, -1].item()
assert (y != -100).sum(-1).tolist() == [2, 3]

measurement_path = OUT / "inputs/dpo-measurement.json"
measurement = json.loads(measurement_path.read_text())
ns = {"random": random, "json": json, "hashlib": hashlib, "render_chat": render_chat}
original_functions(OUT / "inputs/text.py", ["arithmetic_records"], ns)
original_functions(OUT / "inputs/common.py", ["split_records"], ns)
original_functions(OUT / "inputs/behavior-measurement-revision.py",
                   ["_preference_parts", "_pair_examples"], ns)
parts = ns["split_records"](ns["arithmetic_records"](), seed=measurement["seed"])
pairs = ns["_preference_parts"](parts)
splits = {}
for name, records in pairs.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
    digest = hashlib.sha256(raw).hexdigest()
    expected = measurement["results"]["data"][name]
    assert digest == expected["sha256"]
    assert len(records) == expected["records"]
    splits[name] = {"records": len(records), "family_count": len({r["family"] for r in records}),
                    "jsonl_sha256": digest}
examples = ns["_pair_examples"](pairs["train"], 128)
replays = {}
for name in ["model", "beta1"]:
    training = measurement["results"]["runs"][name]["training"]
    steps = training["steps"]
    assert steps == 250
    assert training["history"][-1]["step"] == 250
    sampler = random.Random(measurement["seed"])
    totals = [0, 0]
    selected_indices = set()
    for step in range(steps):
        indices = sampler.choices(range(len(examples)), k=8)
        selected_indices.update(indices)
        for index in indices:
            for side in [0, 1]:
                totals[side] += int((examples[index][side][1] != -100).sum())
    assert sum(totals) == training["effective_answer_tokens_both_sides"] == 9448
    replays[name] = {"beta": measurement["results"]["runs"][name]["beta"],
                     "steps": steps, "pairs_per_step": 8, "pair_exposures": steps * 8,
                     "answer_exposures_both_sides": steps * 8 * 2,
                     "unique_pair_indices_seen": len(selected_indices),
                     "chosen_target_tokens": totals[0], "rejected_target_tokens": totals[1],
                     "both_sides_target_tokens": sum(totals), "included_eos_tokens": steps * 8 * 2,
                     "saved_measurement_both_sides_tokens": training["effective_answer_tokens_both_sides"]}
result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "cuda_build": str(torch.version.cuda)},
    "numeric": {"conditional_0_8_product": 0.8 * 0.8,
                "softmax_denominator": math.exp(2) + 2,
                "single_target_probability": p, "single_natural_log_nats": single,
                "two_target_log_sum_nats": 2 * single, "two_target_probability": p**2,
                "three_target_log_sum_nats": 3 * single, "three_target_mean_nats_per_token": single,
                "observed_original_score_float32": score.item(),
                "observed_original_exp_float32": score.exp().item(),
                "observed_exercise_score_float32": three.item(), "batch_scores": batch.tolist()},
    "mask_contract": {"changed_ignored_position_has_no_effect": True,
                      "per_position_constant_logit_shift_has_no_effect": True,
                      "all_ignored_error": zero_valid, "padded_target_counts": [2, 3]},
    "render_chat_examples": answers,
    "measurement_provenance": {"source_json_sha256": hashlib.sha256(measurement_path.read_bytes()).hexdigest(),
                               "original_revision": measurement["revision"],
                               "original_device": measurement["device"],
                               "inspected_json_pointers": ["/seed", "/device", "/revision", "/step_scale",
                                   "/code_sha256/scripts~1course_experiments~1behavior.py",
                                   "/code_sha256/tiny_perceptron~1alignment.py", "/code_sha256/tiny_perceptron~1data.py",
                                   "/code_sha256/scripts~1course_experiments~1text.py",
                                   "/code_sha256/scripts~1course_experiments~1common.py",
                                   "/results/data/train", "/results/data/validation", "/results/data/test",
                                   "/results/runs/model/beta", "/results/runs/beta1/beta",
                                   "/results/runs/model/training/steps", "/results/runs/beta1/training/steps",
                                   "/results/runs/model/training/effective_answer_tokens_both_sides",
                                   "/results/runs/beta1/training/effective_answer_tokens_both_sides",
                                   f"/results/runs/model/training/history/{len(measurement['results']['runs']['model']['training']['history']) - 1}/step",
                                   f"/results/runs/beta1/training/history/{len(measurement['results']['runs']['beta1']['training']['history']) - 1}/step"]},
    "dataset_reconstruction": splits,
    "sampling_only_replay": replays,
    "scope": "Arithmetic, tokenization, masking, and deterministic sampler replay only. No weights loaded; no training or model evaluation."
}
(OUT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
